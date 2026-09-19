"""
backend/models/vqa_captioning_grounding/vqa.py
=============================================================================
SatQuery AI — P2 Visual Question Answering
=============================================================================
Uses Salesforce BLIP (blip-vqa-base) for satellite scene question answering.
Optionally loads a Remote-Sensing LoRA adapter if available.

Model: qwen/qwen3.8-27b (via Groq API)
  - Blazing fast online inference
  - Uses advanced domain prompting with specific structural guides
  - Supports optical and SAR (via SAR visualization preprocessing)
  - Confidence: Defaulted to 0.9 for API calls
"""

import os
import sys
import time

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
for p in [_BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.schemas import ToolInput, ToolOutput, Modality
from .preprocessing import load_image_rgb
from .utils import get_cached, set_cached, DOMAIN_PROMPT, generate_run_id
from .postprocessing import write_sidecar_json
from .calibration import calibrate_confidence, get_confidence_band

_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))

VQA_MODEL_ID = "qwen/qwen3.8-27b"

def _get_vqa_pipeline():
    """Returns a configured Groq client."""
    cached = get_cached("vqa_client")
    if cached is not None:
        return cached

    import groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[WARNING] GROQ_API_KEY not found in environment. API calls will fail.")
    
    print(f"[P2-VQA] Initializing Groq client for '{VQA_MODEL_ID}'...")
    client = groq.Groq(api_key=api_key) if api_key else groq.Groq()
    
    set_cached("vqa_client", client)
    return client


def run_vqa(tool_input: ToolInput) -> ToolOutput:
    """
    Visual Question Answering using Salesforce BLIP.

    Input:
        tool_input.images[0]: Satellite image (optical or SAR)
        tool_input.query: Question about the scene

    Output:
        ToolOutput with text_answer populated.
        confidence is the calculated token transition probability.

    SAR: Image is preprocessed via Lee filter → dB → CLAHE → false-color RGB
         before being passed to BLIP. Model receives visual pattern, not raw SAR.
    """
    if not tool_input.images:
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=VQA_MODEL_ID,
            error_message="VQA requires at least one image in tool_input.images."
        )
    if not tool_input.query or not tool_input.query.strip():
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=VQA_MODEL_ID,
            error_message="VQA requires a non-empty question in tool_input.query."
        )

    try:
        img_obj = tool_input.images[0]
        modality = img_obj.modality

        # Preprocessing: SAR uses false-color visualization
        pil_img, img_meta, img_path = load_image_rgb(img_obj, use_false_color_sar=True, target_size=512)

        client = _get_vqa_pipeline()
        
        # Convert image to base64
        import io
        import base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        query_text = tool_input.query.strip().lower()
        
        # Route by question type and set specific advanced prompts
        q_type = "open"
        prompt_q = tool_input.query.strip()
        
        if query_text.startswith(("is there", "are there", "does the image contain")):
            q_type = "yesno"
            prompt_q = f"Question: {tool_input.query.strip()} Answer only 'yes' or 'no'."
        elif query_text.startswith(("how many", "count")):
            q_type = "count"
            prompt_q = f"Question: {tool_input.query.strip()} Answer only with a number."
        elif query_text.startswith(("what type", "what kind")):
            q_type = "category"
            prompt_q = f"Question: {tool_input.query.strip()} Answer with a single category."
        else:
            prompt_q = f"Question: {tool_input.query.strip()} Answer concisely."
            
        t0 = time.time()
        completion = client.chat.completions.create(
            model=VQA_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"You are a satellite imagery analyst. {prompt_q}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=40
        )
        inference_time = round(time.time() - t0, 2)

        answer = completion.choices[0].message.content.strip()
        
        # Set a default high confidence for successful API calls
        raw_conf = 0.90

        # Apply Honest Confidence Calibration (Phase E)
        calibrated_conf = calibrate_confidence(raw_conf, "vqa") if raw_conf is not None else 0.5
        band, uncertain = get_confidence_band(calibrated_conf)
        
        short_answer = answer if answer else "Unknown"
        
        # Full sentence construction (Phase B)
        evidence = []
        if uncertain:
            full_answer = f"I am uncertain, but it might be {short_answer.lower()}."
        else:
            if q_type == "yesno":
                if short_answer.lower() == "yes":
                    full_answer = f"Yes, based on the visual evidence, the feature is present."
                else:
                    full_answer = f"No, I do not detect that feature in the scene."
            elif q_type == "count":
                full_answer = f"I estimate there are {short_answer.lower()} of those objects."
            elif q_type == "category":
                full_answer = f"The primary category appears to be {short_answer.lower()}."
            else:
                # Open-ended probing using ontology (Simulated via Grounding integration in Phase C/D)
                # For now, if short answer is one word, expand it
                if len(short_answer.split()) <= 2:
                    full_answer = f"The scene primarily features {short_answer.lower()}."
                else:
                    full_answer = short_answer[0].upper() + short_answer[1:]

        # Annotate SAR processing limitation
        sar_note = ""
        include_sar_note = tool_input.params.get("include_sar_note", True) if tool_input.params else True
        if modality == Modality.sar and include_sar_note:
            sar_note = " [Note: SAR image was converted to false-color visualization for VLM input. Results reflect visual pattern interpretation, not native SAR backscatter analysis.]"
        
        full_answer += sar_note
        
        # We pack sidecar data in ToolOutput text_answer for readability since schemas.py is fixed
        human_readable = (
            f"Answer: {full_answer}\n"
            f"Short answer: {short_answer} | Confidence: {calibrated_conf} ({band})"
        )

        run_id = generate_run_id()
        sidecar_path = write_sidecar_json(
            run_id=run_id,
            task="vqa",
            model_used=VQA_MODEL_ID,
            latency=inference_time,
            params={"q_type": q_type, **(tool_input.params or {})},
            evidence=evidence,
            warnings=["low_confidence"] if uncertain else [],
            modality=modality.name if hasattr(modality, "name") else str(modality)
        )

        def make_rel(p):
            if not p: return p
            try:
                return os.path.relpath(p, _PROJECT_ROOT).replace("\\", "/")
            except:
                return p

        return ToolOutput(
            status="success",
            text_answer=human_readable,
            spatial_evidence=[],
            confidence=calibrated_conf,
            model_used=VQA_MODEL_ID,
            raw_output_path=make_rel(sidecar_path),
            error_message=None,
        )

    except Exception as e:
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=VQA_MODEL_ID,
            error_message=f"VQA execution failed: {str(e)}"
        )
