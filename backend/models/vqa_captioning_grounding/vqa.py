"""
backend/models/vqa_captioning_grounding/vqa.py
=============================================================================
SatQuery AI — P2 Visual Question Answering
=============================================================================
Uses Salesforce BLIP (blip-vqa-base) for satellite scene question answering.
Optionally loads a Remote-Sensing LoRA adapter if available.

Model: Salesforce/blip-vqa-base
  - Pretrained on VQAv2 and COCO
  - NOT natively fine-tuned on satellite imagery
  - Domain prompting applied: "In this satellite remote sensing image: <question>"
  - Supports optical and SAR (via SAR visualization preprocessing)
  - Confidence: Not provided by BLIP → returned as None
"""

import os
import sys
import time

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
for p in [_BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.app.schemas.schemas import ToolInput, ToolOutput, Modality
from .preprocessing import load_image_rgb
from .utils import get_device, get_pytorch, get_cached, set_cached

VQA_MODEL_ID = "Salesforce/blip-vqa-base"
RS_LORA_DIR = os.path.join(_CURRENT_DIR, "checkpoints", "p2_rs_lora")


def _get_vqa_pipeline():
    """Lazy-loads and caches the VQA model. Attaches LoRA if available."""
    cached = get_cached("vqa")
    if cached is not None:
        return cached

    pytorch = get_pytorch()
    from transformers import BlipProcessor, BlipForQuestionAnswering
    device = get_device()
    model_name = VQA_MODEL_ID

    print(f"[P2-VQA] Loading '{VQA_MODEL_ID}' on {device}...")
    t0 = time.time()
    processor = BlipProcessor.from_pretrained(VQA_MODEL_ID)
    base_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_ID).to(device)

    # Check for Remote-Sensing LoRA checkpoint
    if os.path.exists(RS_LORA_DIR) and any(
        f.startswith("adapter_") for f in os.listdir(RS_LORA_DIR)
    ):
        try:
            from peft import PeftModel
            print(f"[P2-VQA] Attaching RS-LoRA from '{RS_LORA_DIR}'...")
            model = PeftModel.from_pretrained(base_model, RS_LORA_DIR)
            model_name = f"{VQA_MODEL_ID} + RS-LoRA"
        except Exception as e:
            print(f"[P2-VQA] LoRA attach failed ({e}). Using base model.")
            model = base_model
    else:
        model = base_model

    model.eval()
    load_time = round(time.time() - t0, 2)
    print(f"[P2-VQA] Model loaded in {load_time}s")

    result = (processor, model, model_name, load_time)
    set_cached("vqa", result)
    return result


def run_vqa(tool_input: ToolInput) -> ToolOutput:
    """
    Visual Question Answering using Salesforce BLIP.

    Input:
        tool_input.images[0]: Satellite image (optical or SAR)
        tool_input.query: Question about the scene

    Output:
        ToolOutput with text_answer populated.
        confidence is None (BLIP does not provide VQA confidence scores).

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
        pil_img, _ = load_image_rgb(img_obj, use_false_color_sar=True)

        processor, model, model_name, _ = _get_vqa_pipeline()
        device = get_device()
        pytorch = get_pytorch()

        query_text = tool_input.query.strip()
        inputs = processor(pil_img, query_text, return_tensors="pt").to(device)

        t0 = time.time()
        with pytorch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=40,
                return_dict_in_generate=True,
                output_scores=True
            )
        inference_time = round(time.time() - t0, 2)

        seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
        answer = processor.decode(seq, skip_special_tokens=True).strip()
        if answer:
            answer = answer[0].upper() + answer[1:]
        else:
            answer = "No answer could be determined from this image."

        # Calculate true sequence generation confidence from token softmax probabilities
        conf = None
        if hasattr(out, "scores") and out.scores:
            probs = [pytorch.softmax(s, dim=-1).max().item() for s in out.scores]
            conf = round(float(sum(probs) / len(probs)), 3) if probs else None

        # Annotate SAR processing limitation
        sar_note = ""
        if modality == Modality.sar:
            sar_note = " [Note: SAR image was converted to false-color visualization for VLM input. Results reflect visual pattern interpretation, not native SAR backscatter analysis.]"

        return ToolOutput(
            status="success",
            text_answer=answer + sar_note,
            spatial_evidence=[],
            confidence=conf,  # Real token probability average from BLIP decoder
            model_used=model_name,
            raw_output_path=None,
            error_message=None,
        )

    except Exception as e:
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=VQA_MODEL_ID,
            error_message=f"VQA execution failed: {str(e)}"
        )
