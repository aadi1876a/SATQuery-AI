"""
backend/models/vqa_captioning_grounding/captioning.py
=============================================================================
SatQuery AI — P2 Image Captioning
=============================================================================
Uses Salesforce BLIP (blip-image-captioning-base) for satellite scene description.

Model: Salesforce/blip-image-captioning-base
  - Pretrained on COCO, Conceptual Captions
  - NOT natively fine-tuned on satellite imagery
  - Domain prompting applied: "A satellite aerial view showing"
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

from backend.schemas import ToolInput, ToolOutput, Modality
from .preprocessing import load_image_rgb
from .utils import get_device, get_pytorch, get_cached, set_cached

CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
RS_LORA_DIR = os.path.join(_CURRENT_DIR, "checkpoints", "p2_rs_lora")


def _get_caption_pipeline():
    """Lazy-loads and caches the captioning model."""
    cached = get_cached("caption")
    if cached is not None:
        return cached

    pytorch = get_pytorch()
    from transformers import BlipProcessor, BlipForConditionalGeneration
    device = get_device()
    model_name = CAPTION_MODEL_ID

    print(f"[P2-Caption] Loading '{CAPTION_MODEL_ID}' on {device}...")
    t0 = time.time()
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
    base_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID).to(device)

    # Check for Remote-Sensing LoRA checkpoint
    if os.path.exists(RS_LORA_DIR) and any(
        f.startswith("adapter_") for f in os.listdir(RS_LORA_DIR)
    ):
        try:
            from peft import PeftModel
            print(f"[P2-Caption] Attaching RS-LoRA from '{RS_LORA_DIR}'...")
            model = PeftModel.from_pretrained(base_model, RS_LORA_DIR)
            model_name = f"{CAPTION_MODEL_ID} + RS-LoRA"
        except Exception as e:
            print(f"[P2-Caption] LoRA attach failed ({e}). Using base model.")
            model = base_model
    else:
        model = base_model

    model.eval()
    load_time = round(time.time() - t0, 2)
    print(f"[P2-Caption] Model loaded in {load_time}s")

    result = (processor, model, model_name, load_time)
    set_cached("caption", result)
    return result


def run_captioning(tool_input: ToolInput) -> ToolOutput:
    """
    Image captioning using Salesforce BLIP.

    Input:
        tool_input.images[0]: Satellite image (optical or SAR)
        tool_input.query: Optional prompt prefix (default: "A satellite aerial view showing")

    Output:
        ToolOutput with text_answer containing the caption.
        confidence is None (BLIP does not provide caption confidence scores).

    SAR: Image is preprocessed via Lee filter → dB → CLAHE → false-color RGB
         before being passed to BLIP. Caption reflects visual interpretation only.
    """
    if not tool_input.images:
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=CAPTION_MODEL_ID,
            error_message="Captioning requires at least one image in tool_input.images."
        )

    try:
        img_obj = tool_input.images[0]
        modality = img_obj.modality

        # Preprocessing: SAR uses false-color visualization
        pil_img, _ = load_image_rgb(img_obj, use_false_color_sar=True)

        processor, model, model_name, _ = _get_caption_pipeline()
        device = get_device()
        pytorch = get_pytorch()

        # Domain prompt: use provided query or default satellite context
        prompt = (
            tool_input.query.strip()
            if tool_input.query and tool_input.query.strip()
            else "A satellite aerial view showing"
        )

        inputs = processor(pil_img, text=prompt, return_tensors="pt").to(device)

        t0 = time.time()
        with pytorch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True
            )
        inference_time = round(time.time() - t0, 2)

        seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
        caption = processor.decode(seq, skip_special_tokens=True).strip()

        if caption:
            caption = caption[0].upper() + caption[1:]
            if not caption.endswith("."):
                caption += "."
        else:
            caption = "A remote sensing satellite image."

        # Calculate true sequence generation confidence from length-normalized per-token probabilities
        conf = None
        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            # Length-normalized per-token geometric mean probability: exp(sequence_log_prob / num_tokens)
            # Prevents joint probability from mathematically decaying to near-zero on longer sentences
            num_tokens = max(1, len(seq) - 1)
            token_log_prob = out.sequences_scores[0].item() / num_tokens
            conf = round(float(pytorch.exp(pytorch.tensor(token_log_prob)).item()), 3)
            conf = max(0.01, min(1.0, conf))
        elif hasattr(out, "scores") and out.scores:
            probs = [pytorch.softmax(s, dim=-1).max().item() for s in out.scores]
            conf = round(float(sum(probs) / len(probs)), 3) if probs else None

        # Annotate SAR processing limitation
        if modality == Modality.sar:
            caption += " [Note: SAR image converted to false-color visualization for VLM input.]"

        return ToolOutput(
            status="success",
            text_answer=caption,
            spatial_evidence=[],
            confidence=conf,  # Real model generation confidence
            model_used=model_name,
            raw_output_path=None,
            error_message=None,
        )

    except Exception as e:
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=CAPTION_MODEL_ID,
            error_message=f"Captioning execution failed: {str(e)}"
        )
