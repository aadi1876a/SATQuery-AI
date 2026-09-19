"""
backend/models/vqa_captioning_grounding/grounding.py
=============================================================================
SatQuery AI — P2 Open-Vocabulary Grounding + Segment-wise Segmentation
=============================================================================
Pipeline:
  Image + Text Query
       ↓
  OWLv2 (google/owlv2-base-patch16-ensemble)
       ↓
  Bounding Boxes + Scores
       ↓
  SAM (facebook/sam-vit-base)
       ↓
  Individual Segment Masks
       ↓
  SpatialEvidence[] (bbox + mask per detected object)
       ↓
  ToolOutput

Models:
  - OWLv2: Open-vocabulary object detection. Supports arbitrary text queries.
    NOT native remote-sensing. Receives RGB or SAR visualization.
  - SAM: Segment Anything Model. Produces precise object masks from bbox prompts.
    NOT native remote-sensing.

Confidence: OWLv2 provides detection scores (0.0–1.0) per detected box.
These are real model scores, not fabricated values.
"""

import os
import sys
import time
import json
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
for p in [_BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.schemas import ToolInput, ToolOutput, SpatialEvidence, Modality
from .preprocessing import load_image_rgb
from .postprocessing import save_mask, generate_overlay, save_bboxes_json, validate_mask
from .utils import get_device, get_pytorch, get_cached, set_cached, safe_plural, ensure_dirs

GROUNDING_MODEL_ID = "google/owlv2-base-patch16-ensemble"
SAM_MODEL_ID = "facebook/sam-vit-base"
COMBINED_MODEL_ID = f"{GROUNDING_MODEL_ID} + {SAM_MODEL_ID}"


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _get_grounding_pipeline():
    """Lazy-loads and caches OWLv2."""
    cached = get_cached("grounding")
    if cached is not None:
        return cached

    pytorch = get_pytorch()
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    device = get_device()

    print(f"[P2-Grounding] Loading '{GROUNDING_MODEL_ID}' on {device}...")
    t0 = time.time()
    processor = Owlv2Processor.from_pretrained(GROUNDING_MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(GROUNDING_MODEL_ID).to(device)
    
    # Check for fine-tuned LoRA weights
    lora_path = os.path.join(_CURRENT_DIR, "checkpoints", "p2_grounding_lora")
    if os.path.exists(lora_path):
        try:
            from peft import PeftModel
            print(f"[P2-Grounding] Loading fine-tuned LoRA weights from {lora_path}...")
            model = PeftModel.from_pretrained(model, lora_path)
            # Re-wrap processor if saved
            if os.path.exists(os.path.join(lora_path, "preprocessor_config.json")):
                processor = Owlv2Processor.from_pretrained(lora_path)
        except ImportError:
            print("[P2-Grounding] LoRA weights found but 'peft' is not installed. Ignoring.")

    model.eval()
    load_time = round(time.time() - t0, 2)
    print(f"[P2-Grounding] OWLv2 loaded in {load_time}s")

    # post_process_object_detection lives on Owlv2ImageProcessor in transformers 5.x
    img_proc = processor.image_processor
    result = (processor, img_proc, model, load_time)
    set_cached("grounding", result)
    return result


def _get_sam_pipeline():
    """Lazy-loads and caches SAM."""
    cached = get_cached("sam")
    if cached is not None:
        return cached

    pytorch = get_pytorch()
    from transformers import SamModel, SamProcessor
    device = get_device()

    print(f"[P2-SAM] Loading '{SAM_MODEL_ID}' on {device}...")
    t0 = time.time()
    processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    model.eval()
    load_time = round(time.time() - t0, 2)
    print(f"[P2-SAM] SAM loaded in {load_time}s")

    result = (processor, model, load_time)
    set_cached("sam", result)
    return result


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def _build_search_terms(raw_query: str) -> List[str]:
    """
    Builds an expanded list of search terms from a raw user query.
    Handles aerial/satellite domain variants and corrects plural expansion.
    """
    target = raw_query.lower().strip()

    # Strip common command prefixes
    for prefix in ["show all", "find all", "detect all", "locate all",
                   "show", "find", "detect", "locate"]:
        if target.startswith(prefix + " "):
            target = target[len(prefix):].strip()
            break

    terms = [target]

    # Add plural form (fixing the "roads" → "roadss" bug)
    plural = safe_plural(target)
    if plural != target:
        terms.append(plural)

    # Add satellite/aerial domain variants
    terms.append(f"satellite {target}")
    terms.append(f"aerial view of {target}")

    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique_terms.append(t)

    # Add contrastive distractors to absorb false positives (sand, grass, water, etc)
    distractors = ["sand", "grass", "water", "tree", "road", "bare ground", "forest", "cloud", "shadow"]
    for d in distractors:
        if d not in seen:
            seen.add(d)
            unique_terms.append(d)

    return unique_terms


# ---------------------------------------------------------------------------
# SAM segmentation
# ---------------------------------------------------------------------------

def _run_sam_segmentation(
    pil_img: Image.Image,
    boxes: List[List[float]],
    device: str,
) -> List[Optional[np.ndarray]]:
    """
    Runs SAM segmentation given OWLv2 bounding boxes.

    Returns list of (H, W) boolean masks, one per box.
    Falls back to adaptive region thresholding if SAM fails.
    """
    pytorch = get_pytorch()
    masks = []

    try:
        sam_proc, sam_model, _ = _get_sam_pipeline()

        # SAM expects boxes as list of [x1,y1,x2,y2] per image
        sam_inputs = sam_proc(
            pil_img,
            input_boxes=[[boxes]],  # shape: [1, N, 4] — one image, N boxes
            return_tensors="pt"
        ).to(device)

        with pytorch.no_grad():
            sam_outputs = sam_model(**sam_inputs)

        # transformers 5.x: post_process_masks is on sam_proc.image_processor
        # or directly on sam_proc. Try both.
        _postproc = getattr(sam_proc, "image_processor", sam_proc)
        if not hasattr(_postproc, "post_process_masks"):
            _postproc = sam_proc

        sam_masks = _postproc.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )

        if sam_masks and len(sam_masks) > 0:
            # sam_masks[0] shape: (N_boxes, N_predicted_masks_per_box, H, W)
            pred = sam_masks[0]  # shape: (N, M, H, W)
            for i in range(len(boxes)):
                if i < pred.shape[0]:
                    m_tensor = pred[i, 0]
                    if hasattr(m_tensor, "detach"):
                        m = m_tensor.detach().cpu().numpy().astype(bool)
                    elif hasattr(m_tensor, "numpy"):
                        m = m_tensor.numpy().astype(bool)
                    else:
                        m = np.asarray(m_tensor).astype(bool)
                else:
                    m = None
                masks.append(m)
        else:
            raise ValueError("SAM returned empty masks")

    except Exception as sam_err:
        print(f"[P2-SAM] SAM fallback triggered: {sam_err}")
        # Fallback: adaptive region thresholding inside each detected box
        img_arr = np.array(pil_img.convert("L"))
        h, w = img_arr.shape
        for b in boxes:
            m = np.zeros((h, w), dtype=bool)
            bx1, by1, bx2, by2 = int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))
            if bx2 > bx1 and by2 > by1:
                region = img_arr[by1:by2, bx1:bx2]
                threshold = np.mean(region)
                m[by1:by2, bx1:bx2] = region > threshold
            masks.append(m)

    return masks


# ---------------------------------------------------------------------------
# Main grounding function
# ---------------------------------------------------------------------------

def run_grounding(tool_input: ToolInput) -> ToolOutput:
    """
    Open-vocabulary grounding (OWLv2) + segment-wise segmentation (SAM).

    Input:
        tool_input.images[0]: Satellite image (optical or SAR)
        tool_input.query: Text description of what to locate (e.g. "buildings")
        tool_input.params.get("threshold", 0.10): Detection confidence threshold

    Output:
        ToolOutput with spatial_evidence containing SpatialEvidence objects:
          - Per detected object: SpatialEvidence(type="mask", mask_path=..., label=...)
          - Per detected object: SpatialEvidence(type="bbox", coords=[x1,y1,x2,y2], label=...)

    Confidence:
        Real OWLv2 detection score (average of all detections). True model score.

    SAR:
        Image is preprocessed to grayscale-as-RGB (better for object boundary detection).
        Model receives visual pattern, not raw SAR backscatter.
    """
    if not tool_input.images:
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=COMBINED_MODEL_ID,
            error_message="Grounding requires at least one image."
        )
    if not tool_input.query or not tool_input.query.strip():
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=COMBINED_MODEL_ID,
            error_message="Grounding requires a target query."
        )

    img_obj = tool_input.images[0]
    modality = img_obj.modality
    image_id = img_obj.image_id

    try:
        # For grounding, SAR uses grayscale-as-RGB (better edge definition than false-color)
        pil_img, _ = load_image_rgb(img_obj, use_false_color_sar=False)
        pytorch = get_pytorch()
        processor, img_proc, owlv2_model, _ = _get_grounding_pipeline()
        device = get_device()

        search_terms = _build_search_terms(tool_input.query)
        w, h = pil_img.size

        # OWLv2 inference
        target_sizes = pytorch.tensor([[h, w]], device=device)
        inputs = processor(
            text=[search_terms], images=pil_img, return_tensors="pt"
        ).to(device)

        t0 = time.time()
        with pytorch.no_grad():
            outputs = owlv2_model(**inputs)
        grounding_time = round(time.time() - t0, 2)

        # Post-process: get boxes above threshold
        # In satellite imagery, open-vocabulary VLM scores are typically 0.005-0.05
        user_threshold = tool_input.params.get("threshold")
        threshold = float(user_threshold) if user_threshold is not None else 0.01

        _postproc = (
            img_proc
            if hasattr(img_proc, "post_process_object_detection")
            else processor
        )
        results = _postproc.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )

        # Removed adaptive fallback to prevent false positive massive detections

        boxes, scores, label_indices = [], [], []
        if results and len(results) > 0:
            res = results[0]
            det_boxes = res["boxes"].detach().cpu().tolist()
            det_scores = res["scores"].detach().cpu().tolist()
            det_labels = res["labels"].detach().cpu().tolist()

            raw_detections = []
            for b, sc, lb in zip(det_boxes, det_scores, det_labels):
                x1 = max(0.0, float(b[0]))
                y1 = max(0.0, float(b[1]))
                x2 = min(float(w), float(b[2]))
                y2 = min(float(h), float(b[3]))
                if (x2 - x1) > 4 and (y2 - y1) > 4:
                    raw_detections.append(([x1, y1, x2, y2], round(float(sc), 4), lb))

            # Filter out distractors before taking top-k
            filtered_detections = []
            distractors = ["sand", "grass", "water", "tree", "road", "bare ground", "forest", "cloud", "shadow"]
            for b, sc, lb in raw_detections:
                lbl_text = search_terms[lb] if lb < len(search_terms) else search_terms[0]
                if lbl_text not in distractors:
                    filtered_detections.append((b, sc, lb))

            # Sort by score descending and take top-k (increase max default to 25 for dense areas)
            max_detections = int(tool_input.params.get("max_detections", 25))
            filtered_detections.sort(key=lambda x: x[1], reverse=True)
            top_detections = filtered_detections[:max_detections]

            for b, sc, lb in top_detections:
                boxes.append(b)
                scores.append(sc)
                label_indices.append(lb)

        if not boxes:
            return ToolOutput(
                status="success",
                text_answer=f"No regions detected matching '{tool_input.query}' "
                            f"(threshold={threshold}). Try lowering the threshold or rephrasing the query.",
                spatial_evidence=[],
                confidence=None,
                model_used=COMBINED_MODEL_ID,
                raw_output_path=None,
                error_message=None,
            )

        # Map label indices to term strings
        labels = []
        for lb in label_indices:
            lbl_text = search_terms[lb] if lb < len(search_terms) else search_terms[0]
            labels.append(lbl_text)

        # SAM segmentation
        t1 = time.time()
        masks = _run_sam_segmentation(pil_img, boxes, device)
        seg_time = round(time.time() - t1, 2)

        # Save individual masks + overlays + bbox JSON
        spatial_evidence = []
        for i, (m_arr, b, lbl, sc) in enumerate(zip(masks, boxes, labels, scores)):
            if m_arr is not None:
                m_path = save_mask(m_arr, image_id, lbl, i + 1, modality=str(modality))
                validation = validate_mask(m_path)
                spatial_evidence.append(SpatialEvidence(
                    type="mask", mask_path=m_path, label=lbl
                ))
            spatial_evidence.append(SpatialEvidence(
                type="bbox",
                coords=[round(c, 2) for c in b],
                label=lbl
            ))

        overlay_path = generate_overlay(pil_img, boxes, masks, labels, image_id, modality=str(modality))
        save_bboxes_json(boxes, labels, scores, image_id, modality=str(modality))
        avg_conf = round(sum(scores) / len(scores), 4) if scores else None

        sar_note = ""
        if modality == Modality.sar:
            sar_note = " [SAR preprocessed to grayscale-as-RGB for OWLv2 input. Not native SAR understanding.]"

        return ToolOutput(
            status="success",
            text_answer=f"Detected {len(boxes)} region(s) matching '{tool_input.query}'.{sar_note}",
            spatial_evidence=spatial_evidence,
            confidence=avg_conf,  # Real OWLv2 detection score
            model_used=COMBINED_MODEL_ID,
            raw_output_path=overlay_path,
            error_message=None,
        )

    except Exception as e:
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=COMBINED_MODEL_ID,
            error_message=f"Grounding execution failed: {str(e)}"
        )
