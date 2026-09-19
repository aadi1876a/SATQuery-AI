"""
backend/models/vqa_captioning_grounding/grounding.py
=============================================================================
SatQuery AI — P2 Visual Grounding
=============================================================================
Uses Groq API (qwen/qwen3.8-27b) for visual grounding instead of local models.
Outputs bounding boxes. SAM segmentation is removed to keep it lightweight.
"""

import os
import sys
import time
import json
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for p in [_PROJECT_ROOT, _BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.schemas import ToolInput, ToolOutput, SpatialEvidence, Modality
from .preprocessing import load_image_rgb
from .postprocessing import generate_overlay, generate_panel, save_bboxes_json, write_sidecar_json
from .utils import get_cached, set_cached, generate_run_id
from .calibration import get_class_threshold, calibrate_confidence, get_confidence_band

GROUNDING_MODEL_ID = "qwen/qwen3.8-27b"

def _get_grounding_pipeline():
    cached = get_cached("grounding_client")
    if cached is not None:
        return cached

    import groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[WARNING] GROQ_API_KEY not found in environment. API calls will fail.")
    
    print(f"[P2-Grounding] Initializing Groq client for '{GROUNDING_MODEL_ID}'...")
    client = groq.Groq(api_key=api_key) if api_key else groq.Groq()
    set_cached("grounding_client", client)
    return client

def run_grounding(tool_input: ToolInput) -> ToolOutput:
    if not tool_input.images:
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=GROUNDING_MODEL_ID,
            error_message="Grounding requires at least one image."
        )
    if not tool_input.query or not tool_input.query.strip():
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=GROUNDING_MODEL_ID,
            error_message="Grounding requires a target query."
        )

    img_obj = tool_input.images[0]
    modality = img_obj.modality
    image_id = img_obj.image_id
    
    def make_rel(p):
        if not p: return p
        try:
            return os.path.relpath(p, _PROJECT_ROOT).replace("\\", "/")
        except:
            return p

    try:
        # Load image for API
        pil_img, img_meta, img_path = load_image_rgb(img_obj, use_false_color_sar=True, target_size=960)
        
        client = _get_grounding_pipeline()
        
        import io
        import base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        w, h = pil_img.size
        
        prompt = f"You are a visual grounding model. Detect all '{tool_input.query.strip()}' in this image. Return ONLY the bounding boxes in the format: <box>(ymin,xmin),(ymax,xmax)</box> where coordinates are scaled from 0 to 1000. Do not output anything else."

        t0 = time.time()
        completion = client.chat.completions.create(
            model=GROUNDING_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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
            max_tokens=200
        )
        grounding_time = round(time.time() - t0, 2)
        
        response_text = completion.choices[0].message.content.strip()
        
        # Parse bounding boxes
        boxes = []
        scores = []
        labels = []
        
        # Qwen-VL box format: <box>(ymin,xmin),(ymax,xmax)</box>
        pattern = r"<box>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*</box>"
        matches = re.findall(pattern, response_text)
        
        for match in matches:
            ymin, xmin, ymax, xmax = map(int, match)
            # Scale coordinates from 0-1000 to image size
            x1 = max(0.0, (xmin / 1000.0) * w)
            y1 = max(0.0, (ymin / 1000.0) * h)
            x2 = min(float(w), (xmax / 1000.0) * w)
            y2 = min(float(h), (ymax / 1000.0) * h)
            
            # Ensure valid box
            if (x2 - x1) > 2 and (y2 - y1) > 2:
                boxes.append([x1, y1, x2, y2])
                scores.append(0.9)  # Default high score for API detection
                labels.append(tool_input.query.strip())
                
        # Limit to max detections
        max_detections = int(tool_input.params.get("top_k", tool_input.params.get("max_detections", 5)))
        boxes = boxes[:max_detections]
        scores = scores[:max_detections]
        labels = labels[:max_detections]

        if not boxes:
            return ToolOutput(
                status="success",
                text_answer=f"No regions detected matching '{tool_input.query}'.",
                spatial_evidence=[],
                confidence=None,
                model_used=GROUNDING_MODEL_ID,
                raw_output_path=None,
                error_message=None,
            )

        # No masks for online API pipeline to keep it lightweight
        masks = [None] * len(boxes)

        spatial_evidence = []
        for i, (b, lbl, sc) in enumerate(zip(boxes, labels, scores)):
            spatial_evidence.append(SpatialEvidence(
                type="bbox",
                coords=[round(c, 2) for c in b],
                label=lbl
            ))

        # Postprocessing: Upscaled Overlay & Panel (Phase D)
        overlay_path = generate_overlay(pil_img, boxes, masks, labels, scores, image_id, modality=str(modality))
        panel_path = generate_panel(pil_img, overlay_path, masks, image_id, modality=str(modality))
        save_bboxes_json(boxes, labels, scores, image_id, modality=str(modality))
        
        raw_conf = 0.90
        calibrated_conf = calibrate_confidence(raw_conf, "grounding")
        band, uncertain = get_confidence_band(calibrated_conf)

        sar_note = ""
        include_sar_note = tool_input.params.get("include_sar_note", True) if tool_input.params else True
        if modality == Modality.sar and include_sar_note:
            sar_note = " [Note: SAR preprocessed to grayscale-as-RGB for VLM input.]"
            
        ans_text = f"Detected {len(boxes)} region(s) matching '{tool_input.query}'."
            
        human_readable = (
            f"Result: {ans_text}\n"
            f"Confidence: {calibrated_conf} ({band})"
            f"{sar_note}"
        )

        run_id = generate_run_id()
        sidecar_path = write_sidecar_json(
            run_id=run_id,
            task="grounding",
            model_used=GROUNDING_MODEL_ID,
            latency=grounding_time,
            params=tool_input.params or {},
            evidence=[{"type": "bbox", "coords": b, "label": l, "score": s} for b, l, s in zip(boxes, labels, scores)],
            warnings=["low_confidence"] if uncertain else [],
            modality=modality.name if hasattr(modality, "name") else str(modality)
        )

        return ToolOutput(
            status="success",
            text_answer=human_readable,
            spatial_evidence=spatial_evidence,
            confidence=calibrated_conf,
            model_used=GROUNDING_MODEL_ID,
            raw_output_path=make_rel(sidecar_path),
            error_message=None,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ToolOutput(
            status="error", text_answer=None, spatial_evidence=[],
            confidence=None, model_used=GROUNDING_MODEL_ID,
            error_message=f"Grounding execution failed: {str(e)}"
        )
