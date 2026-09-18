"""
backend/models/change_detection/vqa_engine.py
Temporal Visual Question Answering (Change VQA) Engine for SatQuery AI (P3 Model).

Integrates a learned Vision-Language Model (Salesforce/blip-image-captioning-base)
to caption T1 vs T2 cropped change regions with remote sensing domain prompting,
with graceful fallback to rule-based spectral analysis if VLM call fails.
"""

import sys
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from schemas import SpatialEvidence, ImageObject

# Module-level model cache (loads once, reuses across calls)
_CACHED_MODELS: Dict[str, Any] = {}


def _get_blip_vlm():
    """
    Loads and caches Salesforce/blip-image-captioning-base processor and model.
    """
    if "blip" in _CACHED_MODELS:
        return _CACHED_MODELS["blip"]

    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration

        model_id = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        _CACHED_MODELS["blip"] = (processor, model, device)
        return _CACHED_MODELS["blip"]
    except Exception as e:
        raise RuntimeError(f"BLIP VLM model initialization failed: {str(e)}")


def _generate_vlm_caption_pair(crop1: Image.Image, crop2: Image.Image, prompt_prefix: str = "A satellite image showing") -> str:
    """
    Generates VLM captions for T1 and T2 cropped change regions with remote sensing domain prompting.
    """
    processor, model, device = _get_blip_vlm()
    import torch

    def caption_single(img: Image.Image) -> str:
        inputs = processor(images=img, text=prompt_prefix, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        text = processor.decode(out[0], skip_special_tokens=True).strip()
        # Clean up prompt prefix if duplicated in output
        if text.lower().startswith(prompt_prefix.lower()):
            text = text[len(prompt_prefix):].strip()
        return text if text else "land surface area"

    cap1 = caption_single(crop1)
    cap2 = caption_single(crop2)
    return f"T1 showed: {cap1}. T2 showed: {cap2}."


class TemporalVQAEngine:
    """
    Temporal Visual Question Answering Engine with VLM Captioning backbone.
    """

    def __init__(self):
        pass

    def answer_query(
        self,
        query: str,
        img_t1: Image.Image,
        img_t2: Image.Image,
        spatial_evidence: List[SpatialEvidence],
        img_obj_t1: ImageObject,
        img_obj_t2: ImageObject,
        changed_pixel_pct: float
    ) -> Tuple[str, bool]:
        """
        Main entry point for Temporal VQA reasoning.
        Returns Tuple of (text_answer, fallback_used).
        """
        query_lower = query.lower()
        bboxes = [ev for ev in spatial_evidence if ev.type == "bbox"]
        num_changes = len(bboxes)

        date_t1 = img_obj_t1.acquisition_date or "T1"
        date_t2 = img_obj_t2.acquisition_date or "T2"

        # 1. Analyze spatial-spectral & VLM captioning properties for each region
        region_analyses, fallback_used = self._analyze_region_features(img_t1, img_t2, bboxes)

        # 2. Determine Query Intent
        intent = self._classify_query_intent(query_lower)

        # 3. Generate tailored response based on intent
        if intent == "counting":
            answer = self._generate_counting_answer(query, num_changes, region_analyses, date_t1, date_t2)
        elif intent == "location":
            answer = self._generate_location_answer(query, num_changes, region_analyses, date_t1, date_t2)
        elif intent == "land_cover":
            answer = self._generate_landcover_answer(query, num_changes, region_analyses, changed_pixel_pct, date_t1, date_t2)
        else:
            answer = self._generate_general_summary_answer(query, num_changes, region_analyses, changed_pixel_pct, date_t1, date_t2)

        return answer, fallback_used

    def _classify_query_intent(self, query: str) -> str:
        if any(w in query for w in ["how many", "count", "number of", "how much count"]):
            return "counting"
        elif any(w in query for w in ["where", "location", "which quadrant", "position", "which region", "located"]):
            return "location"
        elif any(w in query for w in ["water", "flood", "forest", "tree", "vegetation", "building", "structure", "highway", "road", "solar", "scar", "land cover"]):
            return "land_cover"
        else:
            return "general"

    def _analyze_region_features(
        self, img1: Image.Image, img2: Image.Image, bboxes: List[SpatialEvidence]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Extracts spatial quadrant and region descriptions using VLM captioning (with rule-based fallback).
        Returns (analyses, fallback_used).
        """
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        H, W, _ = arr1.shape

        analyses = []
        fallback_used = False

        for idx, bbox in enumerate(bboxes, 1):
            if not bbox.coords:
                continue

            ymin, xmin, ymax, xmax = bbox.coords
            y1, y2 = max(0, int(ymin * H)), min(H, int(ymax * H))
            x1, x2 = max(0, int(xmin * W)), min(W, int(xmax * W))

            # Crop T1 and T2 images for VLM captioning
            crop1_pil = img1.crop((x1, y1, x2, y2))
            crop2_pil = img2.crop((x1, y1, x2, y2))

            # Determine Spatial Quadrant Location
            cy, cx = (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
            if cy < 0.45:
                lat_str = "North"
            elif cy > 0.55:
                lat_str = "South"
            else:
                lat_str = "Central"

            if cx < 0.45:
                lon_str = "West"
            elif cx > 0.55:
                lon_str = "East"
            else:
                lon_str = "Central"

            if lat_str == "Central" and lon_str == "Central":
                quadrant = "Center of the scene"
            else:
                quadrant = f"{lat_str}-{lon_str} quadrant"

            # Try VLM captioning first
            change_description = None
            try:
                change_description = _generate_vlm_caption_pair(crop1_pil, crop2_pil)
            except Exception:
                # Fallback to spectral rule-based heuristic
                fallback_used = True
                crop1 = arr1[y1:y2, x1:x2]
                crop2 = arr2[y1:y2, x1:x2]
                if crop1.size > 0 and crop2.size > 0:
                    mean1 = np.mean(crop1, axis=(0, 1))
                    mean2 = np.mean(crop2, axis=(0, 1))
                    diff_mean = mean2 - mean1
                    r_diff, g_diff, b_diff = diff_mean[0], diff_mean[1], diff_mean[2]

                    if b_diff > 30 and r_diff < -10:
                        change_description = "Water Inundation / Flooding"
                    elif g_diff < -20:
                        change_description = "Vegetation Loss / Canopy Deforestation"
                    elif abs(r_diff - g_diff) < 20 and np.mean(mean2) > np.mean(mean1) + 25:
                        change_description = "New Building / Concrete Structure"
                    elif r_diff > 20:
                        change_description = "Land Clearing / Soil Excavation"
                    else:
                        change_description = "Structural & Land Surface Modification"
                else:
                    change_description = "Surface Change Region"

            analyses.append({
                "index": idx,
                "quadrant": quadrant,
                "coords": bbox.coords,
                "change_type": change_description
            })

        return analyses, fallback_used

    def _generate_counting_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"Based on temporal change analysis between {date_t1} and {date_t2}, no significant changes were detected for query '{query}'."

        descriptions = [f"Region #{a['index']} ({a['quadrant']}): {a['change_type']}" for a in analyses]
        details = "; ".join(descriptions)
        return (
            f"A total of {num_changes} distinct changed region(s) were identified between {date_t1} and {date_t2}. "
            f"Detailed breakdown of detected modifications: {details}."
        )

    def _generate_location_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"No spatial change locations were found between {date_t1} and {date_t2} for query '{query}'."

        loc_descriptions = []
        for a in analyses:
            loc_descriptions.append(f"Region #{a['index']} located in the {a['quadrant']} ({a['change_type']})")

        loc_str = "; ".join(loc_descriptions)
        return (
            f"Detected {num_changes} spatial change zone(s) between {date_t1} and {date_t2}: {loc_str}."
        )

    def _generate_landcover_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], changed_pct: float, date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"No land cover modifications were detected between {date_t1} and {date_t2} relating to '{query}'."

        details = "; ".join([f"Zone #{a['index']} in {a['quadrant']} ({a['change_type']})" for a in analyses])

        return (
            f"Temporal analysis for query '{query}' between {date_t1} and {date_t2} observed approx {changed_pct}% "
            f"area footprint change across {num_changes} localized zone(s). Observed details: {details}."
        )

    def _generate_general_summary_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], changed_pct: float, date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"No visual or structural changes were detected between acquisition dates {date_t1} and {date_t2}."

        details = "; ".join([f"Region #{a['index']} ({a['quadrant']}): {a['change_type']}" for a in analyses])

        return (
            f"Bi-temporal VQA report ({date_t1} -> {date_t2}) for query '{query}': "
            f"Identified {num_changes} primary change region(s) spanning ~{changed_pct}% of the analyzed scene. "
            f"Observed changes: {details}."
        )
