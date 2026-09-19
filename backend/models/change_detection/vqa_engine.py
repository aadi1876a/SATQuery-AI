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
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple
from schemas import SpatialEvidence, ImageObject

# Module-level model cache (loads once, reuses across calls)
_CACHED_MODELS: Dict[str, Any] = {}

# INTERIM FIX: Disable un-tuned general domain BLIP captioning to prevent satellite imagery hallucinations
# (e.g., "mars rover", "plane crash"). Set to True only once fine-tuned on RS image-caption pairs (VRSBench).
ENABLE_BLIP_VLM: bool = False


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

        # Inject region labels back into the spatial evidence mapping for JSON output
        for ev, analysis in zip(bboxes, region_analyses):
            ev.label = analysis["change_type"]

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
        fallback_used = not ENABLE_BLIP_VLM

        for idx, bbox in enumerate(bboxes, 1):
            if not bbox.coords:
                continue

            ymin, xmin, ymax, xmax = bbox.coords
            y1, y2 = max(0, int(ymin * H)), min(H, int(ymax * H))
            x1, x2 = max(0, int(xmin * W)), min(W, int(xmax * W))

            # Crop T1 and T2 images for analysis
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

            # Try VLM captioning if enabled; otherwise fallback to rule-based spectral heuristic
            change_description = None
            if ENABLE_BLIP_VLM:
                try:
                    change_description = _generate_vlm_caption_pair(crop1_pil, crop2_pil)
                except Exception:
                    pass

            if change_description is None:
                # Multi-Domain Spectral & Edge Classifier (Rule-based domain engine)
                fallback_used = True
                crop1 = arr1[y1:y2, x1:x2]
                crop2 = arr2[y1:y2, x1:x2]
                if crop1.size > 0 and crop2.size > 0:
                    eps = 1e-5
                    r1, g1, b1 = crop1[:, :, 0], crop1[:, :, 1], crop1[:, :, 2]
                    r2, g2, b2 = crop2[:, :, 0], crop2[:, :, 1], crop2[:, :, 2]

                    mean_r1, mean_g1, mean_b1 = np.mean(r1), np.mean(g1), np.mean(b1)
                    mean_r2, mean_g2, mean_b2 = np.mean(r2), np.mean(g2), np.mean(b2)

                    lum1 = 0.299 * mean_r1 + 0.587 * mean_g1 + 0.114 * mean_b1
                    lum2 = 0.299 * mean_r2 + 0.587 * mean_g2 + 0.114 * mean_b2
                    d_lum = lum2 - lum1

                    # NDVI Greenness surrogates
                    ndvi1 = (mean_g1 - mean_r1) / (mean_g1 + mean_r1 + eps)
                    ndvi2 = (mean_g2 - mean_r2) / (mean_g2 + mean_r2 + eps)
                    d_ndvi = ndvi2 - ndvi1

                    # NDWI Water surrogates
                    ndwi1 = (mean_b1 - mean_r1) / (mean_b1 + mean_r1 + eps)
                    ndwi2 = (mean_b2 - mean_r2) / (mean_b2 + mean_r2 + eps)
                    d_ndwi = ndwi2 - ndwi1

                    # Edge Density Delta (Structural/Building indicator)
                    u_crop1 = np.clip(crop1, 0, 255).astype(np.uint8)
                    u_crop2 = np.clip(crop2, 0, 255).astype(np.uint8)
                    gray1 = cv2.cvtColor(u_crop1, cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(u_crop2, cv2.COLOR_RGB2GRAY)
                    edge1 = np.mean(cv2.Canny(gray1, 40, 120))
                    edge2 = np.mean(cv2.Canny(gray2, 40, 120))
                    d_edge = edge2 - edge1

                    # Color Neutrality / Grayness metric (for built-up structures)
                    neutrality2 = abs(mean_r2 - mean_g2) + abs(mean_g2 - mean_b2) + abs(mean_b2 - mean_r2)

                    # Domain classification logic: Evaluate dominant spectral and structural signals
                    # 1. Water Dynamics (Inundation vs. Recession)
                    # Covers Standard Dark Water, Rayleigh Shadow rejections, and shallow Cyan/Turquoise water
                    if (d_ndwi > 0.15 and d_lum < 0) or (ndwi2 > 0.08 and mean_b2 > 35 and d_lum < -5 and ndvi2 < 0.25) or (ndwi2 > 0.1 and mean_b2 > 45) or (mean_b1 > 45 and d_ndwi < -0.02) or (ndwi2 > 0.03 and d_ndwi > 0.02 and d_lum > 5):
                        if d_ndwi < -0.02:
                            change_description = "Water Body Recession / Lake Shrinkage"
                        else:
                            change_description = "Water Inundation / Surface Flooding"

                    # 3. Significant Vegetation Loss / Deforestation (Prioritized if NDVI drop is prominent)
                    elif d_ndvi < -0.07 or (mean_g1 - mean_g2 > 12 and d_ndvi < -0.03):
                        if (d_lum > 15 and neutrality2 < 35) or d_edge > 4.0 or lum2 > 180:
                            change_description = "New Building / Concrete Structure"
                        elif d_ndwi > 0.15 and d_lum < 0:
                            change_description = "Water Inundation / Surface Flooding"
                        else:
                            change_description = "Vegetation Loss / Crop Harvesting & Deforestation"

                    # 4. Significant Vegetation Growth / Reforestation
                    elif d_ndvi > 0.07 or (mean_g2 - mean_g1 > 12 and d_ndvi > 0.03):
                        change_description = "Vegetation Growth / Crop Canopy & Reforestation"

                    # 5. New Building / Concrete / Urban Structure
                    elif d_edge > 3.0 or (d_lum > 10 and neutrality2 < 35) or (d_lum > 20 and neutrality2 < 45):
                        change_description = "New Building / Concrete Structure"

                    # 6. Land Clearing / Soil Excavation
                    elif mean_r2 - mean_r1 > 12 or (d_lum > 15 and d_ndvi < 0):
                        change_description = "Land Clearing / Bare Soil Excavation"

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
            return f"No change detected. Temporal analysis between {date_t1} and {date_t2} confirmed exactly zero structural or surface modifications."

        unique_types = list(set([a["change_type"].split(" / ")[0] for a in analyses]))
        summary_types = ", ".join(unique_types)
        
        descriptions = [f"• Region #{a['index']} ({a['quadrant']}): {a['change_type']}" for a in analyses]
        details = "\n".join(descriptions)
        return (
            f"Bi-Temporal Change Analysis Report ({date_t1} -> {date_t2}):\n"
            f"Generated a count of {num_changes} distinct change region(s), involving: {summary_types}.\n\n"
            f"Detected Zone Breakdown:\n{details}"
        )

    def _generate_location_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"No change detected. Spatial analysis between {date_t1} and {date_t2} found zero modified locations."

        unique_types = list(set([a["change_type"].split(" / ")[0] for a in analyses]))
        summary_types = ", ".join(unique_types)
        
        loc_descriptions = [f"• Region #{a['index']} in {a['quadrant']}: {a['change_type']}" for a in analyses]
        loc_str = "\n".join(loc_descriptions)
        return (
            f"Bi-Temporal Spatial Location Report ({date_t1} -> {date_t2}):\n"
            f"Identified {num_changes} spatial change zone(s) consisting of {summary_types}:\n{loc_str}"
        )

    def _generate_landcover_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], changed_pct: float, date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"No change detected. Temporal land cover analysis between {date_t1} and {date_t2} shows zero modifications."

        unique_types = list(set([a["change_type"].split(" / ")[0] for a in analyses]))
        summary_types = ", ".join(unique_types)
        details = "\n".join([f"• Zone #{a['index']} ({a['quadrant']}): {a['change_type']}" for a in analyses])

        return (
            f"Bi-Temporal Land Cover Report ({date_t1} -> {date_t2}):\n"
            f"• Dominant Scene Changes: {summary_types}\n"
            f"• Total Changed Footprint: ~{changed_pct}% of scene area across {num_changes} zone(s)\n\n"
            f"Observed Modifications:\n{details}"
        )

    def _generate_general_summary_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], changed_pct: float, date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"No change detected. Bi-Temporal Summary ({date_t1} -> {date_t2}): No visual or structural surface changes found."

        unique_types = list(set([a["change_type"].split(" / ")[0] for a in analyses]))
        summary_types = ", ".join(unique_types)
        details = "\n".join([f"• Region #{a['index']} ({a['quadrant']}): {a['change_type']}" for a in analyses])

        return (
            f"Bi-Temporal VQA Summary ({date_t1} -> {date_t2}):\n"
            f"• Primary Modifications Detected: {summary_types}\n"
            f"• Scene Change Footprint: ~{changed_pct}% of total image area\n"
            f"• Total Change Clusters: {num_changes} primary region(s)\n\n"
            f"Observed Changes:\n{details}"
        )
