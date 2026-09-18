"""
vqa_engine.py
Temporal Visual Question Answering (Change VQA) Engine for SatQuery AI (P3 Model).

Analyzes user queries, spatial evidence (bounding boxes & change masks), and spectral-spatial
image features to generate precise, query-aware natural language answers.
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from schemas import SpatialEvidence, ImageObject


class TemporalVQAEngine:
    """
    Temporal Visual Question Answering Engine for Bi-Temporal Change Detection
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
    ) -> str:
        """
        Main entry point for Temporal VQA reasoning.
        Returns a concise, accurate, query-aware answer.
        """
        query_lower = query.lower()
        bboxes = [ev for ev in spatial_evidence if ev.type == "bbox"]
        num_changes = len(bboxes)

        date_t1 = img_obj_t1.acquisition_date or "T1"
        date_t2 = img_obj_t2.acquisition_date or "T2"

        # 1. Analyze spatial-spectral properties of each change region
        region_analyses = self._analyze_region_features(img_t1, img_t2, bboxes)

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

        return answer

    def _classify_query_intent(self, query: str) -> str:
        """
        Classifies query intent into: 'counting', 'location', 'land_cover', or 'general'
        """
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
    ) -> List[Dict[str, Any]]:
        """
        Extracts spatial quadrant and spectral change characteristics for each bounding box.
        """
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        H, W, _ = arr1.shape

        analyses = []
        for idx, bbox in enumerate(bboxes, 1):
            if not bbox.coords:
                continue

            ymin, xmin, ymax, xmax = bbox.coords
            y1, y2 = int(ymin * H), int(ymax * H)
            x1, x2 = int(xmin * W), int(xmax * W)

            # Extract crop sub-arrays
            crop1 = arr1[y1:y2, x1:x2]
            crop2 = arr2[y1:y2, x1:x2]

            if crop1.size == 0 or crop2.size == 0:
                continue

            # Spectral means (RGB)
            mean1 = np.mean(crop1, axis=(0, 1))
            mean2 = np.mean(crop2, axis=(0, 1))
            diff_mean = mean2 - mean1  # [R_diff, G_diff, B_diff]

            # Determine Quadrant Location
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

            # Determine Change Category based on Spectral Differences
            r_diff, g_diff, b_diff = diff_mean[0], diff_mean[1], diff_mean[2]

            # Classification Heuristics
            if b_diff > 30 and r_diff < -10 and g_diff < -10:
                change_type = "Water Inundation / Flooding"
            elif g_diff < -20 and (r_diff > 0 or b_diff > 0):
                change_type = "Vegetation Loss / Canopy Deforestation / Burn Scar"
            elif abs(r_diff - g_diff) < 20 and abs(g_diff - b_diff) < 20 and np.mean(mean2) > np.mean(mean1) + 25:
                change_type = "New Building / Concrete Infrastructure / Industrial Structure"
            elif r_diff > 20 and g_diff > 10 and b_diff < 0:
                change_type = "Land Clearing / Soil Excavation / Earthwork"
            elif np.mean(mean2) < np.mean(mean1) - 30:
                change_type = "Burn Scar / Dark Surface Inundation"
            else:
                change_type = "Structural & Land Cover Modification"

            analyses.append({
                "index": idx,
                "quadrant": quadrant,
                "coords": bbox.coords,
                "change_type": change_type,
                "mean1": mean1,
                "mean2": mean2
            })

        return analyses

    def _generate_counting_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"Based on temporal change analysis between {date_t1} and {date_t2}, no significant changes were detected for query '{query}'."

        types_count: Dict[str, int] = {}
        for a in analyses:
            t = a["change_type"]
            types_count[t] = types_count.get(t, 0) + 1

        details = ", ".join([f"{cnt} {t}" for t, cnt in types_count.items()])
        return (
            f"A total of {num_changes} distinct changed region(s) were identified between {date_t1} and {date_t2}. "
            f"Breakdown of detected additions/modifications: {details}."
        )

    def _generate_location_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"No spatial change locations were found between {date_t1} and {date_t2} for query '{query}'."

        loc_descriptions = []
        for a in analyses:
            loc_descriptions.append(f"Region #{a['index']} ({a['change_type']}) located in the {a['quadrant']}")

        loc_str = "; ".join(loc_descriptions)
        return (
            f"Detected {num_changes} spatial change zone(s) between {date_t1} and {date_t2}: {loc_str}."
        )

    def _generate_landcover_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], changed_pct: float, date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"No land cover modifications were detected between {date_t1} and {date_t2} relating to '{query}'."

        categories = list(set([a["change_type"] for a in analyses]))
        cat_str = " and ".join(categories)

        return (
            f"Temporal analysis for query '{query}' between {date_t1} and {date_t2} indicates significant {cat_str}. "
            f"A total area footprint change of approx {changed_pct}% was observed across {num_changes} localized zone(s)."
        )

    def _generate_general_summary_answer(
        self, query: str, num_changes: int, analyses: List[Dict[str, Any]], changed_pct: float, date_t1: str, date_t2: str
    ) -> str:
        if num_changes == 0:
            return f"No visual or structural changes were detected between acquisition dates {date_t1} and {date_t2}."

        locations = list(set([a["quadrant"] for a in analyses]))
        loc_str = ", ".join(locations)
        categories = list(set([a["change_type"] for a in analyses]))
        cat_str = ", ".join(categories)

        return (
            f"Bi-temporal temporal VQA report ({date_t1} -> {date_t2}) for query '{query}': "
            f"Identified {num_changes} primary change region(s) spanning ~{changed_pct}% of the analyzed scene. "
            f"Main changes observed: {cat_str}, located primarily across the {loc_str}."
        )
