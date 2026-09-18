"""
p3_model.py
P3 Specialist Model: Change Detection & Temporal Analysis for SatQuery AI.

Receives ToolInput from P5 controller and returns ToolOutput adhering to schemas.py.
Handles bi-temporal optical/SAR analysis, spatial evidence generation (bboxes & change masks),
and temporal VQA reasoning.
"""

import os
import math
import time
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFilter
import numpy as np

from schemas import ToolInput, ToolOutput, SpatialEvidence, TaskType, ImageObject


from vqa_engine import TemporalVQAEngine

class P3ChangeDetectionEngine:
    """
    P3 Specialist Engine: Bi-temporal Change Detection & Temporal VQA
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_name = "P3-BiTemporal-DiffEngine-v1.3 (Multi-Spectral + Temporal VQA)"
        self.vqa_engine = TemporalVQAEngine()

    def run(self, tool_input: ToolInput) -> ToolOutput:
        """
        Main entry point called by P5 controller.
        """
        start_time = time.time()

        # Step 1: Input Validation
        if len(tool_input.images) < 2:
            return ToolOutput(
                status="error",
                model_used=self.model_name,
                error_message=f"P3 Change Detection requires at least 2 images (T1 and T2). Provided: {len(tool_input.images)}"
            )

        img_obj_t1 = tool_input.images[0]
        img_obj_t2 = tool_input.images[1]

        if not os.path.exists(img_obj_t1.file_path):
            return ToolOutput(
                status="error",
                model_used=self.model_name,
                error_message=f"Image T1 file path not found: {img_obj_t1.file_path}"
            )
        if not os.path.exists(img_obj_t2.file_path):
            return ToolOutput(
                status="error",
                model_used=self.model_name,
                error_message=f"Image T2 file path not found: {img_obj_t2.file_path}"
            )

        try:
            # Step 2: Load and align images
            pil_t1 = Image.open(img_obj_t1.file_path).convert("RGB")
            pil_t2 = Image.open(img_obj_t2.file_path).convert("RGB")

            # Ensure matching size (resize T1 to T2 if needed)
            if pil_t1.size != pil_t2.size:
                pil_t1 = pil_t1.resize(pil_t2.size, Image.Resampling.BILINEAR)

            width, height = pil_t2.size

            # Step 3: Bi-Temporal Multi-Spectral Change Analysis
            change_mask, diff_score_map = self._compute_change_mask(pil_t1, pil_t2)

            # Step 4: Extract Spatial Evidence (Bounding boxes & saved Mask file)
            spatial_evidence_list, mask_saved_path = self._extract_spatial_evidence(
                change_mask, width, height, tool_input.query
            )

            # Step 5: Render Overlay Visual Evidence
            raw_output_path = self._generate_visual_overlay(
                pil_t2, spatial_evidence_list, change_mask
            )

            # Step 6: Temporal Visual Question Answering (VQA) Answer Generation
            changed_pixel_ratio = float(np.sum(change_mask > 0)) / (width * height)
            changed_pct = round(changed_pixel_ratio * 100, 2)

            text_answer = self.vqa_engine.answer_query(
                query=tool_input.query,
                img_t1=pil_t1,
                img_t2=pil_t2,
                spatial_evidence=spatial_evidence_list,
                img_obj_t1=img_obj_t1,
                img_obj_t2=img_obj_t2,
                changed_pixel_pct=changed_pct
            )

            confidence = min(0.98, max(0.65, 1.0 - (changed_pixel_ratio * 0.4)))

            tool_output = ToolOutput(
                status="success",
                text_answer=text_answer,
                spatial_evidence=spatial_evidence_list,
                confidence=round(confidence, 3),
                raw_output_path=raw_output_path,
                model_used=self.model_name,
                error_message=None
            )

            # Automatically export ToolOutput schema JSON file into output_dir
            json_filename = f"tool_output_{int(start_time)}.json"
            self.export_output_json(tool_output, json_filename)

            return tool_output

        except Exception as e:
            return ToolOutput(
                status="error",
                model_used=self.model_name,
                error_message=f"P3 Change Detection Pipeline execution failed: {str(e)}"
            )

    def export_output_json(self, tool_output: ToolOutput, filename: str = "tool_output.json") -> str:
        """
        Exports a ToolOutput object to a 100% schema-compliant Pydantic JSON file.
        """
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(tool_output.model_dump_json(indent=2))
        return file_path



    def _compute_change_mask(self, img1: Image.Image, img2: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes multi-spectral 3-channel difference between T1 and T2 images using OpenCV & CIELAB.
        Detects color, luminance, and structural changes equally well across all spectral bands.
        Returns precise binary mask (0 or 255) and continuous diff intensity map.
        """
        import cv2

        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Convert to CIELAB color space (L=Lightness, a=Green-Red, b=Blue-Yellow)
        lab1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Gaussian blur each channel to reduce high-frequency sensor noise
        lab1 = cv2.GaussianBlur(lab1, (5, 5), 0)
        lab2 = cv2.GaussianBlur(lab2, (5, 5), 0)

        # Compute Euclidean distance across CIELAB color channels + RGB max channel diff
        diff_lab = np.linalg.norm(lab1 - lab2, axis=2)
        diff_rgb = np.max(np.abs(arr1.astype(np.float32) - arr2.astype(np.float32)), axis=2)

        # Combined multi-spectral difference
        combined_diff = np.maximum(diff_lab, diff_rgb)

        # Automatic Otsu Thresholding for precise change isolation
        diff_uint8 = np.clip(combined_diff, 0, 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological Opening & Closing to eliminate isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        diff_norm = combined_diff / 255.0
        return clean_mask, diff_norm


    def _extract_spatial_evidence(
        self, mask: np.ndarray, width: int, height: int, query: str
    ) -> Tuple[List[SpatialEvidence], str]:
        """
        Extracts exact tight bounding boxes using OpenCV contour analysis.
        Coords format: [ymin, xmin, ymax, xmax] in normalized scale [0.0, 1.0].
        """
        spatial_evidence: List[SpatialEvidence] = []

        # Save precise binary change mask PNG
        mask_filename = f"change_mask_{int(time.time())}.png"
        mask_saved_path = os.path.join(self.output_dir, mask_filename)
        Image.fromarray(mask).save(mask_saved_path)

        # Add binary mask spatial evidence
        spatial_evidence.append(
            SpatialEvidence(
                type="mask",
                mask_path=mask_saved_path,
                label="Binary Change Mask"
            )
        )

        # Extract precise tight bounding boxes via OpenCV contour detection
        boxes = self._find_contour_bboxes(mask, width, height, min_area_pixels=100)

        for i, box in enumerate(boxes):
            spatial_evidence.append(
                SpatialEvidence(
                    type="bbox",
                    coords=box,  # [ymin, xmin, ymax, xmax] normalized
                    label=f"Change Region #{i+1}"
                )
            )

        return spatial_evidence, mask_saved_path

    def _find_contour_bboxes(
        self, mask: np.ndarray, width: int, height: int, min_area_pixels: int = 100
    ) -> List[List[float]]:
        """
        Finds exact, tight bounding boxes around connected change regions using OpenCV contours.
        Returns list of normalized [ymin, xmin, ymax, xmax].
        """
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        # Sort contours by area (largest changes first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_pixels:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            ymin = round(y / float(height), 4)
            xmin = round(x / float(width), 4)
            ymax = round(min(1.0, (y + h) / float(height)), 4)
            xmax = round(min(1.0, (x + w) / float(width)), 4)

            bboxes.append([ymin, xmin, ymax, xmax])

        return bboxes


    def _generate_visual_overlay(
        self, base_img: Image.Image, spatial_evidence: List[SpatialEvidence], mask: np.ndarray
    ) -> str:
        """
        Draws red bounding boxes and semi-transparent red change highlights over T2 image.
        Saves output to raw_output_path.
        """
        overlay = base_img.copy().convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        W, H = base_img.size

        # Draw semi-transparent change highlight
        mask_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = [255, 0, 0, 100]  # Red overlay with alpha=100
        highlight_img = Image.fromarray(mask_rgba, mode="RGBA")
        overlay = Image.alpha_composite(overlay, highlight_img)

        # Draw bounding boxes
        draw_boxes = ImageDraw.Draw(overlay)
        for evidence in spatial_evidence:
            if evidence.type == "bbox" and evidence.coords:
                ymin, xmin, ymax, xmax = evidence.coords
                box_pixels = [xmin * W, ymin * H, xmax * W, ymax * H]
                draw_boxes.rectangle(box_pixels, outline="red", width=3)
                if evidence.label:
                    draw_boxes.text((xmin * W + 4, ymin * H + 4), evidence.label, fill="yellow")

        out_filename = f"change_overlay_{int(time.time())}.png"
        out_path = os.path.join(self.output_dir, out_filename)
        overlay.convert("RGB").save(out_path)
        return out_path
