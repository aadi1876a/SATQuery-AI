"""
backend/models/change_detection/change_engine.py
P3 Specialist Engine: Bi-temporal Change Detection & Temporal VQA

Core internal class implementation for SatQuery AI P3 model.
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Tuple, Optional

# Ensure repository root is in sys.path for importing schemas
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from backend.app.schemas.schemas import ToolInput, ToolOutput, SpatialEvidence, TaskType, ImageObject
from .vqa_engine import TemporalVQAEngine


class P3ChangeDetectionEngine:
    """
    P3 Specialist Engine: Bi-temporal Change Detection & Learned VLM Temporal VQA
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Also ensure 'outputs' exists for backwards compatibility
        os.makedirs("outputs", exist_ok=True)
        self.base_model_name = "P3-BiTemporal-DiffEngine-v1.3 (Multi-Spectral + BLIP VLM VQA)"
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
                model_used=self.base_model_name,
                error_message=f"P3 Change Detection requires at least 2 images (T1 and T2). Provided: {len(tool_input.images)}"
            )

        img_obj_t1 = tool_input.images[0]
        img_obj_t2 = tool_input.images[1]

        if not os.path.exists(img_obj_t1.file_path):
            return ToolOutput(
                status="error",
                model_used=self.base_model_name,
                error_message=f"Image T1 file path not found: {img_obj_t1.file_path}"
            )
        if not os.path.exists(img_obj_t2.file_path):
            return ToolOutput(
                status="error",
                model_used=self.base_model_name,
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

            text_answer, fallback_used = self.vqa_engine.answer_query(
                query=tool_input.query,
                img_t1=pil_t1,
                img_t2=pil_t2,
                spatial_evidence=spatial_evidence_list,
                img_obj_t1=img_obj_t1,
                img_obj_t2=img_obj_t2,
                changed_pixel_pct=changed_pct
            )

            model_name = self.base_model_name
            # Robust Signal-to-Noise Ratio (SNR) & Contrast-based Confidence Calculation
            if np.any(change_mask > 0):
                fg_diff = float(np.mean(diff_score_map[change_mask > 0]))
                bg_diff = float(np.mean(diff_score_map[change_mask == 0])) + 1.0
                snr_ratio = max(0.0, (fg_diff - bg_diff) / bg_diff)
                confidence = 0.86 + min(0.11, (snr_ratio / 2.5) * 0.11)
            else:
                confidence = 0.95

            confidence = round(float(confidence), 3)

            tool_output = ToolOutput(
                status="success",
                text_answer=text_answer,
                spatial_evidence=spatial_evidence_list,
                confidence=round(confidence, 3),
                raw_output_path=raw_output_path,
                model_used=model_name,
                error_message=None
            )

            # Automatically export ToolOutput schema JSON file into output_dir
            json_filename = f"tool_output_{int(start_time)}.json"
            self.export_output_json(tool_output, json_filename)

            return tool_output

        except Exception as e:
            return ToolOutput(
                status="error",
                model_used=self.base_model_name,
                error_message=f"P3 Change Detection Pipeline execution failed: {str(e)}"
            )

    def export_output_json(self, tool_output: ToolOutput, filename: str = "tool_output.json") -> str:
        """
        Exports a ToolOutput object to a 100% schema-compliant Pydantic JSON file.
        """
        run_dir = os.path.dirname(tool_output.raw_output_path) if tool_output.raw_output_path else self.output_dir
        file_path = os.path.join(run_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(tool_output.model_dump_json(indent=2))
        return file_path

    def _compute_change_mask(self, img1: Image.Image, img2: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes multi-spectral composite difference map across all land-cover domains:
        - Greenery/Vegetation (NDVI-Surrogate)
        - Water Bodies / Flooding (NDWI-Surrogate)
        - Building / Concrete / Urban Construction (Luminance L* & Structural Edge Density)
        - Soil / Land Clearing (Redness/Bare-Earth Index)
        - Color/Hue distance (CIELAB ΔE)
        """
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        # Convert to uint8 for OpenCV color conversions & edge detection
        u1 = arr1.astype(np.uint8)
        u2 = arr2.astype(np.uint8)

        # 1. CIELAB Color Space Distance
        lab1 = cv2.cvtColor(u1, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(u2, cv2.COLOR_RGB2LAB).astype(np.float32)
        cielab_dist = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
        cielab_norm = np.clip(cielab_dist * (255.0 / 120.0), 0, 255)

        # 2. Luminance / Reflectance Delta (L* channel: sensitive to building/concrete/asphalt)
        lum_diff = np.abs(lab1[:, :, 0] - lab2[:, :, 0])
        lum_norm = np.clip(lum_diff * (255.0 / 80.0), 0, 255)

        # 3. Greenery / Vegetation Index (NDVI Surrogate: (G - R) / (G + R + eps))
        eps = 1e-5
        r1, g1, b1 = arr1[:, :, 0], arr1[:, :, 1], arr1[:, :, 2]
        r2, g2, b2 = arr2[:, :, 0], arr2[:, :, 1], arr2[:, :, 2]

        ndvi1 = (g1 - r1) / (g1 + r1 + eps)
        ndvi2 = (g2 - r2) / (g2 + r2 + eps)
        ndvi_diff = np.abs(ndvi2 - ndvi1)
        ndvi_norm = np.clip(ndvi_diff * 255.0, 0, 255)

        # 4. Water Level / Inundation Index (NDWI Surrogate: (B - R) / (B + R + eps))
        ndwi1 = (b1 - r1) / (b1 + r1 + eps)
        ndwi2 = (b2 - r2) / (b2 + r2 + eps)
        ndwi_diff = np.abs(ndwi2 - ndwi1)
        ndwi_norm = np.clip(ndwi_diff * 255.0, 0, 255)

        # 5. Structural Edge Density Difference (Sobel magnitude for building outlines & infrastructure)
        gray1 = cv2.cvtColor(u1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(u2, cv2.COLOR_RGB2GRAY)
        sobel1 = np.hypot(cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3))
        sobel2 = np.hypot(cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3))
        edge_diff = np.abs(sobel2 - sobel1)
        edge_norm = np.clip(edge_diff * 1.5, 0, 255)

        # Multi-feature max/weighted composite fusion
        composite_diff = np.maximum.reduce([
            cielab_norm * 0.5,
            lum_norm * 0.7,
            ndvi_norm * 0.8,
            ndwi_norm * 0.8,
            edge_norm * 0.6
        ])

        blurred_diff = cv2.GaussianBlur(composite_diff.astype(np.float32), (5, 5), 0)
        diff_uint8 = np.clip(blurred_diff, 0, 255).astype(np.uint8)

        # Noise floor guard: if maximum composite difference across scene is below noise threshold, return zero mask
        max_diff = float(np.max(diff_uint8))
        mean_val = float(np.mean(diff_uint8))
        std_val = float(np.std(diff_uint8))

        if max_diff < 16.0 or (mean_val < 3.5 and std_val < 3.5):
            return np.zeros_like(diff_uint8), blurred_diff

        # Robust adaptive thresholding: blend Otsu threshold with statistical mean + std threshold
        _, otsu_th = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        stat_threshold = min(200.0, max(28.0, mean_val + 1.1 * std_val))

        otsu_val = _
        final_threshold = min(otsu_val, stat_threshold)
        if final_threshold < 20:
            final_threshold = 20

        _, binary_mask = cv2.threshold(diff_uint8, int(final_threshold), 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return clean_mask, blurred_diff

    def _extract_spatial_evidence(
        self, change_mask: np.ndarray, width: int, height: int, query: str
    ) -> Tuple[List[SpatialEvidence], str]:
        """
        Extracts spatial evidence (binary mask path + tight bounding boxes).
        """
        evidence_list: List[SpatialEvidence] = []
        timestamp = int(time.time())
        run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Save Binary Mask
        mask_filename = f"change_mask_{timestamp}.png"
        mask_filepath = os.path.join(run_dir, mask_filename)
        mask_pil = Image.fromarray(change_mask)
        mask_pil.save(mask_filepath)

        evidence_list.append(
            SpatialEvidence(
                type="mask",
                coords=None,
                mask_path=mask_filepath,
                label="Binary Change Mask"
            )
        )

        # OpenCV Contour extraction for tight bounding boxes
        contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_px = (width * height) * 0.0015  # 0.15% minimum area threshold

        box_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area_px:
                x, y, w, h = cv2.boundingRect(cnt)
                ymin = round(float(y) / height, 4)
                xmin = round(float(x) / width, 4)
                ymax = round(float(y + h) / height, 4)
                xmax = round(float(x + w) / width, 4)

                box_count += 1
                evidence_list.append(
                    SpatialEvidence(
                        type="bbox",
                        coords=[ymin, xmin, ymax, xmax],
                        mask_path=None,
                        label=f"Change Region #{box_count}"
                    )
                )

        return evidence_list, mask_filepath

    def _generate_visual_overlay(
        self, base_img: Image.Image, spatial_evidence: List[SpatialEvidence], change_mask: np.ndarray
    ) -> str:
        """
        Renders red bounding box overlay and red semi-transparent change highlights.
        """
        overlay = base_img.copy().convert("RGBA")
        red_tint = Image.new("RGBA", base_img.size, (255, 0, 0, 90))

        mask_pil = Image.fromarray(change_mask).convert("L")
        overlay = Image.composite(red_tint, overlay, mask_pil)

        draw = ImageDraw.Draw(overlay)
        width, height = base_img.size

        bboxes = [ev for ev in spatial_evidence if ev.type == "bbox"]
        for ev in bboxes:
            if ev.coords and len(ev.coords) == 4:
                ymin, xmin, ymax, xmax = ev.coords
                y1, y2 = int(ymin * height), int(ymax * height)
                x1, x2 = int(xmin * width), int(xmax * width)

                # Draw bold red bounding box
                for offset in range(3):
                    draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=(255, 0, 0, 255))

                # Draw label banner
                banner_text = f" {ev.label} "
                draw.rectangle([x1, max(0, y1 - 20), x1 + 140, y1], fill=(255, 0, 0, 230))
                draw.text((x1 + 4, max(0, y1 - 18)), banner_text, fill=(255, 255, 255, 255))

        timestamp = int(time.time())
        run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        overlay_filename = f"change_overlay_{timestamp}.png"
        overlay_filepath = os.path.join(run_dir, overlay_filename)

        overlay.convert("RGB").save(overlay_filepath)
        return overlay_filepath
