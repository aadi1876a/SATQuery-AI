"""
backend/models/vqa_captioning_grounding/create_sample_sar.py
=============================================================================
SatQuery AI — P2 Synthetic SAR Test Image Generator
=============================================================================
Generates physically realistic synthetic Sentinel-1 GRD SAR imagery with:
  1. Low-backscatter water bodies (specular reflection, ~ -22 dB)
  2. Medium-backscatter agricultural/vegetation zones (volume scattering, ~ -12 dB)
  3. High-backscatter urban structures / corner reflectors (double bounce, ~ +3 dB)
  4. Multiplicative Rayleigh / Exponential speckle noise characteristic of SAR sensors.

Outputs:
  - outputs/test_images/sar/sar_sample_patch.png
  - outputs/test_images/sar/sar_sample_patch.tif (GeoTIFF float32 linear backscatter)
"""

import os
import sys
import numpy as np
from PIL import Image

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUTS_DIR = os.path.join(_CURRENT_DIR, "outputs")
SAR_TEST_DIR = os.path.join(_OUTPUTS_DIR, "test_images", "sar")


def generate_synthetic_sar_scene(
    width: int = 512,
    height: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates a 2D float32 linear backscatter array (sigma_0).

    Returns:
        np.ndarray: float32 linear intensity image (H, W).
    """
    np.random.seed(seed)

    # 1. Base terrain: Rural/agricultural field with diffuse volume scattering
    # Mean backscatter ~ -12 dB (linear ~ 0.063)
    base_linear = np.full((height, width), 0.065, dtype=np.float32)

    # Add slight terrain spatial variation
    y_coords, x_coords = np.indices((height, width))
    terrain_var = 0.015 * np.sin(x_coords / 40.0) * np.cos(y_coords / 50.0)
    base_linear += terrain_var

    # 2. Water body / river: Specular reflection directed away from sensor
    # Mean backscatter ~ -24 dB (linear ~ 0.004)
    # Winding river curve
    river_center = 200 + 80 * np.sin(x_coords / 70.0)
    river_mask = np.abs(y_coords - river_center) < 32
    base_linear[river_mask] = 0.005

    # 3. Roads / Runway: Smooth asphalt surface (low-to-moderate backscatter ~ -18 dB)
    road_mask = (np.abs(x_coords - 120) < 6) | (np.abs(x_coords - 380) < 6)
    base_linear[road_mask] = 0.018

    # 4. Urban clusters / Buildings / Corner reflectors
    # High backscatter double-bounce reflections ~ +2 to +6 dB (linear ~ 1.5 to 4.0)
    buildings = [
        # (y1, y2, x1, x2)
        (70, 130, 200, 260),
        (80, 140, 290, 350),
        (350, 420, 200, 270),
        (360, 430, 300, 370),
        (220, 260, 40, 90),
    ]

    for y1, y2, x1, x2 in buildings:
        base_linear[y1:y2, x1:x2] = 2.5
        # Highly reflective metal edge / wall corner
        base_linear[y1:y1 + 4, x1:x2] = 4.2
        base_linear[y1:y2, x1:x1 + 4] = 3.8

    # 5. Add realistic SAR speckle noise (multiplicative exponential noise, 1-look)
    speckle = np.random.exponential(scale=1.0, size=(height, width)).astype(np.float32)
    # 4-look equivalent averaging for Sentinel-1 GRD
    speckle_4look = (
        speckle +
        np.random.exponential(scale=1.0, size=(height, width)).astype(np.float32) +
        np.random.exponential(scale=1.0, size=(height, width)).astype(np.float32) +
        np.random.exponential(scale=1.0, size=(height, width)).astype(np.float32)
    ) / 4.0

    sar_intensity = base_linear * speckle_4look
    sar_intensity = np.clip(sar_intensity, 0.0001, 10.0)
    return sar_intensity.astype(np.float32)


def save_sar_test_patch(output_dir: str = SAR_TEST_DIR) -> dict:
    """Generates and saves synthetic SAR patches in both PNG and GeoTIFF formats."""
    os.makedirs(output_dir, exist_ok=True)
    raw_sar = generate_synthetic_sar_scene(512, 512)

    png_path = os.path.join(output_dir, "sar_sample_patch.png")
    tif_path = os.path.join(output_dir, "sar_sample_patch.tif")

    # 1. Save normalized 8-bit grayscale PNG
    db = 10.0 * np.log10(raw_sar + 1e-10)
    db_min, db_max = np.percentile(db, 1), np.percentile(db, 99)
    norm = np.clip((db - db_min) / (db_max - db_min + 1e-10), 0.0, 1.0)
    img_uint8 = (norm * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(png_path)

    # 2. Save 32-bit float GeoTIFF if rasterio is available
    try:
        import rasterio
        from rasterio.transform import from_bounds

        transform = from_bounds(77.10, 28.50, 77.25, 28.65, 512, 512)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=512,
            width=512,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(raw_sar, 1)
        has_tif = True
    except Exception as e:
        print(f"[SAR Generator] Note: GeoTIFF write skipped ({e}), PNG available.")
        has_tif = False

    return {
        "png": png_path,
        "tif": tif_path if has_tif else png_path,
    }


if __name__ == "__main__":
    paths = save_sar_test_patch()
    print(f"[SAR Generator] Synthetic SAR test patch created:")
    print(f"  PNG: {paths['png']}")
    print(f"  TIF: {paths['tif']}")
