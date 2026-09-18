"""Sample Geospatial Data Generator for SatQuery AI test suite."""

import os
from typing import Optional
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False


def create_geotiff(
    filepath: str,
    width: int = 128,
    height: int = 128,
    num_bands: int = 4,
    bounds: Optional[list] = None,
    crs: Optional[str] = "EPSG:4326",
    acquisition_date: Optional[str] = "2024:01:15 10:30:00",
    is_sar: bool = False,
):
    """Generate a valid synthetic GeoTIFF with geospatial metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if bounds is None:
        bounds = [72.8500, 19.0500, 72.9500, 19.1500]

    min_x, min_y, max_x, max_y = bounds
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    tags = {}
    if acquisition_date:
        tags["TIFFTAG_DATETIME"] = acquisition_date
        tags["ACQUISITION_DATE"] = acquisition_date

    if is_sar:
        # 2 bands: VV and VH in dB scale (typically negative floats, e.g. -25 to -5 dB)
        descriptions = ("VV", "VH") if num_bands == 2 else ("VV",)
        tags["SENSOR_TYPE"] = "SAR"
        tags["POLARIZATION"] = "VV,VH" if num_bands == 2 else "VV"

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=num_bands,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
        ) as dst:
            for b in range(1, num_bands + 1):
                # Backscatter simulation with Rayleigh/speckle distribution
                data = -15.0 + 5.0 * np.random.randn(height, width).astype(np.float32)
                dst.write(data, b)
                if b <= len(descriptions):
                    dst.set_band_description(b, descriptions[b - 1])
            dst.update_tags(**tags)
    else:
        # Optical multispectral: 4 bands (Red, Green, Blue, NIR)
        descriptions = ("Red", "Green", "Blue", "NIR") if num_bands >= 4 else ("Red", "Green", "Blue")
        tags["SENSOR_TYPE"] = "OPTICAL"
        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=num_bands,
            dtype=rasterio.uint16,
            crs=crs,
            transform=transform,
        ) as dst:
            for b in range(1, num_bands + 1):
                data = np.random.randint(200, 4000, size=(height, width), dtype=np.uint16)
                dst.write(data, b)
                if b <= len(descriptions):
                    dst.set_band_description(b, descriptions[b - 1])
            dst.update_tags(**tags)


def generate_all_samples(samples_dir: str):
    """Generate all test files required by the test suite."""
    os.makedirs(samples_dir, exist_ok=True)

    # 1. Optical Image 1 (T1: 2024-01-15)
    create_geotiff(
        os.path.join(samples_dir, "optical_t1_20240115.tif"),
        num_bands=4,
        crs="EPSG:4326",
        bounds=[72.85, 19.05, 72.95, 19.15],
        acquisition_date="2024:01:15 10:00:00",
        is_sar=False,
    )

    # 2. Optical Image 2 (T2: 2024-06-15, same CRS & bounds -> valid bi-temporal pair)
    create_geotiff(
        os.path.join(samples_dir, "optical_t2_20240615.tif"),
        num_bands=4,
        crs="EPSG:4326",
        bounds=[72.85, 19.05, 72.95, 19.15],
        acquisition_date="2024:06:15 10:00:00",
        is_sar=False,
    )

    # 3. SAR Image (S1: 2024-01-15, dual-pol, same CRS & bounds -> valid optical-SAR cross-modal pair)
    create_geotiff(
        os.path.join(samples_dir, "sar_s1_20240115.tif"),
        num_bands=2,
        crs="EPSG:4326",
        bounds=[72.85, 19.05, 72.95, 19.15],
        acquisition_date="2024:01:15 10:00:00",
        is_sar=True,
    )

    # 4. Optical with mismatched bounds (for co-registration rejection)
    create_geotiff(
        os.path.join(samples_dir, "optical_mismatched_bounds.tif"),
        num_bands=4,
        crs="EPSG:4326",
        bounds=[75.00, 25.00, 75.10, 25.10],
        acquisition_date="2024:06:15 10:00:00",
        is_sar=False,
    )

    # 5. Optical with different CRS (EPSG:32643 - UTM Zone 43N)
    create_geotiff(
        os.path.join(samples_dir, "optical_utm_crs.tif"),
        num_bands=4,
        crs="EPSG:32643",
        bounds=[273000.0, 2108000.0, 283000.0, 2118000.0],
        acquisition_date="2024:01:15 10:00:00",
        is_sar=False,
    )

    # 6. Optical missing CRS (tests graceful degradation)
    create_geotiff(
        os.path.join(samples_dir, "optical_missing_crs.tif"),
        num_bands=4,
        crs=None,
        acquisition_date="2024:01:15 10:00:00",
        is_sar=False,
    )

    # 7. Benchmark PNG image
    png_path = os.path.join(samples_dir, "benchmark_optical_20240201.png")
    img_data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(img_data).save(png_path)

    # 8. Benchmark JPEG image
    jpeg_path = os.path.join(samples_dir, "benchmark_optical_20240301.jpg")
    Image.fromarray(img_data).save(jpeg_path, quality=90)

    # 9. Deliberately corrupted file
    corrupt_path = os.path.join(samples_dir, "corrupt_truncated.tif")
    with open(corrupt_path, "wb") as f:
        f.write(b"II*\x00\x08\x00\x00\x00corrupt_random_bytes_that_fail_to_parse")

    # 10. Unsupported format file
    unsupported_path = os.path.join(samples_dir, "invalid_format.txt")
    with open(unsupported_path, "w") as f:
        f.write("This is a text file, not an image.")


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/samples"
    generate_all_samples(out)
    print(f"Sample data generated successfully in {out}")
