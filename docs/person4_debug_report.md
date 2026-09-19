# Person 4 Debug Report

## ERROR 1: ModuleNotFoundError when running train.py
- **ERROR:** `ModuleNotFoundError: No module named 'backend'`
- **ROOT CAUSE:** `train.py` is executed directly as a script (`python train.py`), but it uses absolute imports (`from backend.models.fusion.fusion_model import OpticalSARFusionModel`).
- **FILE:** `backend/models/fusion/train.py`
- **FIX:** Run the script as a module from the root directory (`python -m backend.models.fusion.train`) or modify `sys.path`.
- **TEST:** `python -m backend.models.fusion.train`
- **RESULT:** The script runs successfully and starts training on the mock dataset.

## ERROR 2: SAR preprocessing uses PIL Image
- **ERROR:** SAR data is processed exactly like RGB optical data.
- **ROOT CAUSE:** `inference.py` uses `PIL.Image.open` and converts SAR to an 8-bit grayscale image (`img.convert("L")`). Real SAR data (e.g., Sentinel-1 GRD or SLC) is typically stored as 16-bit or 32-bit floats (GeoTIFF) containing backscatter values, which PIL cannot process correctly.
- **FILE:** `backend/models/fusion/inference.py` (Line 42)
- **FIX:** Must replace `PIL` with `rasterio` or similar geospatial libraries for reading SAR data, and implement appropriate dB conversion and normalization.
- **TEST:** (Pending integration with real data)
- **RESULT:** Current implementation is fundamentally broken for real SAR data.

## ERROR 3: Mock Dataset prevents real learning
- **ERROR:** The model trains perfectly on noise but learns nothing.
- **ROOT CAUSE:** `train.py` uses `MockFusionDataset`, which generates `torch.randn` images.
- **FILE:** `backend/models/fusion/train.py`
- **FIX:** Connect the training loop to actual satellite datasets (e.g., `BigEarthNet` or `SEN12MS`).
- **TEST:** (Pending dataset implementation)
- **RESULT:** Training completes without errors but produces a useless model.

## ERROR 4: No Test Set or Generalization Testing
- **ERROR:** The training pipeline only splits data into Train and Validation.
- **ROOT CAUSE:** The prompt specifically requires testing on a Held-Out Test Set and Completely Unseen External Data, but `train.py` lacks this logic.
- **FILE:** `backend/models/fusion/train.py`
- **FIX:** Implement separate `test()` function evaluating on different geographic splits.
- **TEST:** (Pending implementation)
- **RESULT:** The current evaluation pipeline violates P4 strict requirements.
