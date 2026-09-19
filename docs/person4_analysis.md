# Person 4 Analysis: Optical + SAR Fusion Model

## Existing P4 Architecture
The current P4 model is a simple multimodal classifier that combines Optical (3-channel) and SAR (1-channel) inputs.
- **Optical Encoder:** Pretrained ResNet-18 (ImageNet weights). Removes the final classification layer and projects features to a `feature_dim` of 256.
- **SAR Encoder:** Unpretrained ResNet-18 with a modified first convolutional layer (`conv1`) to accept 1-channel inputs. Projects features to a `feature_dim` of 256.
- **Fusion Method:** Feature-level fusion. The 256-dim optical features and 256-dim SAR features are concatenated to form a 512-dim vector.
- **Task Prediction:** A simple fully connected layer (Analysis Head) classifies the fused vector into 10 dummy land cover classes.

## Existing P4 Files
- `backend/models/fusion/fusion_model.py`: Contains `OpticalEncoder`, `SAREncoder`, and `OpticalSARFusionModel`.
- `backend/models/fusion/train.py`: Contains the training script, `MockFusionDataset`, and a simple training loop.
- `backend/models/fusion/inference.py`: Contains the `call_fusion_model` function for the P5 agent to run inference on incoming ToolInputs.

## Dataset
- Currently using a `MockFusionDataset` in `train.py` that generates random tensors (`torch.randn`) for both Optical and SAR modalities.
- `MockFusionDataset` produces 3-channel optical tensors, 1-channel SAR tensors, and random integer labels (0 to 9).
- There is NO real dataset being used right now. The `inference.py` script attempts to load images from disk using `PIL.Image`, which will fail for actual SAR datasets (usually TIFF or GeoTIFF).

## Preprocessing
- **Optical:** In `inference.py`, it resizes to 224x224 and converts to a PyTorch tensor (which scales values to [0,1]). It forces conversion to RGB.
- **SAR:** In `inference.py`, it uses the exact same optical preprocessing (`PIL.Image` resize, convert to "L" grayscale, and `ToTensor()`). This is incorrect for true SAR data, which usually contains float values, complex backscatter coefficients, and requires specialized clipping/normalization.

## Training Pipeline
- A basic PyTorch training loop in `train.py`.
- Uses CrossEntropyLoss and Adam optimizer (lr=1e-4).
- Saves the best model checkpoint based on validation loss to `backend/models/fusion/checkpoints/best_fusion_model.pth`.

## Evaluation Pipeline
- Evaluates on a validation split of the mock dataset during training. Calculates loss and accuracy.
- Lacks a test set or generalization evaluation.
- Lacks per-class metrics, confusion matrices, or modality ablation (Optical-only, SAR-only).

## Current Problems
1. **No Real Dataset:** It relies entirely on randomly generated noise data for training.
2. **Incorrect SAR Preprocessing:** SAR data is read using PIL and treated as a standard 8-bit image, which destroys true backscatter information.
3. **Execution Error (ModuleNotFoundError):** Running `train.py` directly throws a `ModuleNotFoundError` because of absolute imports (`from backend...`).
4. **No Real Evaluation:** The pipeline does not evaluate on test sets or unseen data. No ablation studies exist.
5. **No Actual Alignment Checking:** The code assumes optical and SAR tensors are already perfectly aligned and just processes them.

## Recommended P4 Implementation
- **Data Loaders:** Integrate real remote sensing loaders (e.g., rasterio-based) from P1 or implement robust loaders for Sentinel-1/Sentinel-2 datasets (like BigEarthNet-MM or SEN12MS).
- **SAR Preprocessing:** Implement proper SAR normalization (e.g., dB conversion, clipping outliers).
- **Ablation Ready Architecture:** Refactor the model to easily output predictions from Optical-only and SAR-only branches for the required ablation studies.
- **Evaluation Framework:** Create an evaluation script that calculates F1, Precision, Recall, and tests on distinct geographic splits for generalization testing.
