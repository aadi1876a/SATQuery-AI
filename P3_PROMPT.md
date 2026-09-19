# Prompt for Antigravity (SatQuery AI - P3 Specialist Model)

Use the following master prompt to prompt Antigravity or any AI subagent to build or extend the **P3 Specialist Model: Change Detection & Temporal Analysis** module.

---

### 📋 COPY-PASTE PROMPT FOR ANTIGRAVITY:

```markdown
Role & System Context:
You are building the P3 Specialist Model module for the "SatQuery AI" platform.
Your domain is: Change Detection + Temporal Analysis (Bi-temporal satellite and geospatial analysis).
Your team structure:
- P1: Remote Sensing + Geospatial (Data preprocessing, GeoTIFF, SAR registration)
- P2: VLM + VQA + Grounding (Single-image AI)
- P3: Change Detection + Temporal Analysis (Your role: Bi-temporal change detection & temporal VQA)
- P4: Optical-SAR Fusion + Multimodal AI
- P5: Agentic AI + Backend + Integration (Sends ToolInput to you and expects ToolOutput in return)

Data Contracts & Schemas (Strict Requirement):
All inputs and outputs MUST strictly adhere to `schemas.py`:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class Modality(str, Enum):
    optical = "optical"
    sar = "sar"

class ImageObject(BaseModel):
    image_id: str
    file_path: str
    modality: Modality
    format: str
    bands: int
    resolution_m: float
    crs: str
    bbox: List[float]  # [minLon, minLat, maxLon, maxLat]
    acquisition_date: Optional[str] = None
    width: int
    height: int
    thumbnail_path: Optional[str] = None

class TaskType(str, Enum):
    vqa = "vqa"
    captioning = "captioning"
    grounding = "grounding"
    change_vqa = "change_vqa"
    fusion_analysis = "fusion_analysis"

class ToolInput(BaseModel):
    task: TaskType
    query: str
    images: List[ImageObject]
    params: dict = Field(default_factory=dict)

class SpatialEvidence(BaseModel):
    type: Literal["bbox", "mask", "none"]
    coords: Optional[List[float]] = None # [ymin, xmin, ymax, xmax] normalized [0.0, 1.0]
    mask_path: Optional[str] = None
    label: Optional[str] = None

class ToolOutput(BaseModel):
    status: Literal["success", "error"]
    text_answer: Optional[str] = None
    spatial_evidence: List[SpatialEvidence] = Field(default_factory=list)
    confidence: Optional[float] = None
    raw_output_path: Optional[str] = None
    model_used: str
    error_message: Optional[str] = None
```

Objective & Key Requirements:
1. Create `p3_model.py` containing a `P3ChangeDetectionEngine` class with a `run(tool_input: ToolInput) -> ToolOutput` method.
2. The engine must accept bi-temporal images (Image T1 at t=0, Image T2 at t=1) provided in `tool_input.images`.
3. Perform multi-spectral 3-channel change detection using CIELAB color space distance + RGB max channel difference, Gaussian blurring, Otsu automatic thresholding, and morphological filtering.
4. Extract precise, tight bounding boxes for ALL detected changes using OpenCV contour analysis (`cv2.findContours` + `cv2.boundingRect`) normalized to `[ymin, xmin, ymax, xmax]` in `[0.0, 1.0]` scale.
5. Save spatial evidence:
   - Save the exact binary change mask PNG to `outputs/`.
   - Draw an overlay image highlighting ALL changes and tight bounding boxes on T2 and save to `raw_output_path`.
6. Generate a clear textual description in `text_answer` answering the user's `query` with temporal change stats (e.g. total area changed %, number of clusters).
7. Provide a standalone test script `test_p3.py` with synthetic bi-temporal test images to verify the full flow end-to-end.
```
