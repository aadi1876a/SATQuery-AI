"""
backend/schemas.py
Shared data contracts for SatQuery AI.
EVERY teammate imports models from this file - never redefine these shapes elsewhere.
"""

from enum import Enum
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Universal Image Object (produced by P1's preprocessing pipeline)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Task types - single source of truth for task-name strings
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    vqa = "vqa"
    captioning = "captioning"
    grounding = "grounding"
    change_vqa = "change_vqa"
    fusion_analysis = "fusion_analysis"


# ---------------------------------------------------------------------------
# 3. Tool Input (P5 controller -> P2/P3/P4 model functions)
# ---------------------------------------------------------------------------

class ToolInput(BaseModel):
    task: TaskType
    query: str
    images: List[ImageObject]
    params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4. Tool Output (P2/P3/P4 model functions -> P5 controller)
# Every specialist model MUST return exactly this shape.
# ---------------------------------------------------------------------------

class SpatialEvidence(BaseModel):
    type: Literal["bbox", "mask", "none"]
    coords: Optional[List[float]] = None
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


# ---------------------------------------------------------------------------
# 5. Execution trace (the graded reasoning record)
# ---------------------------------------------------------------------------

class InputValidation(BaseModel):
    images_provided: int
    modalities: List[str] = Field(default_factory=list)
    co_registered: Optional[bool] = None
    format: Optional[str] = None
    status: Literal["valid", "invalid"]
    failure_reason: Optional[str] = None


class ExecutionTrace(BaseModel):
    task_detected: str
    input_validation: InputValidation
    tools_selected: List[str] = Field(default_factory=list)
    parameters_used: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 6. Final API response (P5 -> frontend)
# ---------------------------------------------------------------------------

class QueryResponse(BaseModel):
    status: Literal["success", "rejected", "error"]
    query: str
    answer: Optional[str] = None
    visual_evidence: List[SpatialEvidence] = Field(default_factory=list)
    confidence: Optional[float] = None
    execution_trace: ExecutionTrace
    report_download_url: Optional[str] = None
    reason: Optional[str] = None  # populated when status == "rejected"
