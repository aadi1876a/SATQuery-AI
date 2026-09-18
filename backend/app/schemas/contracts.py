from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"

class InputConfig(str, Enum):
    SINGLE = "single"
    CROSS_MODAL_PAIR = "cross_modal_pair"
    BI_TEMPORAL_PAIR = "bi_temporal_pair"
    INVALID = "invalid"

class BigEarthNetMeta(BaseModel):
    patch_id: str
    labels: List[str]
    split: str
    country: str
    s1_name: str
    s2v1_name: str

class ImageMeta(BaseModel):
    filename: str
    format: str                          # "GeoTIFF" | "TIFF" | "PNG" | "JPEG"
    modality: Modality
    width: int
    height: int
    crs: Optional[str] = None            # coordinate reference system, e.g. "EPSG:32643"
    acquisition_date: Optional[str] = None
    bounds: Optional[List[float]] = None # [min_lon, min_lat, max_lon, max_lat]
    ben_meta: Optional[BigEarthNetMeta] = None

class ValidationResult(BaseModel):
    is_valid: bool
    reason: Optional[str] = None         # REQUIRED when is_valid=False — shown directly to the user
    images_meta: List[ImageMeta] = Field(default_factory=list)
    detected_config: InputConfig = InputConfig.INVALID
