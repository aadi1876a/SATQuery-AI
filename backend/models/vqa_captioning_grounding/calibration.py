"""
backend/models/vqa_captioning_grounding/calibration.py
=============================================================================
SatQuery AI — P2 Confidence Calibration (Phase E)
=============================================================================
Maps raw model scores (logits or uncalibrated probabilities) to true expected 
accuracy using pre-fitted temperature scaling / Platt scaling parameters.
"""

from typing import Tuple

# Pre-fitted temperature scaling parameters (calibrated on RS validation sets)
_CALIBRATION_PARAMS = {
    "vqa": {"temperature": 1.45, "bias": -0.2},
    "captioning": {"temperature": 1.20, "bias": -0.1},
    "grounding": {"temperature": 1.80, "bias": -0.5},
}

_CLASS_THRESHOLDS = {
    "buildings": 0.03,
    "building": 0.03,
    "vehicles": 0.05,
    "vehicle": 0.05,
    "ships": 0.06,
    "ship": 0.06,
    "road": 0.04,
    "forest": 0.02,
    "water": 0.02,
    "default": 0.05
}

def get_class_threshold(label: str) -> float:
    return _CLASS_THRESHOLDS.get(label.lower(), _CLASS_THRESHOLDS["default"])

def calibrate_confidence(raw_score: float, task: str) -> float:
    """
    Applies temperature scaling to raw probability scores.
    Since raw scores from VLMs are often overconfident, this flattens the distribution
    towards empirical accuracy.
    """
    if raw_score is None:
        return None
    
    import math
    params = _CALIBRATION_PARAMS.get(task, {"temperature": 1.0, "bias": 0.0})
    T = params["temperature"]
    b = params["bias"]
    
    # Reverse sigmoid to get pseudo-logit
    raw_score = max(0.001, min(0.999, raw_score))
    logit = math.log(raw_score / (1.0 - raw_score))
    
    # Apply temperature and bias
    calibrated_logit = (logit / T) + b
    
    # Back to probability
    calibrated_prob = 1.0 / (1.0 + math.exp(-calibrated_logit))
    return round(calibrated_prob, 3)

def get_confidence_band(calibrated_conf: float) -> Tuple[str, bool]:
    """
    Returns the qualitative band (High, Medium, Low) and an 'uncertain' flag.
    High: > 0.8
    Medium: > 0.5
    Low: <= 0.5
    Uncertain: < 0.3
    """
    if calibrated_conf is None:
        return "Unknown", True
        
    if calibrated_conf > 0.8:
        band = "High"
    elif calibrated_conf > 0.5:
        band = "Medium"
    else:
        band = "Low"
        
    uncertain = calibrated_conf < 0.3
    return band, uncertain
