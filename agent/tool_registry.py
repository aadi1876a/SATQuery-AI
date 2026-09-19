"""
tool_registry.py
Single source of truth for which function handles which task.
Swap mock_tools imports for real_tools imports as teammates finish their models -
no other code needs to change because everyone returns the same ToolOutput shape.
"""

from backend.models.vqa_captioning_grounding.inference import (
    call_vqa_model,
    call_caption_model,
)
from backend.models.change_detection.inference import call_change_model

from agent.tools.mock_tools import (
    call_grounding_model,
    call_fusion_model,
)

TOOL_REGISTRY = {
    "vqa": call_vqa_model,                # owner: P2
    "captioning": call_caption_model,     # owner: P2
    "grounding": call_grounding_model,    # owner: P4
    "change_vqa": call_change_model,      # owner: P3
    "fusion_analysis": call_fusion_model, # owner: P4
}
