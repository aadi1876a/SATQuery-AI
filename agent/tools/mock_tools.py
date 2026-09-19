"""
mock_tools.py
Fake versions of P2/P3/P4's model functions.
They return correctly-shaped ToolOutput objects with dummy data,
so the controller can be built and tested before real models exist.

Swap these one at a time for the real functions in real_tools.py
once each teammate's model is ready - the shape never changes.
"""

from backend.app.schemas.schemas import ToolInput, ToolOutput, SpatialEvidence


def call_vqa_model(tool_input: ToolInput) -> ToolOutput:
    return ToolOutput(
        status="success",
        text_answer="This is a mock VQA answer describing the land cover.",
        spatial_evidence=[],
        confidence=0.91,
        model_used="mock_vqa_v0",
    )


def call_caption_model(tool_input: ToolInput) -> ToolOutput:
    return ToolOutput(
        status="success",
        text_answer="Mock caption: an urban area near a river with agricultural land nearby.",
        spatial_evidence=[],
        confidence=0.88,
        model_used="mock_caption_v0",
    )


def call_grounding_model(tool_input: ToolInput) -> ToolOutput:
    return ToolOutput(
        status="success",
        text_answer="Located the requested region.",
        spatial_evidence=[
            SpatialEvidence(type="bbox", coords=[120, 340, 210, 400], label="water_body")
        ],
        confidence=0.85,
        model_used="mock_grounding_v0",
    )


def call_change_model(tool_input: ToolInput) -> ToolOutput:
    return ToolOutput(
        status="success",
        text_answer="Mock change result: new built-up construction appeared in the northeast region.",
        spatial_evidence=[
            SpatialEvidence(type="bbox", coords=[80, 90, 180, 190], label="new_construction")
        ],
        confidence=0.87,
        model_used="mock_change_vqa_v0",
    )


def call_fusion_model(tool_input: ToolInput) -> ToolOutput:
    return ToolOutput(
        status="success",
        text_answer="Mock fusion result: built-up 23%, water 11%, based on combined optical+SAR analysis.",
        spatial_evidence=[],
        confidence=0.9,
        model_used="mock_fusion_v0",
    )
