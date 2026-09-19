"""
backend/models/change_detection/inference.py
SatQuery AI - P3 Specialist Model Integration Interface.

This is the single entry point for the P5 Controller to call the P3 Change Detection specialist model.
"""

import sys
import os

# Ensure repository root is in sys.path for importing schemas
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from typing import Union, Dict, Any
from schemas import ToolInput, ToolOutput
from .change_engine import P3ChangeDetectionEngine

_engine = P3ChangeDetectionEngine()


def call_change_model(tool_input: Union[ToolInput, Dict[str, Any], str]) -> ToolOutput:
    """
    Agreed integration contract function for P5 Controller.
    
    Accepts:
      - ToolInput Pydantic object
      - Python dictionary matching ToolInput schema
      - Raw JSON string matching ToolInput schema
      
    Returns:
      - ToolOutput Pydantic object (which supports .model_dump_json() and .model_dump())
    """
    if isinstance(tool_input, str):
        validated_input = ToolInput.model_validate_json(tool_input)
    elif isinstance(tool_input, dict):
        validated_input = ToolInput.model_validate(tool_input)
    elif isinstance(tool_input, ToolInput):
        validated_input = tool_input
    else:
        raise ValueError(f"Invalid input type for call_change_model: {type(tool_input)}. Expected ToolInput, dict, or JSON string.")

    return _engine.run(validated_input)


def call_change_model_json(tool_input_json: str) -> str:
    """
    JSON-in / JSON-out integration wrapper for P5 Controller.
    Takes a raw JSON string input and returns a structured 100% schema-compliant JSON string response.
    """
    output: ToolOutput = call_change_model(tool_input_json)
    return output.model_dump_json(indent=2)
