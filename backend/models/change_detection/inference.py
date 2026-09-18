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

from schemas import ToolInput, ToolOutput
from .change_engine import P3ChangeDetectionEngine

_engine = P3ChangeDetectionEngine()


def call_change_model(tool_input: ToolInput) -> ToolOutput:
    """
    Agreed integration contract function for P5 Controller.
    Takes ToolInput and returns ToolOutput.
    """
    return _engine.run(tool_input)
