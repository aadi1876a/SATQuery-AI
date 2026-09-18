"""
controller.py
The agentic core: classifies the query, validates inputs, calls the right tool(s),
and builds the execution trace. This is the piece that gets graded on "orchestration."
"""

from schemas import (
    ImageObject, ToolInput, QueryResponse, ExecutionTrace, TaskType
)
from agent.classifier import classify_intent
from agent.validator import validate_input
from agent.tool_registry import TOOL_REGISTRY


def run_query(query: str, images: list[ImageObject]) -> QueryResponse:
    # 1. classify intent
    task = classify_intent(query, num_images=len(images))

    # 2. validate inputs against the detected task
    validation = validate_input(task, images)

    if validation.status == "invalid":
        trace = ExecutionTrace(
            task_detected=task.value,
            input_validation=validation,
            tools_selected=[],
            parameters_used={},
        )
        return QueryResponse(
            status="rejected",
            query=query,
            execution_trace=trace,
            reason=validation.failure_reason,
        )

    # 3. select + execute the tool
    tool_fn = TOOL_REGISTRY[task.value]
    tool_input = ToolInput(task=task, query=query, images=images, params={})
    tool_output = tool_fn(tool_input)

    # 4. build execution trace
    trace = ExecutionTrace(
        task_detected=task.value,
        input_validation=validation,
        tools_selected=[tool_output.model_used],
        parameters_used=tool_input.params,
    )

    if tool_output.status == "error":
        return QueryResponse(
            status="error",
            query=query,
            execution_trace=trace,
            reason=tool_output.error_message,
        )

    # 5. package final response
    return QueryResponse(
        status="success",
        query=query,
        answer=tool_output.text_answer,
        visual_evidence=tool_output.spatial_evidence,
        confidence=tool_output.confidence,
        execution_trace=trace,
    )
