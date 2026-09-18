"""
classifier.py
Rule-based intent classifier. Zero dependencies, works offline -
this is your demo-safe fallback if the Gemini API is ever unavailable on stage.
"""

from schemas import TaskType


def classify_intent(query: str, num_images: int) -> TaskType:
    q = query.lower()

    if any(w in q for w in ["changed", "change between", "compare dates", "over time"]):
        return TaskType.change_vqa

    if any(w in q for w in ["highlight", "locate", "find the", "where is", "point out"]):
        return TaskType.grounding

    if any(w in q for w in ["optical and sar", "sar and optical", "together", "combine", "fuse"]):
        return TaskType.fusion_analysis

    if any(w in q for w in ["describe", "caption", "what does this image show", "summarize this image"]):
        return TaskType.captioning

    # default fallback: treat as a plain visual question
    return TaskType.vqa
