"""
validator.py
Checks whether the uploaded image(s) actually support the requested task,
before any model is called. Returns an InputValidation object every time.
"""

from backend.app.schemas.schemas import ImageObject, TaskType, InputValidation

# how many images each task requires
REQUIRED_IMAGES = {
    TaskType.vqa: 1,
    TaskType.captioning: 1,
    TaskType.grounding: 1,
    TaskType.change_vqa: 2,
    TaskType.fusion_analysis: 2,
}


def check_coregistration(img_a: ImageObject, img_b: ImageObject) -> bool:
    """
    Placeholder for real geospatial check.
    P1 will replace this with a rasterio/Shapely-based bbox+CRS comparison.
    """
    return img_a.crs == img_b.crs and img_a.bbox == img_b.bbox


def validate_input(task: TaskType, images: list[ImageObject]) -> InputValidation:
    required = REQUIRED_IMAGES[task]
    modalities = [img.modality.value for img in images]

    if len(images) < required:
        return InputValidation(
            images_provided=len(images),
            modalities=modalities,
            status="invalid",
            failure_reason=f"{task.value} requires {required} image(s), {len(images)} provided.",
        )

    co_registered = None
    if task in (TaskType.change_vqa, TaskType.fusion_analysis):
        co_registered = check_coregistration(images[0], images[1])
        if not co_registered:
            return InputValidation(
                images_provided=len(images),
                modalities=modalities,
                co_registered=False,
                status="invalid",
                failure_reason="Images are not co-registered (mismatched CRS/bbox).",
            )

    return InputValidation(
        images_provided=len(images),
        modalities=modalities,
        co_registered=co_registered,
        format=images[0].format if images else None,
        status="valid",
    )
