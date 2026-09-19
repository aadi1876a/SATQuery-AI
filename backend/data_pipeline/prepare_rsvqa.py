"""
backend/data_pipeline/prepare_rsvqa.py
Extracts a clean, real RSVQA subset (images + annotations)
from dmarsili/RSVQA-LR-2k for P2 domain adaptation and evaluation.
"""

import os
import io
import json
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

def prepare_data():
    target_dir = os.path.join(os.path.dirname(__file__), "rsvqa_subset")
    images_dir = os.path.join(target_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("[RSVQA] Downloading/reading parquet from Hugging Face cache...")
    parquet_path = hf_hub_download(
        repo_id="dmarsili/RSVQA-LR-2k",
        filename="data/validation-00000-of-00001.parquet",
        repo_type="dataset"
    )
    df = pd.read_parquet(parquet_path)
    print(f"[RSVQA] Loaded {len(df)} total pairs.")

    # We select 250 diverse training pairs and 20 distinct held-out evaluation pairs
    train_records = []
    test_records = []

    # Filter out empty or uninformative questions if any
    valid_indices = []
    for idx, row in df.iterrows():
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        if q and a:
            valid_indices.append(idx)

    # Save images and build annotation records
    saved_images = {}
    train_count = 250
    test_count = 20

    for i, idx in enumerate(valid_indices[: train_count + test_count]):
        row = df.iloc[idx]
        img_dict = row["image"]
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) and "bytes" in img_dict else img_dict

        img_filename = f"rsvqa_{i:04d}.png"
        img_path = os.path.join(images_dir, img_filename)

        if img_filename not in saved_images:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img.save(img_path)
            saved_images[img_filename] = img_path

        record = {
            "image": img_filename,
            "question": str(row["question"]).strip(),
            "answer": str(row["answer"]).strip()
        }

        if i < train_count:
            train_records.append(record)
        else:
            test_records.append(record)

    train_json_path = os.path.join(target_dir, "train_annotations.json")
    test_json_path = os.path.join(target_dir, "test_annotations.json")

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)

    print(f"[RSVQA] Success!")
    print(f" - Images saved in: {images_dir} ({len(saved_images)} images)")
    print(f" - Train annotations: {train_json_path} ({len(train_records)} pairs)")
    print(f" - Test annotations: {test_json_path} ({len(test_records)} pairs)")

if __name__ == "__main__":
    prepare_data()
