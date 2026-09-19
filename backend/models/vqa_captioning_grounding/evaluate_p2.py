"""
backend/models/vqa_captioning_grounding/evaluate_p2.py
=============================================================================
SatQuery AI — P2 Remote Sensing Domain Adaptation Evaluation Benchmark
=============================================================================
Runs held-out RSVQA test questions through:
  1. Base Model: Salesforce/blip-vqa-base
  2. RS-LoRA Adapted Model: Salesforce/blip-vqa-base + RS-LoRA adapter

Compares domain accuracy, terminology, and reasoning against ground truth.
Outputs a markdown comparison table for project documentation.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from PIL import Image

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_CURRENT_DIR)
_BACKEND_DIR = os.path.dirname(_MODELS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for p in [_PROJECT_ROOT, _BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.schemas import ToolInput, TaskType, Modality, ImageObject
from backend.models.vqa_captioning_grounding.utils import get_device, get_pytorch

CHECKPOINTS_DIR = os.path.join(_CURRENT_DIR, "checkpoints", "p2_rs_lora")
BASE_MODEL_ID = "Salesforce/blip-vqa-base"
DEFAULT_TEST_JSON = os.path.join(_BACKEND_DIR, "data_pipeline", "rsvqa_subset", "test_annotations.json")
DEFAULT_IMAGES_DIR = os.path.join(_BACKEND_DIR, "data_pipeline", "rsvqa_subset", "images")


def run_evaluation(test_json: str = DEFAULT_TEST_JSON, images_dir: str = DEFAULT_IMAGES_DIR):
    if not os.path.exists(test_json):
        print(f"[!] Test annotations not found at: {test_json}")
        return

    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    device = get_device()
    print("=" * 80)
    print(" SatQuery AI — P2 Evaluation: Base BLIP vs. RS-LoRA Adapted BLIP")
    print("=" * 80)
    print(f" Test Set Size: {len(test_data)} held-out RSVQA questions")
    print(f" Compute Device: {device}")
    print("=" * 80)

    from transformers import BlipProcessor, BlipForQuestionAnswering
    from peft import PeftModel

    processor = BlipProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL_ID).to(device)
    base_model.eval()

    lora_available = os.path.exists(CHECKPOINTS_DIR) and any(
        f.startswith("adapter_") for f in os.listdir(CHECKPOINTS_DIR)
    )
    if lora_available:
        print(f"[+] Found LoRA checkpoint at {CHECKPOINTS_DIR}. Loading adapted model...")
        lora_base = BlipForQuestionAnswering.from_pretrained(BASE_MODEL_ID).to(device)
        lora_model = PeftModel.from_pretrained(lora_base, CHECKPOINTS_DIR).to(device)
        lora_model.eval()
    else:
        print("[!] No LoRA checkpoint found. Running base model only.")
        lora_model = None

    results = []
    base_correct = 0
    lora_correct = 0

    print("\nRunning inference on held-out questions...\n")

    for i, item in enumerate(test_data):
        img_path = os.path.join(images_dir, item["image"])
        if not os.path.exists(img_path):
            continue

        raw_image = Image.open(img_path).convert("RGB")
        q = item["question"]
        gt = str(item["answer"]).strip().lower()

        prompt_q = f"In this satellite remote sensing image: {q}"

        # 1. Base Model
        inputs_base = processor(images=raw_image, text=prompt_q, return_tensors="pt").to(device)
        out_base = base_model.generate(**inputs_base, max_new_tokens=20)
        ans_base = processor.decode(out_base[0], skip_special_tokens=True).strip()

        # 2. LoRA Model
        if lora_model is not None:
            inputs_lora = processor(images=raw_image, text=prompt_q, return_tensors="pt").to(device)
            out_lora = lora_model.generate(**inputs_lora, max_new_tokens=20)
            ans_lora = processor.decode(out_lora[0], skip_special_tokens=True).strip()
        else:
            ans_lora = "N/A"

        # Judgment
        c_base = (gt in ans_base.lower()) or (ans_base.lower() in gt)
        c_lora = (gt in ans_lora.lower()) or (ans_lora.lower() in gt) if lora_model else False

        if c_base:
            base_correct += 1
        if c_lora:
            lora_correct += 1

        results.append({
            "idx": i + 1,
            "question": q,
            "ground_truth": gt,
            "base_pred": ans_base,
            "lora_pred": ans_lora,
            "base_ok": c_base,
            "lora_ok": c_lora
        })

        ok_base = "[OK]" if c_base else "[X]"
        ok_lora = "[OK]" if c_lora else "[X]"
        print(f"[{i+1}/{len(test_data)}] Q: {q}")
        print(f"       GT: {gt} | Base: {ans_base} ({ok_base}) | LoRA: {ans_lora} ({ok_lora})\n")

    total = len(results)
    base_acc = (base_correct / total) * 100 if total else 0
    lora_acc = (lora_correct / total) * 100 if total else 0

    print("=" * 80)
    print(" EVALUATION SUMMARY")
    print("=" * 80)
    print(f" Total Held-Out Questions: {total}")
    print(f" Base BLIP Accuracy      : {base_acc:.1f}% ({base_correct}/{total})")
    if lora_model:
        print(f" RS-LoRA Adapted Accuracy: {lora_acc:.1f}% ({lora_correct}/{total})")
        delta = lora_acc - base_acc
        print(f" Net Domain Gain         : {delta:+.1f}%")
    print("=" * 80)

    # Output markdown table
    out_table_path = os.path.join(_BACKEND_DIR, "docs", "p2_benchmark_results.json")
    with open(out_table_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "base_accuracy": base_acc,
            "lora_accuracy": lora_acc,
            "results": results
        }, f, indent=2)

    return results

if __name__ == "__main__":
    run_evaluation()
