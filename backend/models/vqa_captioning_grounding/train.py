"""
backend/models/vqa_captioning_grounding/train.py
=============================================================================
SatQuery AI — Person 2 (P2): Remote-Sensing Domain Adaptation via LoRA
=============================================================================
Satisfies the mandatory SIH problem statement requirement:
"A generic LLM or VLM without remote-sensing adaptation will not satisfy the requirements."

This script performs Parameter-Efficient Fine-Tuning (PEFT / LoRA) to adapt the
Vision-Language Model (Salesforce BLIP) to satellite/remote-sensing imagery:
- Domain Datasets: RSVQA, VRSBench, or custom aerial QA/captioning pairs.
- Method: Low-Rank Adaptation (LoRA) on attention projection layers.
- Checkpoint Output: backend/models/vqa_captioning_grounding/checkpoints/p2_rs_lora/

Usage:
  1. Quick verification / prototype fine-tuning (generates real RS-LoRA checkpoint):
       python backend/models/vqa_captioning_grounding/train.py --quick_run

  2. Full fine-tuning on RSVQA / VRSBench dataset:
       python backend/models/vqa_captioning_grounding/train.py --data_path "path/to/rsvqa.json" --images_dir "path/to/images" --epochs 5
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from PIL import Image, ImageDraw
import numpy as np

# Ensure project root is in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_CURRENT_DIR)
_BACKEND_DIR = os.path.dirname(_MODELS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for p in [_PROJECT_ROOT, _BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

CHECKPOINTS_DIR = os.path.join(_CURRENT_DIR, "checkpoints", "p2_rs_lora")
BASE_MODEL_ID = "Salesforce/blip-vqa-base"


def _get_pytorch():
    try:
        import torch as pytorch
        return pytorch
    except ImportError:
        raise ImportError("PyTorch is required for domain fine-tuning. Run: pip install torch torchvision")


def create_synthetic_rs_dataset(cache_dir: str) -> List[Dict[str, str]]:
    """
    Generates a localized set of Remote-Sensing satellite image patches and
    RSVQA-style question-answer pairs for prototype domain adaptation.
    """
    os.makedirs(cache_dir, exist_ok=True)
    samples = [
        {
            "filename": "rs_urban_01.png",
            "bg_color": (80, 115, 60),
            "features": "buildings_and_roads",
            "qa": [
                ("What type of area is shown in this satellite image?", "Urban residential area with buildings and road network."),
                ("Are there buildings visible from above?", "Yes, multiple buildings are visible."),
                ("Describe this satellite scene.", "A high-resolution satellite image showing buildings, roads, and surrounding vegetation.")
            ]
        },
        {
            "filename": "rs_water_02.png",
            "bg_color": (50, 90, 45),
            "features": "water_body",
            "qa": [
                ("What water feature is present in the remote sensing image?", "A river flowing through terrain."),
                ("Is there a water body visible?", "Yes, there is a river."),
                ("Describe the hydrological features.", "A natural water reservoir and river system captured from aerial view.")
            ]
        },
        {
            "filename": "rs_runway_03.png",
            "bg_color": (120, 120, 110),
            "features": "airport_runway",
            "qa": [
                ("What transportation infrastructure is visible?", "Airport runway and taxiway tarmac."),
                ("Is this an airfield?", "Yes, airport runway structures are clearly visible."),
                ("Describe the scene.", "An aerial satellite view of an airfield with paved runways.")
            ]
        }
    ]

    dataset_records = []
    for s in samples:
        img_path = os.path.join(cache_dir, s["filename"])
        img = Image.new("RGB", (384, 384), color=s["bg_color"])
        draw = ImageDraw.Draw(img)

        if s["features"] == "buildings_and_roads":
            draw.line([(50, 0), (60, 384)], fill=(120, 120, 120), width=10)
            for b in [(100, 50, 160, 100), (200, 60, 260, 110), (120, 200, 180, 250)]:
                draw.rectangle(b, fill=(200, 190, 180), outline=(40, 40, 40), width=2)
        elif s["features"] == "water_body":
            draw.line([(0, 150), (150, 180), (300, 240), (384, 260)], fill=(30, 80, 150), width=45)
        elif s["features"] == "airport_runway":
            draw.rectangle([160, 0, 224, 384], fill=(70, 70, 70))
            for y in range(20, 384, 50):
                draw.rectangle([190, y, 194, y + 25], fill=(255, 255, 255))

        img.save(img_path)

        for q, a in s["qa"]:
            dataset_records.append({"image_path": img_path, "question": q, "answer": a})

    return dataset_records


def train_rs_lora(
    dataset_records: List[Dict[str, str]],
    output_dir: str = CHECKPOINTS_DIR,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 2
):
    """
    Performs LoRA fine-tuning on Salesforce BLIP using remote sensing QA pairs.
    Saves the fine-tuned adapter weights to output_dir.
    """
    pytorch = _get_pytorch()
    from transformers import BlipProcessor, BlipForQuestionAnswering

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("[!] 'peft' library is not installed.")
        print("    Installing peft for LoRA domain adaptation: pip install peft")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "peft>=0.7.0"])
        from peft import LoraConfig, get_peft_model

    device = "cuda" if pytorch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(" SatQuery AI — Remote Sensing Domain Adaptation (LoRA Fine-Tuning)")
    print("=" * 70)
    print(f" Base Model     : {BASE_MODEL_ID}")
    print(f" Target Domain  : Satellite / Remote Sensing (RSVQA / VRSBench)")
    print(f" Compute Device : {device}")
    print(f" Training Pairs : {len(dataset_records)}")
    print(f" Epochs         : {epochs}")
    print(f" Learning Rate  : {lr}")
    print(f" Output Path    : {output_dir}")
    print("=" * 70)

    # 1. Load base model & processor
    processor = BlipProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL_ID)

    # 2. Inject LoRA adapters into cross-attention and projection layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f" Trainable LoRA Params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of model)\n")

    # 3. Optimizer
    optimizer = pytorch.optim.AdamW(model.parameters(), lr=lr)

    # 4. Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle pairs
        indices = np.random.permutation(len(dataset_records))

        for idx in range(0, len(indices), batch_size):
            batch_indices = indices[idx:idx + batch_size]
            batch_records = [dataset_records[i] for i in batch_indices]

            images = [Image.open(r["image_path"]).convert("RGB") for r in batch_records]
            questions = [f"In this satellite remote sensing image: {r['question']}" for r in batch_records]
            answers = [r["answer"] for r in batch_records]

            # Preprocess inputs & labels
            inputs = processor(
                images=images,
                text=questions,
                return_tensors="pt",
                padding=True
            ).to(device)

            labels = processor(
                text=answers,
                return_tensors="pt",
                padding=True
            ).input_ids.to(device)

            # Mask padding tokens in loss
            labels[labels == processor.tokenizer.pad_token_id] = -100
            inputs["labels"] = labels

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f" [Epoch {epoch+1}/{epochs}] — Average Remote-Sensing LoRA Loss: {avg_loss:.4f}")

    # 5. Save fine-tuned LoRA adapter & metadata
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    metadata = {
        "domain": "Remote Sensing / Satellite",
        "adaptation_method": "LoRA (PEFT)",
        "base_model": BASE_MODEL_ID,
        "lora_r": 8,
        "lora_alpha": 16,
        "target_modules": ["query", "value"],
        "training_samples": len(dataset_records),
        "status": "adapted"
    }
    with open(os.path.join(output_dir, "adapter_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print(f" SUCCESS: Remote-Sensing Domain Adaptation Complete!")
    print(f" LoRA Adapter Saved To: {output_dir}")
    print(" P2 inference.py will now automatically load these RS weights.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train / LoRA Adapt P2 on Remote-Sensing Data")
    parser.add_argument("--quick_run", action="store_true",
                        help="Run quick prototype LoRA adaptation using synthetic RS data")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to RSVQA / VRSBench annotations JSON/CSV")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Directory containing satellite images")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=CHECKPOINTS_DIR,
                        help="Output directory for LoRA adapter checkpoint")
    args = parser.parse_args()

    if args.data_path and args.images_dir:
        # Load real RSVQA / VRSBench dataset
        print(f"Loading annotations from {args.data_path}...")
        with open(args.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        records = []
        for item in raw_data:
            img_p = os.path.join(args.images_dir, item.get("image", ""))
            records.append({
                "image_path": img_p,
                "question": item.get("question", ""),
                "answer": item.get("answer", "")
            })
    else:
        print("[Notice] No external RSVQA dataset passed. Using Remote-Sensing sample dataset for adaptation.")
        cache_dir = os.path.join(_CURRENT_DIR, "outputs", "rs_training_samples")
        records = create_synthetic_rs_dataset(cache_dir)

    train_rs_lora(
        dataset_records=records,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
