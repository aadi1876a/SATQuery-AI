# SATQuery-AI 🛰️🤖
> **Multi-Modal Geospatial AI Agent for Remote Sensing (Optical & SAR) VQA, Captioning, Grounding, Change Detection, and Multi-Sensor Fusion.**

---

## 🎯 Architecture Overview

SATQuery-AI is divided into modular pipelines conforming to strict Pydantic v2 data contracts defined in `backend/schemas.py`:

- **P1 (Data & Preprocessing)**: Geospatial input validator, band alignment, BigEarthNet pipeline.
- **P2 (VLM, VQA & Grounding)**:
  - **Remote Sensing VQA**: `Salesforce/blip-vqa-base` adapted with **PEFT LoRA** trained on **RSVQA**.
  - **Image Captioning**: `Salesforce/blip-image-captioning-base` with geometric-mean confidence normalization.
  - **Open-Vocabulary Grounding & Segmentation**: `OWLv2` (`google/owlv2-base-patch16-ensemble`) + `SAM` (`facebook/sam-vit-base`).
  - **SAR Preprocessing**: Speckle reduction (Lee filter), dB scale transformation, CLAHE contrast enhancement.
- **P3 (Change Detection)**: Bi-temporal Siamese CNN / visual differencing and VQA engine.
- **P4 (Multi-Sensor Fusion)**: Optical + SAR feature alignment and fusion.
- **P5 (Agent Controller & API)**: ReAct geospatial agent orchestrator routing user natural language queries to specialized tools.

---

## 🚀 Person 2 (P2) Remote-Sensing Domain Adaptation

### Problem Statement Compliance
Generic vision-language models fail on satellite imagery due to nadir perspective, high feature density, and unique spectral signatures. SATQuery-AI addresses this with domain-specific LoRA fine-tuning.

### Evaluation: Base BLIP vs. RS-LoRA Adapted BLIP

| Metric | Base BLIP (`Salesforce/blip-vqa-base`) | RS-LoRA Adapted BLIP (`+ RS-LoRA-Adapted`) |
|---|:---:|:---:|
| **Overall Accuracy** | **TBD%** | **TBD%** |
| **Yes/No Accuracy** | **TBD%** | **TBD%** |
| **Count Accuracy** | **TBD%** | **TBD%** |
| **Count MAE** | **TBD** | **TBD** |

Detailed benchmark results and architectural decisions are documented in [`backend/docs/p2_evaluation.md`](backend/docs/p2_evaluation.md).

---

## 🛠️ Grounding Architectural Decision

Grounding utilizes **Zero-Shot OWLv2 + Segment Anything Model (SAM)**:
- **Open-Vocabulary**: Detects arbitrary user-specified objects (*"storage tanks"*, *"runways"*, *"solar panels"*).
- **Sub-Pixel Polygon Masks**: SAM converts bounding boxes into precise segmentation masks.
- **Calibrated Scores**: Raw OWLv2 detection logits.
- **NMS**: Per-class Non-Maximum Suppression with IoU 0.5.

---

## 🧪 Testing & Verification

Run the test suite:
```bash
# Verify P2 tests
python backend/tests/test_p2.py
# or
pytest backend/tests/test_p2.py -v -s
```

Run independent model testing:
```bash
python backend/models/vqa_captioning_grounding/test_all_models.py
```
