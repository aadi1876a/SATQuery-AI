# P3 Specialist Model — Next Steps & Fine-Tuning Roadmap

## 🛑 Important Architectural Directive

**Do NOT attempt Task 2 (prompt engineering / denylist tweaks).**

Prompt and denylist tweaks are temporary band-aids on a base Vision-Language Model (Salesforce BLIP) that fundamentally lacks remote sensing domain training. They will **not** prevent out-of-domain hallucinations (e.g. *"mars rover"*, *"plane crash"*) when evaluated on live satellite imagery from ISRO/SAC (such as Cartosat-2S or RISAT evaluation sets).

---

## 🛡️ Interim Fix (Currently Active)

- **BLIP Captioning Path Disabled**: Set `ENABLE_BLIP_VLM = False` in `sih26/backend/models/change_detection/vqa_engine.py`.
- **Honest Fallback Reporting**: The engine now routes all change descriptions through rule-based spectral analysis (NDVI/NDWI heuristics) and appends `(rule-based fallback)` to `ToolOutput.model_used`.
- **Why this is better for evaluation**: While crude, spectral heuristics produce zero hallucinations and honestly disclose the model state rather than claiming artificial intelligence while producing wild hallucinations.

---

## 🚀 The Real Fix: Proceed Straight to Task 3 (LoRA Fine-Tuning)

P3 requires a **LoRA (Low-Rank Adaptation) fine-tune** on Remote Sensing image-caption pairs.

### Recommended Steps for Task 3:
1. **Dataset Selection**: Use the **VRSBench** dataset (or equivalent Remote Sensing VQA / Change Captioning datasets).
2. **Reuse P2 Infrastructure**:
   - P2 has already established dataset formatting, tokenization, and LoRA fine-tuning code for single-image VQA.
   - Reuse P2's dataset formatting scripts and training pipeline to avoid duplicate effort.
3. **Training & Adapter Integration**:
   - Train LoRA adapters for BLIP / VLM on bi-temporal cropping pairs or RS image captioning.
   - Save adapter weights in `backend/models/change_detection/weights/`.
4. **Re-activation**:
   - Update `_get_blip_vlm()` in `vqa_engine.py` to load the LoRA weights.
   - Set `ENABLE_BLIP_VLM = True`.
