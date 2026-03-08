# Relationship Extraction Model Distillation Plan

## Goal
Distill a frontier-LLM-quality relationship extraction pipeline into a local ~7B model that can process 2M+ Wikipedia person articles on a single RTX 5070 Ti (16GB VRAM).

---

## Hardware
| Resource | Spec |
|----------|------|
| GPU | RTX 5070 Ti, 16GB VRAM |
| CPU | 2× AMD EPYC 7532 (128 threads total) |
| RAM | 256 GB |
| CUDA | 13.0 |

---

## Data Inventory

| File | Description | Stats |
|------|-------------|-------|
| `human_articles_en.ndjson` | All Wikipedia person articles | 2,009,804 articles |
| `llm-parse-test7d-openrouter.csv` | Golden dataset from Grok-4-fast | ~375K rows, 9,928 unique articles |
| `prompt4_v41_v2.txt` | System prompt used for extraction | ~2.5KB |
| `relationship-types.json` | 16 categories, ~70 directional pairs | ~3KB |

**Article text stats** (sampled 5K): median 16.8K chars, mean 23.8K chars, P90 56K chars.

**CSV data quality**: ~242 truly malformed rows out of 375K (0.06%). The CSV has no header row. Columns: `qid, model, provider, cost, name, relationship_type, evidence..., confidence`. The evidence field contains unquoted commas — fixable by treating cols 6 through -2 as evidence.

---

## Phase 1: Data Preparation

### 1a. Clean the golden CSV
- Parse rows using flexible column logic (cols 0–5 fixed, last col = confidence, middle = evidence).
- Drop rows where `len(row) < 7`, or confidence not in `{HIGH, MEDIUM, LOW}` (~242 rows).
- Drop rows with garbled relationship types (e.g., not matching `CATEGORY(role/role)` pattern).

### 1b. Build joined training JSON
- Group CSV rows by QID → list of `{name, relationship, evidence, confidence}`.
- Join with article data from `human_articles_en.ndjson` by QID.
- Each training example becomes:

```json
{
  "qid": "Q1000592",
  "title": "Tyson Fury",
  "article_text": "...",
  "relationships": [
    {"name": "Amber Fury", "relationship": "FAMILY(son/mother)", "evidence": "...", "confidence": "HIGH"},
    ...
  ]
}
```

### 1c. Train / Eval split
- **Train**: 90% of articles
- **Eval**: 10% held out
- Split by article (QID), not by row, to avoid leakage.
- Stratified random split to maintain relationship-type distribution.
- Seed the split for reproducibility.
- Split is applied *after* filtering out articles that exceed max_seq_length (see Phase 5).

### 1d. Convert to chat format for SFT
Each example becomes a conversation:
- **System message**: The extraction prompt (`prompt4_v41_v2.txt`) + relationship types JSON
- **User message**: `Article Data:\n\nTitle: {title}\n\n{article_text}`
- **Assistant message**: The CSV output (reconstructed from grouped rows) — **without the confidence column** (dropped). Each row: `name, CATEGORY(title_role/name_role), evidence sentence`

---

## Phase 2: Model Selection

### Recommended: **Qwen3-8B** (`unsloth/Qwen3-8B-bnb-4bit`)

| Consideration | Details |
|---------------|---------|
| Why Qwen3-8B | Best-in-class 7-8B model for instruction following and structured extraction as of early 2026. Strong at CSV/JSON generation. 8B params fits comfortably in 16GB with 4-bit QLoRA. Supports 32K context natively (covers P90 articles). Apache 2.0 license. |
| Runner-up | Llama-3.1-8B-Instruct — proven but slightly weaker on structured extraction benchmarks. |
| Why not 7B | Qwen3-8B at 8B is close enough to ~7B and meaningfully better. 4-bit quantized it uses ~5.5GB VRAM base. |

---

## Phase 3: Environment Setup

```bash
# Create venv and install unsloth + deps
pip install unsloth[cu130]
# unsloth installs: transformers, peft, trl, bitsandbytes, etc.
```

Initialize a git repo in the working directory for tracking.

---

## Phase 4: Baseline Evaluation (Pre-Training)

Before any fine-tuning, run the base Qwen3-8B (4-bit) on a sample of eval articles to establish:
- **Exact match rate**: % of relationships where (name, relationship_type) matches golden data exactly.
- **Name recall**: % of names from golden data that appear in the model output.
- **Relationship accuracy**: Of matched names, % with correct relationship type.
- **Format compliance**: % of outputs that are valid CSV.

Run on ~50 eval articles (small enough to finish quickly, large enough for signal).

---

## Phase 5: QLoRA Fine-Tuning with Unsloth

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank (r) | 64 | Higher rank for complex structured extraction task |
| LoRA alpha | 64 | alpha = r for stable scaling |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Full attention + MLP for maximum capacity |
| Quantization | 4-bit NF4 (QLoRA) | Fits in 16GB with room for gradients |
| Max sequence length | 16384 | Covers 93% of golden articles without truncation |
| Batch size | 1 | 16GB constraint |
| Gradient accumulation | 16 | Effective batch size = 16 |
| Learning rate | 2e-4 | Standard for QLoRA |
| LR scheduler | cosine | Standard decay |
| Warmup steps | 100 | ~1% of training |
| Epochs | 3 | Typical for SFT distillation |
| Optimizer | adamw_8bit | Memory-efficient |
| bf16 | Yes | RTX 5070 Ti supports bf16 |
| Gradient checkpointing | Yes (unsloth) | Saves VRAM |

### Context Length & Filtering
- Median article is ~17K chars ≈ ~4K tokens. P90 is ~56K chars ≈ ~14K tokens.
- With `max_seq_length=16384`, 93.2% of golden articles fit (9,250 of 9,928).
- **Strategy**: Filter out articles that exceed max_seq_length rather than truncating. This avoids training on incomplete data that would teach the model to produce partial relationship lists.
- The ~7% filtered articles are disproportionately long (biographies of very notable people) — the model will still encounter similar content at inference time, just without specialized training on those particular examples.

### Estimated Training Time
- ~8,325 training examples (90% of 9,250 that fit) × 3 epochs = ~24,975 steps (with grad accum 16 → ~1,561 optimizer steps).
- At ~3-6 sec/step on RTX 5070 Ti with 16K context: **~3–5 hours estimated**.

---

## Phase 6: Post-Training Evaluation

Re-run the same eval metrics from Phase 4 on the fine-tuned model:
- Compare exact match, name recall, relationship accuracy, format compliance.
- Generate a side-by-side comparison report.

---

## Phase 7: Export for Inference

- Merge LoRA weights back into base model.
- Save as GGUF (Q4_K_M or Q5_K_M) for fast inference with llama.cpp or vLLM.
- The merged + quantized model will be ~5GB, enabling fast batch inference over 2M articles.

---

## Decisions Made

1. **Max sequence length**: 16384 — covers 93% of golden articles. Articles that don't fit are excluded from training (not truncated).
2. **Train/eval split**: 90/10 (standard).
3. **Model**: Qwen3-8B.
4. **Output format**: `name, CATEGORY(title_role/name_role), evidence sentence` — no confidence column. Model generates its own evidence sentences.
5. **Long articles**: Filtered out of training set (not truncated). ~7% of golden data excluded.

---

## Execution Order

```
1. git init + commit initial files
2. Run data preparation scripts (Phase 1)
3. Install unsloth + dependencies (Phase 3)  
4. Download Qwen3-8B-4bit (Phase 2)
5. Run baseline eval on 50 eval articles (Phase 4)
6. Kick off QLoRA training (Phase 5)
7. Run post-training eval (Phase 6)
8. Export model (Phase 7)
```

---

*Review this plan and let me know what changes you'd like before I begin execution.*
