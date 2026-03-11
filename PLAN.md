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
| CUDA | 12.0 (toolkit 12.8) |

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
| Max sequence length | 8192 | Covers ~61% of golden articles; 16384 caused OOM |
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
- Initial plan: `max_seq_length=16384` covering 93% of articles.
- **Actual**: Reduced to `max_seq_length=8192` due to OOM at 16384 on 16GB VRAM.
- Token length distribution (full conversations): median 7029, mean 7701, P90 12821, P95 14186.
- At 8192 limit: 61.2% of golden articles fit → 4,726 train / 526 eval (4,675 filtered out).
- **Strategy**: Filter out articles that exceed max_seq_length rather than truncating.

### Actual Training Results
- **Training time**: 13.8 hours (49,660 seconds)
- **Total steps**: 886 (4,726 examples × 3 epochs ÷ 16 grad accum)
- **Step time**: ~54–58 seconds/step
- **Final train loss**: 0.0809 (converged to ~0.046 in final steps)
- **VRAM usage**: 15.4–15.8 GB (tight but stable, 100% GPU utilization)
- **Checkpoints**: Saved at steps 200, 400, 600
- **LoRA adapter**: `output/lora_adapter/`

---

## Phase 6: Post-Training Evaluation

### Results (49 eval samples, 809 golden relationships)

#### General
| Metric | Base Qwen3-8B | Fine-tuned |
|--------|---------------|------------|
| Format compliance | 32.7% | **100.0%** |
| Avg time/sample | 88.7s | **47.3s** |
| Total predicted | 89 | 839 |

#### Name-level matching
| Metric | Base Qwen3-8B | Fine-tuned |
|--------|---------------|------------|
| **Precision** | 7.9% | **87.8%** |
| **Recall** | 0.9% | **91.1%** |
| **F1** | 1.6% | **89.4%** |
| True Positives | 7 | 737 |
| False Positives | 82 | 102 |
| False Negatives | 802 | 72 |

#### Name+Relationship tuple matching
| Metric | Base Qwen3-8B | Fine-tuned |
|--------|---------------|------------|
| **Precision** | 3.4% | **76.5%** |
| **Recall** | 0.4% | **79.4%** |
| **F1** | 0.7% | **77.9%** |
| True Positives | 3 | 642 |
| False Positives | 86 | 197 |
| False Negatives | 806 | 167 |

#### Analysis
- Of 839 fine-tuned predictions, 197 (23.5%) are false positive tuples — the model identifies the right person but assigns the wrong relationship category, or hallucinates entities.
- Of those 197 FP tuples, 102 are completely wrong names (not in golden set) and 95 have the right name but wrong relationship type (737 name matches − 642 tuple matches).
- 167 golden tuples are missed entirely (false negatives), mostly from articles with many relationships where the model generates slightly fewer than the golden set.

---

## Phase 7: Export for Inference

- Merged LoRA weights into base model on CPU (8B fp16 doesn't fit in 16GB VRAM alongside adapter).
- Converted to GGUF F16 via `llama.cpp/convert_hf_to_gguf.py`, then quantized with `llama-quantize`.
- **Output files**:
  - `output/gguf/model-f16.gguf` — 15.26 GB
  - `output/gguf/model-Q5_K_M.gguf` — 5.45 GB
  - `output/lora_adapter/` — LoRA adapter weights

---

## Phase 8: Dataset Size Ablation Study

Trained Qwen3-8B (`unsloth/Qwen3-8B-bnb-4bit`) with 100, 500, and 1000 articles (subsampled from the 4,726-article training pool, seed=42) to find the minimum viable dataset size. The full training set contains 4,726 articles. All runs used the same 50-sample eval set (from 526-article eval pool) and identical hyperparameters (3 epochs, batch=1, grad_accum=16, lr=2e-4, max_seq=8192).

### Ablation Results (Qwen3-8B, varying dataset size)

| Articles | Train Examples | Train Loss | Train Time | Name F1 | Tuple F1 | Name P/R | Tuple P/R |
|----------|---------------|------------|------------|---------|----------|----------|-----------|
| 100 | 100 | 0.215 | 17.1 min | 54.2% | 26.3% | 44.3% / 70.0% | 21.5% / 34.0% |
| 500 | 500 | 0.143 | 86.0 min | 80.5% | 60.3% | 77.3% / 83.9% | 58.0% / 62.9% |
| 1000 | 1000 | 0.112 | 175.4 min | 86.1% | 71.2% | 84.7% / 87.6% | 70.0% / 72.4% |
| 4726 (full) | 4726 | 0.081 | 13.8 hr | 89.4% | 77.9% | 87.8% / 91.1% | 76.5% / 79.4% |

### Ablation Analysis
- **100→500 articles**: Largest jump — Name F1 +26.3pp, Tuple F1 +34.0pp. The 100-art model hallucinates heavily (FP=1004 tuple false positives vs 369 at 500).
- **500→1000 articles**: Solid improvement — Name F1 +5.6pp, Tuple F1 +10.9pp. Precision and recall both improve.
- **1000→4726 articles**: Diminishing returns — Name F1 +3.3pp, Tuple F1 +6.7pp. The 1000-art model already captures most of the pattern.
- **Recommendation**: 500 articles is the minimum viable size for reasonable quality. 1000 articles reaches ~91% of full-run performance at ~2% of training time.

### Adapter Locations
- `output/ablation_100art/lora_adapter/`
- `output/ablation_500art/lora_adapter/`
- `output/ablation_1000art/lora_adapter/`

---

## Phase 9: Cross-Model Comparison (1000 articles)

Training the same 1000-article subset across different model architectures to compare quality and training efficiency.

### Models Under Test

| Model | HuggingFace ID | Architecture | Params | Type |
|-------|---------------|--------------|--------|------|
| Qwen3-8B | `unsloth/Qwen3-8B-bnb-4bit` | Qwen3ForCausalLM | ~8B | Text-only |
| Gemma3-4B | `unsloth/gemma-3-4b-it-bnb-4bit` | Gemma3ForConditionalGeneration | ~4.4B | VLM (text+vision) |
| Qwen3.5-4B | `unsloth/Qwen3.5-4B` | Qwen3_5ForConditionalGeneration | ~4.6B | VLM (text+vision) |
| Qwen3.5-9B | `unsloth/Qwen3.5-9B` | Qwen3_5ForConditionalGeneration | ~9B | VLM — OOM on 16GB |
| Qwen3.5-35B-A3B | `unsloth/Qwen3.5-35B-A3B` | Qwen3_5MoeForCausalLM | 36B total / 3B active | MoE, bf16 LoRA on A100 |

### Cross-Model Results (1000 articles, same eval set)

All models trained on the same 1000-article subset (subsampled from the 4,726-article training pool, seed=42). Evaluated on the same 50-sample eval set (from 526-article eval pool). The full Qwen3-8B run used all 4,726 training articles — all other runs below used 1,000.

| Model | Train Loss | Train Time | Name F1 | Tuple F1 | Inference tok/s | VRAM | Notes |
|-------|-----------|------------|---------|----------|----------------|------|-------|
| **Qwen3-8B (1000 art)** | 0.112 | 175.4 min | 86.1% | 71.2% | **16.1** | ~15.4 GB | Text-only, pre-quantized 4-bit, ~54s/step |
| **Qwen3-8B (4726 art)** | 0.081 | 13.8 hr | 89.4% | 77.9% | **15.9** | ~15.4 GB | Full dataset, same model |
| Qwen3.5-4B (1000 art) | 0.068 | 669.7 min (11.16 hr) | **86.6%** | **71.7%** | 9.3 | ~14.6 GB | VLM, ~228s/step |
| Gemma3-4B (1000 art) | 0.126 | 257.9 min (4.30 hr) | 84.0% | 66.3% | 6.3 | ~15.5 GB | VLM, ~82.7s/step |
| Qwen3.5-9B | — | — | — | — | — | >16 GB | **Does not fit** — fp16→4bit conversion OOMs on 16GB |
| **Qwen3.5-35B-A3B (1000 art)** | 0.076 | 480.1 min (8.00 hr) | **89.0%** | **76.1%** | — | ~75 GB | MoE (36B/3B active), bf16 LoRA r=16 on A100-80GB, ~152s/step |

### Inference Speed Benchmark

Benchmarked on the same 5 eval articles (Lou Cannon, Dawn O'Porter, Brent Venables, Bruno Guimarães, Dacre Montgomery) with identical generation settings (max_new_tokens=2048, temperature=0.1, top_p=0.95). All models loaded in 4-bit via unsloth on RTX 5070 Ti.

| Model | Avg tok/s | Overall tok/s | Rank |
|-------|----------|--------------|------|
| Qwen3-8B (1000-art) | 15.8 | **16.1** | 1st (fastest) |
| Qwen3-8B (4726-art) | 15.1 | 15.9 | 2nd |
| Qwen3.5-4B (1000-art) | 8.8 | 9.3 | 3rd |
| Gemma3-4B (1000-art) | 6.1 | 6.3 | 4th (slowest) |

Qwen3-8B is **2.5× faster** at inference than Gemma3-4B and **1.7× faster** than Qwen3.5-4B, despite being the largest model. The text-only `Qwen3ForCausalLM` architecture benefits from well-optimized unsloth inference patches compared to the VLM architectures.

### Qwen3.5-35B-A3B: Thinking Mode Gotcha

Qwen3.5 models have a **thinking mode** enabled by default. When `apply_chat_template()` is called without `enable_thinking=False`, the prompt ends with:
```
<|im_start|>assistant\n<think>\n
```
This forces the model to generate a long internal reasoning chain inside `<think>...</think>` before producing the actual CSV output. With `max_new_tokens=2048`, the thinking consumed nearly all available tokens, leaving the CSV response truncated or empty.

**Symptoms**: Format compliance dropped to 18.8%, pred=0 for ~40/50 samples, Name/Tuple F1=5.6%. Training loss was excellent (0.076), so the model had learned correctly — it just couldn't output the answer due to thinking overhead. Inference took ~240s/sample (wasted on thinking tokens).

**Fix**: Pass `enable_thinking=False` to `apply_chat_template()`:
```python
text = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False,
    enable_thinking=False,  # Critical for Qwen3.5 models
)
```
With `enable_thinking=False`, the prompt pre-closes the think block (`<think>\n\n</think>\n\n`), so the model outputs CSV directly. Results jumped from 5.6% → 89.0% Name F1, inference dropped from ~240s → ~20s/sample for small articles, and format compliance hit 100%.

**Note**: This applies to all Qwen3.5 models (4B, 9B, 35B MoE, 122B). The training data contains no thinking tokens, so the model should never be prompted to think during inference.

### Cross-Model Analysis
- **Qwen3.5-35B-A3B (MoE)**: **Best quality on 1000 articles** — Name F1=89.0%, Tuple F1=76.1%, matching the full 4,726-article Qwen3-8B run quality. Trained with bf16 LoRA (r=16) on A100-80GB. MoE architecture means only 3B params active per token despite 36B total. Required `FastModel` (not `FastLanguageModel`) and `enable_thinking=False` for correct inference. 8 hours to train (~152s/step).
- **Qwen3-8B** is the fastest to train (~54s/step), fastest at inference (~16 tok/s), and delivers strong results. The full 4,726-article run reaches Tuple F1=77.9%. Clear winner on consumer hardware (16GB).
- **Gemma3-4B**: Lowest quality (Tuple F1=66.3%) AND slowest inference (6.3 tok/s). Required SDPA attention workaround and checkpoint recomputation patch. Not recommended.
- **Qwen3.5-4B**: Slightly outperforms Qwen3-8B on quality (Tuple F1 71.7% vs 71.2%) but ~4× slower to train and ~1.7× slower at inference due to VLM overhead.
- **Qwen3.5-9B**: Cannot be loaded on 16GB GPU.
- **Recommendation**: For consumer GPU (16GB), Qwen3-8B remains the best choice — fast training, fast inference, strong quality. For maximum quality with cloud GPU, Qwen3.5-35B-A3B achieves near-parity with the full Qwen3-8B dataset using only 1000 articles.

### Adapter Locations
- Qwen3-8B full (4726-art): `output/lora_adapter/`
- Qwen3-8B (1000-art): `output/ablation_1000art/lora_adapter/`
- Gemma3-4B (1000-art): `output/1000art_gemma3_4b/lora_adapter/`
- Qwen3.5-4B (1000-art): `output/1000art_qwen35_4b/lora_adapter/`
- Qwen3.5-35B-A3B (1000-art): `output/1000art_qwen35_35b/lora_adapter/` (bf16 LoRA, requires A100+)

### Running Models
To load a fine-tuned adapter for inference:
```python
from unsloth import FastLanguageModel
from peft import PeftModel

# For Qwen3-8B
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-8B-bnb-4bit", max_seq_length=8192, load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "output/ablation_1000art/lora_adapter")
FastLanguageModel.for_inference(model)

# For Gemma3-4B
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gemma-3-4b-it-bnb-4bit", max_seq_length=8192, load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "output/1000art_gemma3_4b/lora_adapter")
FastLanguageModel.for_inference(model)

# For Qwen3.5-4B
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3.5-4B", max_seq_length=8192, load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "output/1000art_qwen35_4b/lora_adapter")
FastLanguageModel.for_inference(model)
```

---

## Decisions Made

1. **Max sequence length**: 8192 (reduced from planned 16384 due to OOM). Covers 61% of golden articles. Articles that don't fit are excluded (not truncated).
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
