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
| 1000 | 1000 | 0.112 | 2.92 hr | 86.1% | 71.2% | 84.7% / 87.6% | 70.0% / 72.4% |
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
| Qwen3.5-9B | `unsloth/Qwen3.5-9B` | Qwen3_5ForConditionalGeneration | ~9.5B | VLM (text+vision) |
| Nanbeige4.1-3B | `Nanbeige/Nanbeige4.1-3B` | LlamaForCausalLM | ~4B | Text-only |
| Qwen3.5-4B-Claude-Opus | `Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled` | Qwen3_5ForConditionalGeneration | ~4.6B | VLM (text+vision), Claude-Opus reasoning distilled |
| Qwen3.5-4B Abliterated | `SicariusSicariiStuff/Qwen3.5-4B_Abliterated` | Qwen3_5ForConditionalGeneration | ~4.6B | VLM (text+vision), abliterated |
| Qwen3.5-9B | `unsloth/Qwen3.5-9B` | Qwen3_5ForConditionalGeneration | ~9B | VLM — OOM on 16GB |
| Qwen3.5-27B | `unsloth/Qwen3.5-27B` | Qwen3_5ForConditionalGeneration | ~27B | VLM (text+vision), bf16 LoRA on H100 |
| Qwen3.5-35B-A3B | `unsloth/Qwen3.5-35B-A3B` | Qwen3_5MoeForCausalLM | 36B total / 3B active | MoE, bf16 LoRA on A100 |
| Llama-3.2-3B | `unsloth/Llama-3.2-3B-Instruct` | LlamaForCausalLM | ~3.2B | Text-only |

### Cross-Model Results (1000 articles, same eval set)

All models trained on the same 1000-article subset (subsampled from the 4,726-article training pool, seed=42). Evaluated on the same 50-sample eval set (from 526-article eval pool). Full-dataset rows (Qwen3-8B 4726-art, Qwen3.5-9B 4726-art, Llama-3.2-3B 4726-art) used all available training articles; all other runs used 1,000.

| Model | Train Loss | Train Time | Name F1 | Tuple F1 | Inference tok/s | VRAM | Notes |
|-------|-----------|------------|---------|----------|----------------|------|-------|
| Qwen3-8B (4726 art) | 0.081 | 13.8 hr | 89.4% | 77.9% | 15.9 | ~15.4 GB | Full dataset, same model |
| Qwen3.5-9B (4726 art) | 0.052 | 29.0 hr | 90.6% | **77.3%** | TBD | ~32 GB | VLM, bf16 LoRA r=64 (not QLoRA), full dataset, RTX 5090 — Phase A |
| Qwen3.5-35B-A3B (1000 art) | 0.076 | 8.00 hr | 89.0% | 76.1% | — | ~75 GB | MoE (36B/3B active), bf16 LoRA r=16 on A100-80GB, ~152s/step |
| Qwen3.5-27B (1000 art) | 0.050 | 12.26 hr | 87.6% | 73.8% | — | ~96 GB | VLM, bf16 LoRA on H100 NVL, ~234s/step |
| Qwen3.5-9B (1000 art) | 0.063 | 5.37 hr | 86.4% | 73.4% | TBD | ~32 GB | VLM, bf16 LoRA (not QLoRA), ~103s/step, trained on RTX 5090 |
| Qwen3.5-4B-Opus (1000 art) | 0.068 | 11.17 hr | 88.2% | 72.3% | TBD | ~14.6 GB | VLM, Claude-Opus distilled, ~215s/step, requires `</think>` fix |
| Qwen3.5-4B (1000 art) | 0.068 | 11.16 hr | 86.6% | 71.7% | 9.3 | ~14.6 GB | VLM, ~228s/step |
| Qwen3-8B (1000 art) | 0.112 | 2.92 hr | 86.1% | 71.2% | 16.1 | ~15.4 GB | Text-only, pre-quantized 4-bit, ~54s/step |
| Llama-3.2-3B (4726 art) | 0.079 | 6.2 hr | 87.8% | 76.0% | TBD | ~10 GB | Text-only, QLoRA 4-bit, full dataset, RTX 5070 Ti |
| Llama-3.2-3B (1000 art) | 0.111 | 0.59 hr | 85.3% | 71.2% | TBD | ~10 GB | Text-only, ~11s/step, trained on RTX 5090 |
| Gemma3-4B (1000 art) | 0.126 | 4.30 hr | 84.0% | 66.3% | 6.3 | ~15.5 GB | VLM, ~82.7s/step |
| Nanbeige4.1-3B (1000 art) | 0.124 | 0.73 hr | 81.2% | 64.4% | 23.1* | ~12.4 GB | Text-only, ~14.0s/step, *benchmarked on 5090 |
| Qwen3.5-4B Abliterated (1000 art) | 0.068 | 4.83 hr | 26.5% | 21.3% | 22.9* | ~17.1 GB | VLM, ~93s/step, trained+benchmarked on RTX 5090 |

### Inference Speed Benchmark

Benchmarked on the same 5 eval articles (Lou Cannon, Dawn O'Porter, Brent Venables, Bruno Guimarães, Dacre Montgomery) with identical generation settings (max_new_tokens=2048, temperature=0.1, top_p=0.95). All models loaded in 4-bit via unsloth on RTX 5070 Ti.

| Model | Avg tok/s | Overall tok/s | Rank |
|-------|----------|--------------|------|
| Qwen3-8B (1000-art) | 15.8 | **16.1** | 1st (fastest) |
| Qwen3-8B (4726-art) | 15.1 | 15.9 | 2nd |
| Qwen3.5-4B (1000-art) | 8.8 | 9.3 | 3rd |
| Gemma3-4B (1000-art) | 6.1 | 6.3 | 4th (slowest) |
| Nanbeige4.1-3B (1000-art)* | 22.6 | 23.1 | — (different GPU) |
| Qwen3.5-4B Abliterated (1000-art)* | 22.8 | 22.9 | — (different GPU) |

\* Nanbeige4.1-3B was benchmarked on RTX 5090 (32GB), not RTX 5070 Ti. Direct speed comparison with other models is not apples-to-apples.

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

### llama3.2-3b every 500 article eval

| Step | Articles | Name F1 | Name P | Name R | Tuple F1 | Tuple P | Tuple R | Format |
|------|----------|---------|--------|--------|----------|---------|---------|--------|
| base | 0 | 0.1% | 0.1% | 0.1% | 0.0% | 0.0% | 0.0% | 49.0% |
| 94 | 501 | 83.4% | 82.4% | 84.4% | 64.5% | 63.7% | 65.3% | 100.0% |
| 188 | 1003 | 76.4% | 68.2% | 86.9% | 65.9% | 58.8% | 74.9% | 100.0% |
| 282 | 1504 | 87.8% | 87.4% | 88.1% | 74.8% | 74.5% | 75.2% | 100.0% |
| 376 | 2005 | 88.8% | 86.7% | 91.1% | 71.2% | 69.5% | 73.1% | 100.0% |
| 470 | 2507 | 87.6% | 86.4% | 88.9% | 76.2% | 75.1% | 77.3% | 100.0% |
| 564 | 3008 | 87.3% | 85.6% | 89.0% | 77.1% | 75.6% | 78.6% | 100.0% |
| 658 | 3509 | 88.0% | 86.9% | 89.2% | 76.2% | 75.2% | 77.3% | 100.0% |
| 752 | 4011 | 88.5% | 87.7% | 89.2% | 75.6% | 75.0% | 76.3% | 100.0% |
| **846** | **4512** | **89.3%** | **88.4%** | **90.2%** | **76.8%** | **76.0%** | **77.6%** | **100.0%** |
| 888 | 4736 | 87.6% | 84.8% | 90.6% | 76.3% | 73.8% | 78.9% | 100.0% |

### Cross-Model Analysis
- **Qwen3.5-27B**: Strong quality (Name F1=87.6%, Tuple F1=73.8%) but required H100 NVL (96GB VRAM) for bf16 LoRA training. Trained for 12.26 hours with lowest train loss (0.050). Quality exceeds all 1000-art models except the MoE 35B-A3B. Required `enable_thinking=False` for inference (same Qwen3.5 thinking mode gotcha). 2 samples skipped due to input length.
- **Qwen3.5-35B-A3B (MoE)**: **Best quality on 1000 articles** — Name F1=89.0%, Tuple F1=76.1%, matching the full 4,726-article Qwen3-8B run quality. Trained with bf16 LoRA (r=16) on A100-80GB. MoE architecture means only 3B params active per token despite 36B total. Required `FastModel` (not `FastLanguageModel`) and `enable_thinking=False` for correct inference. 8 hours to train (~152s/step).
- **Qwen3-8B** is the fastest to train (~54s/step), fastest at inference (~16 tok/s), and delivers strong results. The full 4,726-article run reaches Tuple F1=77.9%. Clear winner on this hardware.
- **Nanbeige4.1-3B**: Lowest quality (Tuple F1=64.4%, Name F1=81.2%) but fastest training (43.6 min, ~14s/step) and lowest VRAM (~12.4 GB). High inference speed on 5090 (23.1 tok/s) but not directly comparable to 5070 Ti benchmarks. Text-only LlamaForCausalLM architecture.
- **Gemma3-4B**: Low quality (Tuple F1=66.3%) AND slowest inference (6.3 tok/s). Required SDPA attention workaround and checkpoint recomputation patch. Not recommended.
- **Qwen3.5-4B-Claude-Opus**: Fine-tuned from `Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled`, a Claude-4.6-Opus reasoning-distilled variant of Qwen3.5-4B. **Best Name F1 (88.2%) among all 1000-art models**, and Tuple F1=72.3% — outperforms the base Qwen3.5-4B (+1.6pp Name F1, +0.6pp Tuple F1). Same train loss (0.068) and similar training time (~670 min). The tokenizer's chat template always injects `<think>\n` regardless of `enable_thinking=False`, requiring a manual `</think>\n` append to force-close the thinking block. Inference is very slow (~1 tok/s) due to VLM + reasoning architecture overhead. 48/50 eval samples processed (2 skipped: input too long).
- **Qwen3.5-4B**: Slightly outperforms Qwen3-8B on quality (Tuple F1 71.7% vs 71.2%) but ~4× slower to train and ~1.7× slower at inference due to VLM overhead.
- **Qwen3.5-9B**: Best Tuple F1 (73.4%) among 1000-art runs. Requires RTX 5090 (32GB) for bf16 LoRA training — OOMs on 16GB GPUs. Train loss 0.063 (lowest). ~103s/step, 5.37 hours total. Quality is comparable to Qwen3.5-4B (73.4% vs 71.7% Tuple F1) but trains 2× faster per step and benefits from larger model capacity.
- **Llama-3.2-3B (4726 art)**: Full-dataset training improves over 1000-art by +4.8pp Tuple F1 (76.0%) and +2.5pp Name F1 (87.8%). Reaches near Qwen3-8B full-dataset quality (77.9% Tuple F1) with less than half the parameters (3.2B vs 8B). Trained locally on RTX 5070 Ti with QLoRA 4-bit in 6.2 hours (~25s/step). Only ~10GB VRAM. 100% format compliance.
- **Llama-3.2-3B (1000 art)**: Matches Qwen3-8B's Tuple F1 (71.2%) with less than half the parameters (3.2B vs 8B). Fastest to train by far (~11s/step, 35.5 min total vs 175 min for Qwen3-8B). Only ~10GB VRAM during training. Strong format compliance (100%). A compelling lightweight option.
- **Qwen3.5-4B Abliterated**: Abliteration severely damaged instruction-following capacity. Format compliance only 45.8% (many samples produce 0 valid CSV lines). Name F1=26.5%, Tuple F1=21.3%. Despite identical architecture and lower train loss (0.068 vs 0.068 for regular Qwen3.5-4B), the abliterated weights cannot recover structured extraction capability even after fine-tuning. Inference speed on 5090 (22.9 tok/s) is similar to Nanbeige4.1-3B on the same hardware. **Not recommended.**

### Adapter Locations
- Qwen3-8B full (4726-art): `output/lora_adapter/`
- Qwen3-8B (1000-art): `output/ablation_1000art/lora_adapter/`
- Gemma3-4B (1000-art): `output/1000art_gemma3_4b/lora_adapter/`
- Qwen3.5-4B (1000-art): `output/1000art_qwen35_4b/lora_adapter/`
- Qwen3.5-27B (1000-art): `output/1000art_qwen35_27b/lora_adapter/` (bf16 LoRA, requires H100+, on RunPod)
- Qwen3.5-35B-A3B (1000-art): `output/1000art_qwen35_35b/lora_adapter/` (bf16 LoRA, requires A100+)
- Nanbeige4.1-3B (1000-art): `output/1000art_nanbeige41_3b/lora_adapter/`
- Llama-3.2-3B (4726-art): `output/llama32_3b_full/lora_adapter/`
- Llama-3.2-3B (1000-art): `output/1000art_llama32_3b/lora_adapter/`
- Qwen3.5-4B-Claude-Opus (1000-art): `output/1000art_qwen35_4b_opus/lora_adapter/`
- Qwen3.5-4B Abliterated (1000-art): `output/1000art_qwen35_4b_abliterated/lora_adapter/`
- Qwen3.5-9B (1000-art): `output/1000art_qwen35_9b/lora_adapter/`
- Qwen3.5-9B (4726-art): `output/phase_a/phase_a/lora_adapter/` (bf16 LoRA r=64, Phase A)

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

# For Qwen3.5-4B-Claude-Opus (requires </think> fix for inference)
model, tokenizer = FastLanguageModel.from_pretrained(
    "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled", max_seq_length=8192, load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "output/1000art_qwen35_4b_opus/lora_adapter")
FastLanguageModel.for_inference(model)
# NOTE: After apply_chat_template(), append "</think>\n" to close the injected <think> block

# For Qwen3.5-4B Abliterated
model, tokenizer = FastLanguageModel.from_pretrained(
    "SicariusSicariiStuff/Qwen3.5-4B_Abliterated", max_seq_length=8192, load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "output/1000art_qwen35_4b_abliterated/lora_adapter")
# For Qwen3.5-9B (requires 32GB+ VRAM, bf16 LoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3.5-9B", max_seq_length=8192, load_in_4bit=False, load_in_16bit=True,
)
model = PeftModel.from_pretrained(model, "output/1000art_qwen35_9b/lora_adapter")
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
# Runpod Best Practices
- use python sdk (use uv with '--with" for running and managing deps)
- provision machines yourself using these guidelines:
    - 1 x 5090
    - secure cloud
    - 250GB disk
    - use on-demand (not spot)
    - template: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
- always run python unbuffered so that you can track logs (especially during the evaluation phase)
- make logs easy to track for me in case I want to probe them independently
- use rclone for downloading from runpod
- upload code+data via rsync/scp over SSH to the pod
- continuously monitor the training and give me regular updates every 15min
- each phase gets its own output directory so multiple phases can run in parallel on separate pods
- when finished with a pod, download all of the important artifacts and then delete the pod