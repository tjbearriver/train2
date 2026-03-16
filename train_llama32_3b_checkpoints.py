#!/usr/bin/env python3
"""Train Llama-3.2-3B on ALL 4,722 articles, saving a checkpoint every ~500 articles.

Designed for benchmarking how performance scales with dataset size.
Each checkpoint saved at save_steps=94 corresponds to ~500 unique articles trained on
(500 articles × 3 epochs ÷ 16 grad_accum = 93.75 ≈ 94 steps).

Usage:
  mkdir -p output/llama32_3b_ckpt && \
  nohup /home/tj/workspace/tj/cranberry/train2/.venv/bin/python -u \
    train_llama32_3b_checkpoints.py > output/llama32_3b_ckpt/train.log 2>&1 &
"""

import gc
import json
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
FULL_TRAIN_PATH = BASE_DIR / "data" / "train.json"
MAX_SEQ_LENGTH = 8192

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
TAG = "llama32_3b_ckpt"

LORA_R = 64
LORA_ALPHA = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_8bit"
LOGGING_STEPS = 5

# Save every ~500 articles: 500 articles × 3 epochs / 16 grad_accum = 93.75 ≈ 94 steps
SAVE_STEPS = 94
SAVE_TOTAL_LIMIT = 15  # keep all checkpoints


def train_model(train_data: list[dict], output_dir: Path, checkpoint_dir: Path):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    total_steps = (len(train_data) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)
    articles_per_ckpt = SAVE_STEPS * GRAD_ACCUM // EPOCHS
    print(f"Training {MODEL_NAME} on {len(train_data)} examples (QLoRA 4-bit)")
    print(f"Total steps: ~{total_steps}, saving every {SAVE_STEPS} steps (~{articles_per_ckpt} articles)")
    print(f"Expected checkpoints: {total_steps // SAVE_STEPS + 1}")
    print(f"Output: {output_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    def formatting_func(examples):
        texts = []
        for convos in examples["conversations"]:
            text = tokenizer.apply_chat_template(convos, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(formatting_func, batched=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIMIZER,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_only_model=True,   # don't save optimizer state — keep disk use low
        seed=42,
        report_to="none",
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        max_length=MAX_SEQ_LENGTH,
        eos_token=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    inst_part = "<|start_header_id|>user<|end_header_id|>\n\n"
    resp_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    sample_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=False, add_generation_prompt=False,
    )
    assert "<|start_header_id|>" in sample_text, f"Unexpected template: {sample_text[:200]}"

    trainer = train_on_responses_only(
        trainer,
        instruction_part=inst_part,
        response_part=resp_part,
    )

    print(f"\nStarting training [{time.strftime('%H:%M:%S')}]")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed/3600:.2f} hr, loss={train_result.training_loss:.4f}")

    # Save final adapter
    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    stats = {
        "model_name": MODEL_NAME,
        "tag": TAG,
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": elapsed,
        "train_samples": len(train_data),
        "epochs": EPOCHS,
        "total_steps": total_steps,
        "save_steps": SAVE_STEPS,
        "articles_per_checkpoint": articles_per_ckpt,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "quantization": "QLoRA 4-bit",
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return stats


def main():
    print(f"\n{'='*72}")
    print(f"  LLAMA-3.2-3B CHECKPOINT TRAINING (QLoRA 4-bit, 4722 articles)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Checkpoint every: ~500 articles ({SAVE_STEPS} steps)")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}\n")

    output_dir = BASE_DIR / "output" / TAG
    checkpoint_dir = BASE_DIR / "checkpoints" / TAG

    train_data = json.loads(FULL_TRAIN_PATH.read_text())
    print(f"Loaded {len(train_data)} training examples")

    train_stats = train_model(train_data, output_dir, checkpoint_dir)

    print(f"\n{'='*72}")
    print(f"  DONE: loss={train_stats['train_loss']:.4f}, time={train_stats['train_runtime_seconds']/3600:.1f} hr")
    print(f"  Checkpoints saved to: {checkpoint_dir}/")
    print(f"  Final adapter saved to: {output_dir}/lora_adapter/")
    print(f"  Now run: python eval_checkpoints_llama32_3b.py")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
