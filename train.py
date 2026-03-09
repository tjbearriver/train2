#!/usr/bin/env python3
"""QLoRA fine-tuning script using Unsloth for relationship extraction distillation."""

import json
import time
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "data" / "train.json"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"

BASE_MODEL = "unsloth/Qwen3-8B-bnb-4bit"
MAX_SEQ_LENGTH = 8192

# QLoRA config
LORA_R = 64
LORA_ALPHA = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training config
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_8bit"
SAVE_STEPS = 200
LOGGING_STEPS = 10


def formatting_func(examples):
    """Format dataset examples into chat conversations for the tokenizer."""
    texts = []
    for convos in examples["conversations"]:
        text = tokenizer.apply_chat_template(convos, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


def main():
    global tokenizer

    print("=== Loading training data ===")
    with open(TRAIN_PATH) as f:
        train_data = json.load(f)
    print(f"Training examples: {len(train_data)}")

    print("\n=== Loading model ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    print("\n=== Adding LoRA adapters ===")
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

    print("\n=== Preparing dataset ===")
    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(formatting_func, batched=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIMIZER,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
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

    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("\n=== Starting training ===")
    total_steps = (len(train_data) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"Total optimizer steps: ~{total_steps}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Grad accum: {GRAD_ACCUM}")

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    print(f"\n=== Training complete in {elapsed/3600:.1f} hours ===")
    print(f"Train loss: {train_result.training_loss:.4f}")

    # Save final adapter
    final_path = str(OUTPUT_DIR / "lora_adapter")
    print(f"\nSaving LoRA adapter to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save training stats
    stats = {
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": elapsed,
        "train_samples": len(train_data),
        "epochs": EPOCHS,
        "total_steps": total_steps,
        "model": BASE_MODEL,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    with open(OUTPUT_DIR / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
