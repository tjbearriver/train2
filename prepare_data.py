#!/usr/bin/env python3
"""Prepare training data: clean golden CSV, join with articles, split train/eval, format for SFT."""

import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.9
MAX_SEQ_TOKENS_ESTIMATE = 16384
CHARS_PER_TOKEN = 4  # rough estimate

RELATIONSHIP_PATTERN = re.compile(r"^[A-Z\-]+ *\([^/]+/[^)]+\)$")
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "llm-parse-test7d-openrouter.csv"
NDJSON_PATH = BASE_DIR / "human_articles_en.ndjson"
PROMPT_PATH = BASE_DIR / "prompt4_v41_v2.txt"
REL_TYPES_PATH = BASE_DIR / "relationship-types.json"
OUTPUT_DIR = BASE_DIR / "data"


def load_prompt_and_rel_types() -> str:
    """Build the system message from prompt + relationship types."""
    prompt_text = PROMPT_PATH.read_text().strip()
    rel_types_text = REL_TYPES_PATH.read_text().strip()
    return f"{prompt_text}\n\nRelationship Types:\n{rel_types_text}"


def parse_golden_csv() -> dict[str, list[dict]]:
    """Parse the golden CSV into {qid: [rows...]}. Drops malformed rows."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    total = 0
    dropped = 0

    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            total += 1
            # Need at least: qid, model, provider, cost, name, relationship, evidence(1+), confidence
            if len(row) < 8:
                dropped += 1
                continue

            qid = row[0].strip()
            name = row[4].strip()
            rel_type = row[5].strip()
            confidence = row[-1].strip()
            evidence = ", ".join(s.strip() for s in row[6:-1])

            # Validate
            if not qid.startswith("Q"):
                dropped += 1
                continue
            if confidence not in VALID_CONFIDENCE:
                dropped += 1
                continue
            if not RELATIONSHIP_PATTERN.match(rel_type):
                dropped += 1
                continue
            if not name:
                dropped += 1
                continue

            grouped[qid].append({
                "name": name,
                "relationship": rel_type,
                "evidence": evidence,
            })

    print(f"CSV: {total} rows, {dropped} dropped, {total - dropped} kept, {len(grouped)} unique articles")
    return dict(grouped)


def load_articles(qids: set[str]) -> dict[str, dict]:
    """Load article data for the given QIDs from the NDJSON file."""
    articles = {}
    with open(NDJSON_PATH) as f:
        for line in f:
            obj = json.loads(line)
            if obj["id"] in qids:
                articles[obj["id"]] = {"title": obj["title"], "text": obj["text"]}
                if len(articles) == len(qids):
                    break
    print(f"Articles: loaded {len(articles)} of {len(qids)} requested")
    return articles


def estimate_tokens(system_msg: str, article_text: str, title: str, csv_output: str) -> int:
    """Rough token estimate for the full conversation."""
    total_chars = len(system_msg) + len(f"Article Data:\n\nTitle: {title}\n\n{article_text}") + len(csv_output)
    return total_chars // CHARS_PER_TOKEN


def build_csv_output(rows: list[dict]) -> str:
    """Build the assistant's CSV output from relationship rows (no confidence)."""
    lines = []
    for r in rows:
        lines.append(f"{r['name']}, {r['relationship']}, {r['evidence']}")
    return "\n".join(lines)


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=== Loading system prompt + relationship types ===")
    system_msg = load_prompt_and_rel_types()
    print(f"System message: {len(system_msg)} chars")

    print("\n=== Parsing golden CSV ===")
    golden = parse_golden_csv()

    print("\n=== Loading article texts ===")
    articles = load_articles(set(golden.keys()))

    # Only keep QIDs that have both golden data and article text
    valid_qids = set(golden.keys()) & set(articles.keys())
    print(f"\nQIDs with both golden + article: {len(valid_qids)}")

    # Build examples and filter by token length
    print("\n=== Building training examples ===")
    examples = []
    filtered_too_long = 0

    for qid in sorted(valid_qids):
        article = articles[qid]
        rows = golden[qid]
        csv_output = build_csv_output(rows)
        user_msg = f"Article Data:\n\nTitle: {article['title']}\n\n{article['text']}"

        tok_est = estimate_tokens(system_msg, article["text"], article["title"], csv_output)
        if tok_est > MAX_SEQ_TOKENS_ESTIMATE:
            filtered_too_long += 1
            continue

        examples.append({
            "qid": qid,
            "title": article["title"],
            "conversations": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": csv_output},
            ],
        })

    print(f"Total examples: {len(examples)}, filtered (too long): {filtered_too_long}")

    # Stratified-ish split by shuffling and splitting
    random.shuffle(examples)
    split_idx = int(len(examples) * TRAIN_RATIO)
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]

    print(f"\nTrain: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Save
    train_path = OUTPUT_DIR / "train.json"
    eval_path = OUTPUT_DIR / "eval.json"

    with open(train_path, "w") as f:
        json.dump(train_examples, f, ensure_ascii=False)
    with open(eval_path, "w") as f:
        json.dump(eval_examples, f, ensure_ascii=False)

    # Also save a small stats summary
    stats = {
        "total_golden_qids": len(golden),
        "articles_found": len(articles),
        "filtered_too_long": filtered_too_long,
        "total_examples": len(examples),
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "max_seq_tokens": MAX_SEQ_TOKENS_ESTIMATE,
        "seed": SEED,
    }
    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved: {train_path}, {eval_path}, {stats_path}")
    print("Done!")


if __name__ == "__main__":
    main()
