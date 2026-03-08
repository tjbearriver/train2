#!/usr/bin/env python3
"""Query OpenRouter or local llama.cpp server with per-article prompts built from Wikipedia data.

Run with uvx, for example:
  # OpenRouter:
  uvx python batch_article_queries.py --models-file data/models-narrow.txt

  # Local llama-server:
  uvx python batch_article_queries.py --llama-server http://localhost:8080 --models local-model
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import signal
import socket
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal
from urllib import error, request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
QID_RE = re.compile(r"^Q\d+$")
ModelBackend = Literal["openrouter", "llama_server"]

# Graceful shutdown event — set by SIGINT/SIGTERM handler
_shutdown = threading.Event()

RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _install_signal_handlers() -> None:
    """Install signal handlers for graceful shutdown."""
    def _handler(signum: int, frame: object) -> None:
        if _shutdown.is_set():
            # Second signal — force exit
            print("\nForced exit.", file=sys.stderr, flush=True)
            os._exit(1)
        _shutdown.set()
        sig_name = signal.Signals(signum).name
        print(
            f"\n{sig_name} received — finishing in-flight requests, then stopping. "
            f"Press Ctrl+C again to force exit.",
            file=sys.stderr,
            flush=True,
        )

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

OPENROUTER_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = OPENROUTER_DIR.parent
PROVIDER_WHITELIST = [
    "anthropic",
    "arcee-ai",
    "baseten",
    "cerebras",
    "chutes",
    "crusoe",
    "deepinfra",
    "fireworks",
    "google",
    "google-ai-studio",
    "google-vertex",
    "liquid",
    "mistral",
    "openai",
    "parasail",
    "weights-and-biases",
    "wandb",
    "xai",
    "x-ai",
]

DEFAULT_QIDS_CSV = REPO_ROOT / "openrouter/data/llm-parse-test4/pageviews_humans_test.csv"
DEFAULT_ARTICLES = REPO_ROOT / "wikiparser/data/human_articles_en.ndjson"
DEFAULT_RELATIONSHIPS = OPENROUTER_DIR / "data/relationship-types.json"
DEFAULT_PROMPT = OPENROUTER_DIR / "data/llm-parse-test3/prompt1.txt"
DEFAULT_MODELS = OPENROUTER_DIR / "data/models-wide.txt"
DEFAULT_OUTPUT_DIR = OPENROUTER_DIR / "data/llm-parse-test3/results"


@dataclass
class ModelResult:
    model: str
    success: bool
    content: str | None = None
    raw_response: dict | None = None
    status_code: int | None = None
    error_message: str | None = None
    latency_seconds: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each Wikidata QID in a CSV, build a prompt with relationship types "
            "and the article text, then query multiple OpenRouter models."
        )
    )
    parser.add_argument(
        "--qids-csv",
        type=Path,
        default=DEFAULT_QIDS_CSV,
        help=f"CSV containing Wikidata QIDs (default: {DEFAULT_QIDS_CSV}).",
    )
    parser.add_argument(
        "--articles-ndjson",
        type=Path,
        default=DEFAULT_ARTICLES,
        help=f"NDJSON file of human articles (default: {DEFAULT_ARTICLES}).",
    )
    parser.add_argument(
        "--relationship-types",
        type=Path,
        default=DEFAULT_RELATIONSHIPS,
        help=f"Relationship types JSON file (default: {DEFAULT_RELATIONSHIPS}).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT,
        help=f"Base prompt file (default: {DEFAULT_PROMPT}).",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        help="Model identifiers (OpenRouter model names, or arbitrary names for local models).",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=DEFAULT_MODELS,
        help=f"File listing models to query (default: {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write per-article JSON results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of QIDs to process.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing/querying articles that already have a JSON output file.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant. Respond directly to the prompt.",
        help="Optional system message.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature to send to each model (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling probability (top_p). Leave unset to use model/provider default.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling cutoff (top_k). Leave unset to use model/provider default.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Minimum probability sampling (min_p). Leave unset to use model/provider default.",
    )
    parser.add_argument(
        "--top-a",
        type=float,
        default=None,
        help="Top-a sampling (top_a). Leave unset to use model/provider default.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Frequency penalty. Leave unset to use model/provider default.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty. Leave unset to use model/provider default.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty. Leave unset to use model/provider default.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate. Leave unset to let OpenRouter decide.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=360.0,
        help="Per-request timeout in seconds (default: 360).",
    )
    parser.add_argument(
        "--api-key",
        help="Explicit API key (otherwise the OPENROUTER_API_KEY environment variable is used).",
    )
    parser.add_argument(
        "--referer",
        default="https://github.com/tj/cranberry",
        help="HTTP Referer header required by OpenRouter terms (default: repo URL).",
    )
    parser.add_argument(
        "--title",
        default="cranberry-openrouter-cli",
        help="X-Title header for OpenRouter (default: cranberry-openrouter-cli).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and show the first prompt without calling the API.",
    )
    parser.add_argument(
        "--llama-server",
        help="URL of llama-server (e.g., http://localhost:8080). Uses whatever model is loaded in the server.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run each model (default: 1). Appends :1, :2, etc. to model names.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=100,
        help="Maximum number of parallel OpenRouter requests when --runs=1 (default: 100).",
    )
    parser.add_argument(
        "--prompt-pattern",
        default=None,
        metavar="PATTERN",
        help=(
            "Custom prompt layout using letters: B=Base prompt, R=Relationship types, "
            "A=Article data. Default is 'BRA'. Example: 'BRABAR' sends Base, Relationships, "
            "Article, Base, Article, Relationships."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Reasoning effort level to pass to OpenRouter (e.g. 'low', 'medium', 'high').",
    )
    return parser.parse_args()


def collect_models(models_args: list[str] | None, models_file: Path | None, runs: int = 1) -> list[str]:
    models: list[str] = []
    if models_args:
        for explicit in models_args:
            models.extend(token.strip() for token in explicit.split(","))
    if models_file:
        content = models_file.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            models.append(line)
    normalized = list(dict.fromkeys(model for model in (m.strip() for m in models) if model))
    
    # Expand models with run numbers
    if runs > 1:
        expanded = []
        for model in normalized:
            for run_num in range(1, runs + 1):
                expanded.append(f"{model}:{run_num}")
        return expanded
    
    return normalized


def determine_backend(args: argparse.Namespace) -> ModelBackend:
    """Determine which backend to use based on arguments."""
    if args.llama_server:
        return "llama_server"
    return "openrouter"


def ensure_api_key(args: argparse.Namespace) -> str:
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    raise SystemExit(
        textwrap.dedent(
            """
            Missing API key. Create one at https://openrouter.ai/keys when your account is ready,
            then either export OPENROUTER_API_KEY=<key> or pass --api-key <key>.
            """
        ).strip()
    )


def make_headers(api_key: str, referer: str, title: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": title,
    }


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def parse_model_spec(model_str: str) -> tuple[str, list[str] | None]:
    """Parse a model specification that may include explicit providers.

    Format: "model_name;provider1;provider2;..."
    Returns (model_name, providers) where providers is None if none specified.
    When providers are specified, the global PROVIDER_WHITELIST is bypassed.
    """
    parts = model_str.split(";")
    model_name = parts[0]
    if len(parts) > 1:
        providers = [p.strip() for p in parts[1:] if p.strip()]
        return model_name, providers if providers else None
    return model_name, None


def call_model(
    model: str,
    headers: dict[str, str],
    messages: list[dict[str, str]],
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    top_a: float | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    max_tokens: int | None,
    timeout: float,
    providers: list[str] | None = None,
    max_retries: int = 5,
    reasoning_effort: str | None = None,
) -> ModelResult:
    payload: dict[str, object] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if min_p is not None:
        payload["min_p"] = min_p
    if top_a is not None:
        payload["top_a"] = top_a
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if providers:
        payload["provider"] = {"only": providers}
    elif PROVIDER_WHITELIST:
        payload["provider"] = {"only": PROVIDER_WHITELIST}

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        if _shutdown.is_set():
            return ModelResult(
                model=model, success=False, error_message="Shutdown requested"
            )
        req = request.Request(API_URL, data=data, headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                latency = time.perf_counter() - start
                body = resp.read()
                return parse_response(model, body, resp.status, latency)
        except error.HTTPError as exc:
            latency = time.perf_counter() - start
            body = exc.read()
            message = f"HTTP {exc.code}: {body.decode('utf-8', errors='replace')}"
            if exc.code not in RETRYABLE_HTTP_CODES:
                return ModelResult(
                    model=model,
                    success=False,
                    error_message=message,
                    status_code=exc.code,
                    latency_seconds=latency,
                )
            if attempt < max_retries:
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after:
                        try:
                            delay = min(max(float(retry_after), 5), 300)
                        except (ValueError, TypeError):
                            delay = _backoff_delay(attempt)
                    else:
                        delay = _backoff_delay(attempt)
                else:
                    delay = _backoff_delay(attempt)
                print(
                    f"  HTTP error on attempt {attempt}/{max_retries} for {model}: {message}. "
                    f"Retrying in {delay:.0f}s...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                continue
            return ModelResult(
                model=model,
                success=False,
                error_message=message,
                status_code=exc.code,
                latency_seconds=latency,
            )
        except (error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            latency = time.perf_counter() - start
            message = f"Connection error: {exc}"
            if attempt < max_retries:
                delay = _backoff_delay(attempt)
                print(
                    f"  Network error on attempt {attempt}/{max_retries} for {model}: {message}. "
                    f"Retrying in {delay:.0f}s...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                continue
            return ModelResult(
                model=model,
                success=False,
                error_message=message,
                latency_seconds=latency,
            )
    # Should not reach here, but satisfy type checker
    return ModelResult(model=model, success=False, error_message="Max retries exhausted")


def _backoff_delay(attempt: int, base: float = 10.0, cap: float = 300.0) -> float:
    """Exponential backoff with jitter: ~10s, ~20s, ~40s, ~80s, ..."""
    return min(base * (2 ** (attempt - 1)) + random.uniform(0, 10), cap)


def parse_response(model: str, body: bytes, status_code: int, latency: float) -> ModelResult:
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ModelResult(
            model=model,
            success=False,
            error_message="Failed to parse JSON response",
            status_code=status_code,
            latency_seconds=latency,
        )

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ModelResult(
            model=model,
            success=False,
            raw_response=payload,
            status_code=status_code,
            latency_seconds=latency,
            error_message="No choices returned by OpenRouter",
        )
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    return ModelResult(
        model=model,
        success=True,
        content=content,
        raw_response=payload,
        status_code=status_code,
        latency_seconds=latency,
    )


def call_llama_server(
    model: str,
    server_url: str,
    messages: list[dict[str, str]],
    *,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float,
) -> ModelResult:
    """Call llama-server API (OpenAI-compatible endpoint)."""
    payload: dict[str, object] = {"messages": messages, "cache_prompt": False}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # Ensure server_url ends with the right endpoint
    if not server_url.endswith("/v1/chat/completions"):
        server_url = server_url.rstrip("/") + "/v1/chat/completions"

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = request.Request(server_url, data=data, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            latency = time.perf_counter() - start
            body = resp.read()
            return parse_response(model, body, resp.status, latency)
    except error.HTTPError as exc:
        latency = time.perf_counter() - start
        body = exc.read()
        message = f"HTTP {exc.code}: {body.decode('utf-8', errors='replace')}"
        return ModelResult(
            model=model,
            success=False,
            error_message=message,
            status_code=exc.code,
            latency_seconds=latency,
        )
    except error.URLError as exc:
        latency = time.perf_counter() - start
        return ModelResult(
            model=model,
            success=False,
            error_message=f"Connection error: {exc.reason}",
            latency_seconds=latency,
        )


def load_qids(path: Path, limit: int | None) -> list[str]:
    seen: set[str] = set()
    qids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            candidate = row[0].strip()
            if not QID_RE.match(candidate):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            qids.append(candidate)
            if limit is not None and len(qids) >= limit:
                break
    return qids


def load_articles(path: Path, qids: Iterable[str]) -> dict[str, dict]:
    wanted = set(qids)
    found: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not wanted:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            article_id = payload.get("id")
            if article_id in wanted:
                found[article_id] = payload
                wanted.remove(article_id)
    return found


PATTERN_LABELS = {"B": "Base Prompt", "R": "Relationship Types", "A": "Article Data"}
DEFAULT_PROMPT_PATTERN = "BRA"


def resolve_prompt_pattern(args: argparse.Namespace) -> str:
    """Return the validated prompt pattern string (upper-cased)."""
    raw = args.prompt_pattern or DEFAULT_PROMPT_PATTERN
    pattern = raw.upper()
    invalid = set(pattern) - set(PATTERN_LABELS)
    if invalid:
        raise SystemExit(
            f"Invalid characters in --prompt-pattern: {', '.join(sorted(invalid))}. "
            f"Allowed: B (Base Prompt), R (Relationship Types), A (Article Data)."
        )
    if not pattern:
        raise SystemExit("--prompt-pattern must not be empty.")
    return pattern


def build_prompt(
    base_prompt: str,
    relationship_types: str,
    article_payload: dict,
    pattern: str = DEFAULT_PROMPT_PATTERN,
) -> str:
    sections = {
        "B": base_prompt.rstrip(),
        "R": relationship_types.rstrip(),
        "A": json.dumps(article_payload, ensure_ascii=False, indent=2),
    }
    return "\n".join(sections[ch] for ch in pattern) + "\n"


def print_prompt_diagram(
    base_prompt: str,
    relationship_types: str,
    article_payload: dict,
    pattern: str,
) -> None:
    """Print a visual diagram of the prompt structure to stderr."""
    section_chars = {
        "B": len(base_prompt.rstrip()),
        "R": len(relationship_types.rstrip()),
        "A": len(json.dumps(article_payload, ensure_ascii=False, indent=2)),
    }

    rows = [(PATTERN_LABELS[ch], section_chars[ch]) for ch in pattern]

    label_w = max(len(r[0]) for r in rows)
    chars_w = max(len(f"{r[1]:,}") for r in rows) + len(" chars")
    inner_w = label_w + 2 + chars_w
    width = inner_w + 4  # borders + padding

    top    = "\u250c" + "\u2500" * (width - 2) + "\u2510"
    mid    = "\u251c" + "\u2500" * (width - 2) + "\u2524"
    bottom = "\u2514" + "\u2500" * (width - 2) + "\u2518"

    lines = [f"\nPrompt Pattern: {pattern}", top]
    for i, (label, chars) in enumerate(rows):
        right = f"{chars:,} chars"
        line = f"\u2502 {label:<{label_w}}  {right:>{chars_w}} \u2502"
        lines.append(line)
        if i < len(rows) - 1:
            lines.append(mid)
    lines.append(bottom)

    grand_total = sum(c for _, c in rows)
    lines.append(f"Total: {grand_total:,} chars")
    print("\n".join(lines), file=sys.stderr, flush=True)


def serialize_results(results: Iterable[ModelResult]) -> list[dict[str, object]]:
    return [
        {
            "model": r.model,
            "success": r.success,
            "content": r.content,
            "status_code": r.status_code,
            "latency_seconds": r.latency_seconds,
            "error_message": r.error_message,
            "raw_response": r.raw_response,
        }
        for r in results
    ]


def query_models(
    models: list[str],
    headers: dict[str, str],
    messages: list[dict[str, str]],
    *,
    backend: ModelBackend,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    top_a: float | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    max_tokens: int | None,
    timeout: float,
    llama_server: str | None = None,
    batch: int = 100,
    runs: int = 1,
    reasoning_effort: str | None = None,
) -> list[ModelResult]:
    total = len(models)
    index_lookup = {model: idx for idx, model in enumerate(models, start=1)}

    def query_single(model: str) -> ModelResult:
        if _shutdown.is_set():
            return ModelResult(
                model=model, success=False, error_message="Shutdown requested"
            )
        idx = index_lookup[model]
        print(f"[{idx}/{total}] Querying {model}...", file=sys.stderr, flush=True)
        
        # Strip run suffix (:1, :2, etc.) when making the actual API call
        actual_model = model.rsplit(':', 1)[0] if ':' in model and model.rsplit(':', 1)[1].isdigit() else model

        # Parse explicit providers from model spec (e.g. "model;provider1;provider2")
        actual_model, providers = parse_model_spec(actual_model)

        if backend == "openrouter":
            result = call_model(
                actual_model,
                headers,
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                top_a=top_a,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
                timeout=timeout,
                providers=providers,
                reasoning_effort=reasoning_effort,
            )
        elif backend == "llama_server":
            result = call_llama_server(
                actual_model,
                llama_server or "http://localhost:8080",
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        else:
            result = ModelResult(
                model=model,
                success=False,
                error_message=f"Unknown backend: {backend}",
            )
        
        # Update result model name to include run suffix
        result.model = model

        status = "ok" if result.success else "error"
        print(f"[{idx}/{total}] Finished {model}: {status}", file=sys.stderr, flush=True)
        return result

    if backend == "openrouter" and runs == 1:
        max_workers = min(len(models), max(batch, 1))
    else:
        cpu_count = max(os.cpu_count() or 1, 1)
        max_workers = min(len(models), max(cpu_count - 1, 1))
    results_map: dict[str, ModelResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for model in models:
            if _shutdown.is_set():
                break
            future_map[executor.submit(query_single, model)] = model
            time.sleep(random.uniform(0.05, 0.2))
        for future in as_completed(future_map):
            model = future_map[future]
            try:
                results_map[model] = future.result()
            except Exception as exc:  # pragma: no cover - safety net
                idx = index_lookup[model]
                print(
                    f"[{idx}/{total}] Failed {model}: unexpected error {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                results_map[model] = ModelResult(
                    model=model,
                    success=False,
                    error_message=str(exc),
                )
    # Fill in any models skipped due to shutdown
    for model in models:
        if model not in results_map:
            results_map[model] = ModelResult(
                model=model, success=False, error_message="Shutdown requested"
            )

    return [results_map[model] for model in models]


def write_article_result(
    output_dir: Path,
    article: dict,
    models: list[str],
    responses: list[ModelResult],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    article_id = article.get("id") or "unknown"
    output_path = output_dir / f"{article_id}.json"
    has_errors = any(not r.success for r in responses)
    payload = {
        "article": {
            "id": article.get("id"),
            "title": article.get("title"),
        },
        "has_errors": has_errors,
        "query_models": {
            "models": models,
            "responses": serialize_results(responses),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def run() -> int:
    _install_signal_handlers()
    args = parse_args()
    backend = determine_backend(args)

    # When using llama-server, default to a single "local" model if no models specified
    if backend == "llama_server":
        if args.models:
            models = collect_models(args.models, None, args.runs)
        else:
            # Expand "local" model with run numbers
            if args.runs > 1:
                models = [f"local:{i}" for i in range(1, args.runs + 1)]
            else:
                models = ["local"]
    else:
        models = collect_models(args.models, args.models_file, args.runs)
        if not models:
            raise SystemExit("Provide at least one model via --models or --models-file.")

    qids = load_qids(args.qids_csv, args.limit)
    if not qids:
        raise SystemExit(f"No QIDs found in {args.qids_csv}.")

    articles = load_articles(args.articles_ndjson, qids)
    if not articles:
        raise SystemExit(f"No matching articles found in {args.articles_ndjson}.")
    missing = [qid for qid in qids if qid not in articles]
    if missing:
        print(
            f"Warning: {len(missing)} QID(s) missing from {args.articles_ndjson}.",
            file=sys.stderr,
        )

    base_prompt = args.prompt_file.read_text(encoding="utf-8")
    relationship_types = args.relationship_types.read_text(encoding="utf-8")

    pattern = resolve_prompt_pattern(args)

    if args.dry_run:
        first_qid = next(iter(articles))
        prompt = build_prompt(base_prompt, relationship_types, articles[first_qid], pattern)
        print_prompt_diagram(base_prompt, relationship_types, articles[first_qid], pattern)
        try:
            print("DRY RUN ✔")
            print(f"Backend: {backend}")
            if backend == "llama_server":
                print(f"Server URL: {args.llama_server}")
            print(f"Models: {models}")
            if backend == "openrouter" and PROVIDER_WHITELIST:
                print(f"Provider whitelist: {PROVIDER_WHITELIST}")
            print(f"QIDs: {len(qids)} (showing {first_qid})")
            print(f"Prompt preview: {prompt!r}")
        except BrokenPipeError:
            return 0
        return 0

    # Only need API key for OpenRouter
    headers = {}
    if backend == "openrouter":
        api_key = ensure_api_key(args)
        headers = make_headers(api_key, args.referer, args.title)

    total_articles = len(qids)
    work_items: list[tuple[int, str, dict]] = []
    for idx, qid in enumerate(qids, start=1):
        article = articles.get(qid)
        if not article:
            print(f"[{idx}/{total_articles}] Skipping missing QID {qid}", file=sys.stderr)
            continue
        output_path = args.output_dir / f"{qid}.json"
        if args.skip_existing and output_path.exists():
            # Re-process files that had errors
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                if not existing.get("has_errors", False):
                    print(
                        f"[{idx}/{total_articles}] Skipping {qid}; output exists: {output_path}",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"[{idx}/{total_articles}] Re-processing {qid}; previous run had errors",
                    file=sys.stderr,
                )
            except (json.JSONDecodeError, OSError):
                pass  # re-process if file is corrupt
        work_items.append((idx, qid, article))

    # Print the prompt diagram once before the first query
    if work_items:
        first_article = work_items[0][2]
        print_prompt_diagram(base_prompt, relationship_types, first_article, pattern)

    def process_article(item: tuple[int, str, dict]) -> tuple[int, str]:
        nonlocal _inflight_count
        if _shutdown.is_set():
            return item[0], item[1]
        idx, qid, article = item
        prompt = build_prompt(base_prompt, relationship_types, article, pattern)
        messages = build_messages(args.system, prompt)
        with _progress_lock:
            _inflight_count += 1
        print(f"[{idx}/{total_articles}] Article {qid}", file=sys.stderr, flush=True)
        try:
            responses = query_models(
                models,
                headers,
                messages,
                backend=backend,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                top_a=args.top_a,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                llama_server=args.llama_server,
                batch=1,
                runs=args.runs,
                reasoning_effort=args.reasoning_effort,
            )
        finally:
            with _progress_lock:
                _inflight_count -= 1
        output_path = write_article_result(args.output_dir, article, models, responses)

        if any(not r.success for r in responses):
            failed_models = [r.model for r in responses if not r.success]
            print(
                f"[{idx}/{total_articles}] Errors in {qid} for: {', '.join(failed_models)} "
                f"(results saved with has_errors=true, will be retried with --skip-existing)",
                file=sys.stderr,
                flush=True,
            )

        return idx, qid

    # Progress tracking
    _progress_lock = threading.Lock()
    _completed_count = 0
    _success_count = 0
    _error_count = 0
    _inflight_count = 0
    _start_time = time.monotonic()
    _progress_interval = max(len(work_items) // 20, 50)  # ~every 5% or 50 articles

    def _update_progress(qid: str, had_error: bool) -> None:
        nonlocal _completed_count, _success_count, _error_count
        with _progress_lock:
            _completed_count += 1
            if had_error:
                _error_count += 1
            else:
                _success_count += 1
            if _completed_count % _progress_interval == 0 or _completed_count == len(work_items):
                elapsed = time.monotonic() - _start_time
                rate = _completed_count / elapsed if elapsed > 0 else 0
                remaining = (len(work_items) - _completed_count) / rate if rate > 0 else 0
                elapsed_m = elapsed / 60
                remaining_m = remaining / 60
                print(
                    f"  Progress: {_completed_count}/{len(work_items)} "
                    f"({100 * _completed_count / len(work_items):.0f}%) | "
                    f"OK: {_success_count} | Errors: {_error_count} | "
                    f"In-flight: {_inflight_count} | "
                    f"Elapsed: {elapsed_m:.1f}m | ETA: {remaining_m:.1f}m",
                    file=sys.stderr,
                    flush=True,
                )

    if backend == "openrouter" and args.runs == 1:
        worker_count = min(len(work_items), max(args.batch, 1)) if work_items else 1
        print(
            f"Processing {len(work_items)} article(s) with up to {worker_count} parallel OpenRouter worker(s).",
            file=sys.stderr,
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {}
            for item in work_items:
                if _shutdown.is_set():
                    break
                future_map[executor.submit(process_article, item)] = item
                time.sleep(random.uniform(0.05, 0.2))
            for future in as_completed(future_map):
                idx, qid, _article = future_map[future]
                try:
                    future.result()
                    output_path = args.output_dir / f"{qid}.json"
                    had_error = False
                    if output_path.exists():
                        try:
                            data = json.loads(output_path.read_text(encoding="utf-8"))
                            had_error = data.get("has_errors", False)
                        except (json.JSONDecodeError, OSError):
                            had_error = True
                    _update_progress(qid, had_error)
                except Exception as exc:  # pragma: no cover - safety net
                    print(
                        f"[{idx}/{total_articles}] Failed {qid}: unexpected error {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    _update_progress(qid, True)
    else:
        for item in work_items:
            if _shutdown.is_set():
                break
            process_article(item)

    # Final summary
    elapsed_total = time.monotonic() - _start_time
    print(
        f"\nDone. {_completed_count}/{len(work_items)} articles processed in {elapsed_total / 60:.1f}m "
        f"(OK: {_success_count}, Errors: {_error_count})"
        + (" [interrupted by shutdown]" if _shutdown.is_set() else ""),
        file=sys.stderr,
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())