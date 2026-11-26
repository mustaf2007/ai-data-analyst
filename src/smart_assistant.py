"""Smart assistant with optional local Ollama and explicitly enabled API mode."""
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import pandas as pd

from assistant import answer_dataset_question


def build_dataset_context(df: pd.DataFrame, source: str = "Active dataset") -> str:
    """Create a compact, privacy-conscious analytical context; no raw rows are sent."""
    numeric = df.select_dtypes(include="number")
    lines = [
        f"Dataset: {source}",
        f"Dimensions: {len(df):,} rows x {len(df.columns):,} columns",
        "Columns and data types:",
    ]
    for col in df.columns[:80]:
        lines.append(f"- {col}: {df[col].dtype}; missing={int(df[col].isna().sum()):,}; unique={int(df[col].nunique(dropna=True)):,}")
    if not numeric.empty:
        lines.append("Numeric descriptive statistics:")
        lines.append(numeric.describe().round(3).to_string(max_rows=12, max_cols=12))
    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 30]
    for col in categorical[:10]:
        counts = df[col].astype("string").fillna("(Missing)").value_counts().head(8)
        lines.append(f"Top values for {col}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    lines.append("Privacy note: this context contains schema and aggregate summaries, not raw record rows.")
    return "\n".join(lines)


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 90) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"AI service returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach the selected AI service: {exc}") from exc


def ask_smart_assistant(
    question: str,
    history: list[dict] | None = None,
    dataset_context: str = "",
    provider: str = "Dataset tools (no AI server)",
) -> dict:
    """Answer general or dataset-aware questions using the selected provider.

    Ollama runs locally. OpenAI-compatible mode is opt-in and requires environment
    configuration; no credentials are stored in the repository or UI state.
    """
    question = str(question or "").strip()
    if not question:
        raise ValueError("Enter a question first.")

    if provider == "Dataset tools (no AI server)":
        result = answer_dataset_question_from_context(question)
        return {"answer": result, "provider": provider}

    system = (
        "You are a careful, professional AI research and data-analysis assistant. "
        "Answer general questions clearly and help with research, statistics, finance, "
        "policy, programming, and technical writing. When dataset context is supplied, "
        "use only the evidence it contains; do not invent values or claim you inspected "
        "raw records. Distinguish facts, assumptions, estimates and recommendations. "
        "For financial analysis, distinguish portfolio return from benchmark return, "
        "state formulas and time periods, and avoid personalized investment advice. "
        "Use a professional technical-report tone when asked to draft reports."
    )
    if dataset_context:
        system += "\n\nACTIVE DATASET CONTEXT (aggregates/schema only):\n" + dataset_context[:18000]

    messages = [{"role": "system", "content": system}]
    for item in (history or [])[-10:]:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append({"role": item["role"], "content": str(item["content"])[:6000]})
    messages.append({"role": "user", "content": question})

    if provider == "Ollama (local model)":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        result = _post_json(
            base + "/api/chat",
            {"model": model, "messages": messages, "stream": False,
             "options": {"temperature": 0.2}},
        )
        answer = result.get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("The local model returned an empty response.")
        return {"answer": answer, "provider": f"Ollama · {model}"}

    if provider == "OpenAI-compatible API (opt-in)":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Add it as a deployment secret, or use the local provider.")
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        result = _post_json(
            base + "/chat/completions",
            {"model": model, "messages": messages, "temperature": 0.2},
            headers={"Authorization": "Bearer " + key},
        )
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("The configured API returned an empty response.")
        return {"answer": answer, "provider": f"API · {model}"}

    raise ValueError("Select a supported assistant provider.")


def answer_dataset_question_from_context(question: str) -> str:
    """Dataset-only fallback for questions supported by the deterministic engine."""
    # The caller uses the active Data tab for exact computed evidence. This fallback
    # deliberately avoids guessing when no dataframe was passed to the service.
    return (
        "The no-server mode is limited to deterministic dataset questions in the "
        "Ask the Data workspace. For general questions, select Ollama (local model) "
        "or configure an OpenAI-compatible API. For accurate dataset answers, use "
        "Ask the Data tab, which calculates directly from the active DataFrame."
    )
