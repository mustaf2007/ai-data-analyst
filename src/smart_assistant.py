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
    # Compact group-by evidence helps the model explain segment performance without
    # exposing row-level records.
    if not numeric.empty:
        group_columns = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and 1 < df[c].nunique(dropna=True) <= 20
        ]
        for group_col in group_columns[:4]:
            grouped = df.groupby(group_col, dropna=False)[numeric.columns[:5].tolist()].agg(["count", "mean", "sum"])
            lines.append(f"Grouped numeric summaries by {group_col} (count, mean, sum):")
            lines.append(grouped.round(3).to_string(max_rows=12, max_cols=15))
    lines.append("Privacy note: this context contains schema and aggregate summaries, not raw record rows.")
    return "\n".join(lines)


def _setting(name: str, default: str = "") -> str:
    """Read deployment environment variables or Streamlit secrets safely."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def default_assistant_provider() -> str:
    """Use the no-API-key open-source model path by default."""
    return "Ollama (local model)"


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
    df: pd.DataFrame | None = None,
    context: dict | None = None,
) -> dict:
    """Answer general or dataset-aware questions using the selected provider.

    Ollama runs locally. OpenAI-compatible mode is opt-in and requires environment
    configuration; no credentials are stored in the repository or UI state.
    """
    question = str(question or "").strip()
    if not question:
        raise ValueError("Enter a question first.")

    if provider == "Dataset tools (no AI server)":
        if df is None:
            return {"answer": "Load a dataset to use deterministic data questions. For general questions, select Ollama (local model) or configure an OpenAI-compatible API.", "provider": provider}
        result = answer_dataset_question(df, question, context=context)
        return {"answer": result["answer"], "provider": provider, "evidence": result.get("evidence"), "context": result.get("context")}

    system = (
        "You are a capable, general-purpose conversational AI assistant, not a "
        "dataset-only chatbot. Answer open-ended questions across everyday knowledge, "
        "science, technology, education, research, economics, public policy, finance, "
        "statistics, programming, writing, planning and other topics. Do not reject a "
        "question merely because it is outside data analysis or the active dataset. "
        "Adapt the depth and format to the user's request, ask a focused clarification "
        "only when essential, and be transparent when information is uncertain or may "
        "be out of date. When dataset context is supplied, "
        "use only the evidence it contains; do not invent values or claim you inspected "
        "raw records. Distinguish facts, assumptions, estimates and recommendations. "
        "For financial analysis, distinguish portfolio return from benchmark return, "
        "state formulas and time periods, and avoid personalized investment advice. "
        "When drafting a technical report, structure it with a title, executive summary, "
        "background and objectives, scope, data and methodology, findings with quantified "
        "evidence, interpretation, risks and limitations, recommendations, conclusion, "
        "and appendices or sources where supplied. For investment-performance reports, "
        "separate total-portfolio return from invested-sleeve return, benchmark comparison, "
        "holdings attribution, transaction costs, allocation effects and risk. State the "
        "reporting dates, formulas and assumptions. Never invent missing figures or sources. "
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
        base = _setting("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = _setting("OLLAMA_MODEL", "llama3.1:8b")
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
        key = _setting("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Add it as a deployment secret, or use the local provider.")
        base = _setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = _setting("OPENAI_MODEL", "gpt-4o-mini")
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


