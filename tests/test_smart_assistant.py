import pandas as pd
import pytest

from smart_assistant import build_dataset_context, ask_smart_assistant, default_assistant_provider


def test_dataset_context_contains_schema_and_aggregates_not_rows():
    df = pd.DataFrame({"sales": [10, 20, 30], "region": ["N", "S", "N"]})
    context = build_dataset_context(df, "test.csv")
    assert "3 rows x 2 columns" in context
    assert "sales: int64" in context
    assert "mean" in context
    assert "Privacy note" in context
    assert "10, 20, 30" not in context


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        ask_smart_assistant(" ")


def test_api_provider_requires_secret(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        ask_smart_assistant("Explain regression", provider="OpenAI-compatible API (opt-in)")


def test_dataset_context_includes_grouped_aggregates():
    df = pd.DataFrame({
        "sales": [10, 20, 30, 40],
        "region": ["North", "North", "South", "South"],
    })
    context = build_dataset_context(df)
    assert "Grouped numeric summaries by region" in context
    assert "North" in context and "South" in context


def test_default_provider_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert default_assistant_provider() == "Ollama (local model)"
