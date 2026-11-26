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
    assert default_assistant_provider() == "Dataset tools (no AI server)"



def test_default_provider_uses_model_free_dataset_assistant():
    assert default_assistant_provider() == "Dataset tools (no AI server)"


def test_free_provider_requires_at_least_one_key(monkeypatch):
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="No free AI provider is configured"):
        ask_smart_assistant(
            "Explain regression",
            provider="Free hosted AI (automatic fallback)",
            free_api_keys={},
        )


def test_free_provider_falls_back_to_next_configured_provider(monkeypatch):
    import smart_assistant

    calls = []

    def fake_post_json(url, payload, headers=None, timeout=90):
        calls.append(url)
        if "api.groq.com" in url:
            raise RuntimeError("Groq temporarily unavailable")
        return {"choices": [{"message": {"content": "A helpful answer."}}]}

    monkeypatch.setattr(smart_assistant, "_post_json", fake_post_json)
    result = ask_smart_assistant(
        "Explain regression",
        provider="Free hosted AI (automatic fallback)",
        free_api_keys={
            "Groq": "test-groq-key",
            "OpenRouter": "test-openrouter-key",
        },
    )
    assert result["answer"] == "A helpful answer."
    assert result["provider"].startswith("OpenRouter")
    assert len(calls) == 2



def test_gemini_retries_with_current_model_after_retired_model(monkeypatch):
    import smart_assistant

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    attempted_models = []

    def fake_post_json(url, payload, headers=None, timeout=90):
        model = payload["model"]
        attempted_models.append(model)
        if model == "gemini-2.5-flash":
            raise RuntimeError(
                "AI service returned HTTP 404: model is no longer available"
            )
        return {"choices": [{"message": {"content": "Gemini fallback worked."}}]}

    monkeypatch.setattr(smart_assistant, "_post_json", fake_post_json)
    result = ask_smart_assistant(
        "Explain regression",
        provider="Free hosted AI (automatic fallback)",
        free_api_keys={"Google Gemini": "test-gemini-key"},
    )
    assert attempted_models == ["gemini-2.5-flash", "gemini-3.8-flash"]
    assert result["answer"] == "Gemini fallback worked."
    assert result["provider"].endswith("gemini-3.8-flash")


def test_dataset_tools_provider_answers_without_any_api_key(monkeypatch):
    for key in ("OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    df = pd.DataFrame({"sales": [10, 20, 30]})
    result = ask_smart_assistant(
        "What is the average sales?",
        provider="Dataset tools (no AI server)",
        df=df,
    )
    assert result["provider"] == "Dataset tools (no AI server)"
    assert "20" in result["answer"]
    assert result["evidence"].iloc[0]["mean"] == 20


def test_model_free_assistant_interprets_dataset_with_descriptive_evidence():
    df = pd.DataFrame({
        "year": [2021, 2022, 2023, 2024],
        "gdp": [100.0, 110.0, 125.0, 140.0],
        "country": ["A", "A", "B", "B"],
    })
    result = ask_smart_assistant(
        "Explain the most important patterns in this dataset.",
        provider="Dataset tools (no AI server)",
        df=df,
    )
    assert result["provider"] == "Dataset tools (no AI server)"
    assert "4 records and 3 variables" in result["answer"]
    assert "Numeric highlights" in result["answer"]
    assert result["evidence"].set_index("column").loc["gdp", "mean"] == 118.75
    assert result["context"] is not None
