import json

import pandas as pd
import pytest

import hybrid_chatbot as chatbot


def test_parse_plan_rejects_invalid_json_and_unknown_operation():
    with pytest.raises(ValueError, match="valid structured"):
        chatbot._parse_plan("not json")
    with pytest.raises(ValueError, match="not supported"):
        chatbot._parse_plan('{"operation":"execute_python"}')


def test_aggregate_executes_validated_plan():
    df = pd.DataFrame({"sales": [10, 20, 30]})
    result = chatbot._execute_plan(df, {"operation": "aggregate", "column": "sales", "statistic": "mean"})
    assert result["evidence"].iloc[0]["value"] == 20
    assert "20" in result["answer"]


def test_groupby_top_n_and_filter_are_computed_with_pandas():
    df = pd.DataFrame({"region": ["North", "North", "South"], "sales": [10, 20, 50]})
    result = chatbot._execute_plan(df, {
        "operation": "top_n", "column": "sales", "group_by": "region",
        "statistic": "sum", "sort": "desc", "limit": 1,
    })
    assert result["evidence"].iloc[0]["region"] == "South"
    filtered = chatbot._execute_plan(df, {
        "operation": "filter", "filter_column": "sales",
        "filter_operator": "gte", "filter_value": 20,
    })
    assert len(filtered["evidence"]) == 2
    assert "2 row(s)" in filtered["answer"]


def test_plan_cannot_reference_unknown_columns_or_unsafe_operators():
    df = pd.DataFrame({"sales": [10, 20]})
    with pytest.raises(ValueError, match="unknown column"):
        chatbot._execute_plan(df, {"operation": "aggregate", "column": "password", "statistic": "mean"})
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        chatbot._execute_plan(df, {
            "operation": "filter", "filter_column": "sales",
            "filter_operator": "exec", "filter_value": "x",
        })


def test_correlation_and_temporal_trend_are_calculated():
    df = pd.DataFrame({
        "Year": [2022, 2023, 2024],
        "sales": [10, 20, 30],
        "cost": [5, 10, 15],
    })
    corr = chatbot._execute_plan(df, {"operation": "correlation"})
    assert corr["evidence"].iloc[0]["correlation"] == pytest.approx(1.0)
    trend = chatbot._execute_plan(df, {"operation": "trend", "column": "sales"})
    assert trend["evidence"]["value"].tolist() == [10, 20, 30]
    assert trend["evidence"]["period"].tolist() == ["2022", "2023", "2024"]


def test_hybrid_uses_structured_plan_and_keeps_calculation_in_pandas(monkeypatch):
    df = pd.DataFrame({"sales": [10, 20, 30]})
    monkeypatch.setattr(
        chatbot, "_provider_plan",
        lambda question, data, provider, keys, history=None: (
            {"operation": "aggregate", "column": "sales", "statistic": "median"},
            "Mock provider",
        ),
    )
    result = chatbot.answer_hybrid_question("What is the typical sales?", df)
    assert result["evidence"].iloc[0]["value"] == 20
    assert result["provider"] == "Mock provider"
    assert result["plan"]["operation"] == "aggregate"


def test_hybrid_requires_a_non_empty_dataset():
    with pytest.raises(ValueError, match="non-empty dataset"):
        chatbot.answer_hybrid_question("summarise", pd.DataFrame())



def test_provider_plan_sends_schema_metadata_not_raw_rows(monkeypatch):
    sent = {}

    def fake_completion(base_url, model, messages, key):
        sent["messages"] = messages
        return '{"operation":"aggregate","column":"amount","statistic":"sum"}'

    monkeypatch.setattr(chatbot, "_chat_completion", fake_completion)
    df = pd.DataFrame({"amount": [101, 202], "customer_note": ["private-row-a", "private-row-b"]})
    plan, provider = chatbot._provider_plan(
        "total amount", df, "Free hosted AI (automatic fallback)",
        {"Groq": "test-key"},
        history=[{"role": "user", "content": "Earlier question"}],
    )
    payload = "\n".join(item["content"] for item in sent["messages"])
    assert plan["operation"] == "aggregate"
    assert "private-row-a" not in payload and "private-row-b" not in payload
    assert "customer_note" in payload
    assert "Earlier question" in payload
    assert provider.startswith("Groq")


def test_hybrid_planner_receives_recent_conversation_history(monkeypatch):
    df = pd.DataFrame({"sales": [10, 20]})
    seen = {}

    def fake_plan(question, data, provider, keys, history=None):
        seen["history"] = history
        return {"operation": "aggregate", "column": "sales", "statistic": "mean"}, "Mock provider"

    monkeypatch.setattr(chatbot, "_provider_plan", fake_plan)
    chatbot.answer_hybrid_question(
        "What about the average?", df, history=[{"role": "user", "content": "Tell me about sales"}],
    )
    assert seen["history"][0]["content"] == "Tell me about sales"



def test_duplicate_column_names_are_rejected():
    df = pd.DataFrame([[1, 2]], columns=["value", "value"])
    with pytest.raises(ValueError, match="duplicate column names"):
        chatbot.answer_hybrid_question("summarise", df)
