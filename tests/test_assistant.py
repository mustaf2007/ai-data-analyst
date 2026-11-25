import pandas as pd

from assistant import answer_dataset_question, suggest_analyses


def test_suggestions_detect_missing_values_and_correlations():
    df = pd.DataFrame({
        "sales": [10.0, None, 30.0],
        "cost": [4.0, 5.0, 8.0],
        "region": ["North", "South", "North"],
    })
    suggestions = suggest_analyses(df)
    titles = {item["title"] for item in suggestions}
    assert "Review missing values" in titles
    assert "Explore relationships" in titles
    assert all({"title", "reason", "action", "priority"} <= set(item) for item in suggestions)


def test_suggestions_empty_dataset_requests_upload():
    result = suggest_analyses(pd.DataFrame())
    assert result[0]["title"] == "Load a dataset"


def test_chatbot_answers_mean_using_actual_values():
    df = pd.DataFrame({"sales": [10, 20, 30]})
    result = answer_dataset_question(df, "What is the average sales?")
    assert result["intent"] == "mean"
    assert "20" in result["answer"]
    assert result["evidence"].iloc[0]["mean"] == 20


def test_chatbot_answers_grouped_highest():
    df = pd.DataFrame({
        "region": ["North", "North", "South", "South"],
        "sales": [10, 20, 50, 60],
    })
    result = answer_dataset_question(df, "Which region has the highest sales?")
    assert result["intent"] == "grouped"
    assert "South" in result["answer"]
    assert result["evidence"].iloc[0]["region"] == "South"


def test_chatbot_reports_missing_and_duplicate_rows():
    df = pd.DataFrame({"x": [1, 1, None], "y": ["a", "a", "b"]})
    missing = answer_dataset_question(df, "How many missing values?")
    duplicates = answer_dataset_question(df, "Are there duplicate rows?")
    assert missing["intent"] == "missing"
    assert duplicates["intent"] == "duplicates"
    assert "1" in missing["answer"]
    assert "1" in duplicates["answer"]


def test_chatbot_handles_correlations_and_unsupported_questions():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    result = answer_dataset_question(df, "What is the correlation between variables?")
    assert result["intent"] == "correlation"
    assert result["evidence"].iloc[0]["correlation"] > 0.99
    unsupported = answer_dataset_question(df, "Write me a poem")
    assert unsupported["intent"] == "unsupported"
