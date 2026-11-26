import pandas as pd
import pytest

from research_reporting import build_research_report, build_research_excel, build_research_pdf


def sample():
    return pd.DataFrame({
        "region": ["North"] * 8 + ["South"] * 8,
        "income": [10, 11, 9, 12, 10, 11, 9, 10, 20, 19, 21, 18, 20, 22, 19, 21],
        "employment": ["Employed", "Unemployed"] * 8,
    })


def test_research_report_uses_question_and_evidence():
    report = build_research_report(sample(), "Does income differ by region?",
                                   objectives="Compare regional income.", alpha=0.05)
    assert report["research_question"] == "Does income differ by region?"
    assert report["test"]["test"] == "Welch independent-samples t-test"
    assert report["test"]["p_value"] < 0.05
    assert not report["numeric_summary"].empty
    assert not report["charts"] == []
    assert "Methodology" in report["markdown"]
    assert not report["recommendations"].empty


def test_research_report_categorical_only_does_not_invent_test():
    report = build_research_report(pd.DataFrame({"status": ["open", "closed", "open"]}),
                                   "What is the distribution of status?")
    assert report["test"]["test"] == "Descriptive analysis only"


def test_research_report_rejects_empty_or_missing_question():
    with pytest.raises(ValueError):
        build_research_report(pd.DataFrame(), "Question?")
    with pytest.raises(ValueError):
        build_research_report(sample(), " ")


def test_research_excel_contains_evidence_sheets():
    report = build_research_report(sample(), "Does income differ by region?")
    data = build_research_excel(sample(), report)
    assert data[:2] == b"PK"


def test_research_pdf_is_created():
    report = build_research_report(sample(), "Does income differ by region?")
    data = build_research_pdf(report)
    assert data.startswith(b"%PDF")


def test_research_report_applies_selected_alpha_to_interpretation_and_priority(monkeypatch):
    from types import SimpleNamespace
    import research_reporting

    monkeypatch.setattr(
        research_reporting.stats,
        "ttest_ind",
        lambda *args, **kwargs: SimpleNamespace(statistic=2.1, pvalue=0.04),
    )
    data = sample()
    strict = build_research_report(data, "Does income differ by region?", alpha=0.01)
    lenient = build_research_report(data, "Does income differ by region?", alpha=0.10)

    assert strict["test"]["significant_at_alpha"] is False
    assert "Insufficient evidence" in strict["test"]["interpretation"]
    assert strict["recommendations"].iloc[0]["Priority"] == "Medium"
    assert lenient["test"]["significant_at_alpha"] is True
    assert "Evidence of a difference" in lenient["test"]["interpretation"]
    assert lenient["recommendations"].iloc[0]["Priority"] == "High"


def test_research_report_selects_two_distinct_numeric_variables_for_correlation():
    data = pd.DataFrame({
        "income": [1, 2, 3, 4, 5, 6],
        "expenditure": [2, 4, 5, 8, 9, 12],
    })
    report = build_research_report(data, "What is the relationship involving income?")
    assert report["test"]["test"] == "Pearson correlation test"
    assert {report["test"]["outcome"], report["test"]["group"]} == {"income", "expenditure"}
