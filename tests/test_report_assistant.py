import pandas as pd
import pytest

from report_assistant import build_narrative_report


def test_narrative_report_contains_dataset_evidence_and_caveats():
    df = pd.DataFrame({
        "region": ["North", "North", "South", "South"],
        "sales": [10, 20, 30, 40],
        "orders": [1, 2, 3, 4],
    })
    report = build_narrative_report(df, "sales.csv")
    assert "4 rows and 3 columns" in report["executive_summary"]
    assert not report["numeric_summary"].empty
    assert not report["category_summary"].empty
    assert not report["correlations"].empty
    assert "not evidence of causation" in report["findings"][-1]
    assert "sampling representativeness" in report["methodology"]
    assert "# Data Analysis Report" in report["markdown"]


def test_narrative_report_flags_missing_values_and_duplicates():
    df = pd.DataFrame({"value": [1, 1, None], "group": ["A", "A", "B"]})
    report = build_narrative_report(df)
    assert report["profile"]["missing_cells"] == 1
    assert report["profile"]["duplicate_rows"] == 1
    assert any("missing cells" in x for x in report["recommendations"])
    assert any("duplicated rows" in x for x in report["recommendations"])


def test_narrative_report_handles_categorical_only_data():
    report = build_narrative_report(pd.DataFrame({"status": ["open", "closed", "open"]}))
    assert report["numeric_summary"].empty
    assert not report["category_summary"].empty
    assert "No numeric variables" in " ".join(report["recommendations"])


def test_narrative_report_rejects_empty_data():
    with pytest.raises(ValueError):
        build_narrative_report(pd.DataFrame())
