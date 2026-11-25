import pandas as pd
from analytics import numeric_summary, profile_dataset, generate_insights
from reporting import build_pdf_report


def test_pdf_report_is_generated():
    df = pd.DataFrame({"sales": [10, 20, 30], "orders": [1, 2, 3]})
    pdf = build_pdf_report(
        "sample.csv", df, profile_dataset(df), numeric_summary(df), generate_insights(df)
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 500


def test_pdf_report_respects_empty_section_selection():
    df = pd.DataFrame({"sales": [10, 20, 30]})
    pdf = build_pdf_report(
        "sample.csv", df, profile_dataset(df), numeric_summary(df),
        sections=[], report_title="A & B: Analysis",
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 500


def test_pdf_report_includes_optional_analysis_outputs():
    df = pd.DataFrame({"sales": [10, 20, 30], "orders": [1, 2, 3]})
    cleaning = {
        "per_column": pd.DataFrame({
            "column": ["sales"], "missing_before": [1], "missing_after": [0],
        })
    }
    anomalies = pd.DataFrame({
        "sales": [30], "anomaly_flag": ["Anomaly"], "anomaly_score": [-0.1],
    })
    model = {"metrics": {"Accuracy": 0.92, "F1 weighted": 0.9}}
    pdf = build_pdf_report(
        "sample.csv", df, profile_dataset(df), numeric_summary(df),
        sections=["Cleaning audit", "Anomaly results", "Predictive model results"],
        cleaning_report=cleaning, anomalies=anomalies, model_result=model,
        executive_summary="The results are preliminary.",
        methodology="A baseline model was used.",
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 500
