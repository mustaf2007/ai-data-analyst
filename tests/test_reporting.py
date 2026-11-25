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
