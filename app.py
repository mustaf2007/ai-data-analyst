from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from analytics import (
    DataInputError, load_dataset, profile_dataset, column_profile, clean_dataset,
    numeric_summary, categorical_summary, correlation_matrix, grouped_summary,
    infer_datetime_columns, generate_insights, safe_csv_bytes,
)
from modeling import train_predictive_model, detect_anomalies
from reporting import build_pdf_report

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Universal Data Intelligence · Local-first analytics · Open-source Python · No paid APIs")


def initialise_state():
    defaults = {
        "raw": None,
        "active": None,
        "source": "",
        "upload_signature": None,
        "model_result": None,
        "anomalies": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_derived_results():
    st.session_state.model_result = None
    st.session_state.anomalies = None


def set_dataset(data, source, upload_signature=None):
    """Replace the working dataset atomically and invalidate dependent results."""
    st.session_state.raw = data.copy(deep=True)
    st.session_state.active = data.copy(deep=True)
    st.session_state.source = source
    if upload_signature is not None:
        st.session_state.upload_signature = upload_signature
    clear_derived_results()


initialise_state()

with st.sidebar:
    st.header("Workspace")
    upload = st.file_uploader("Upload dataset", type=["csv", "xlsx"])
    if upload is not None:
        content = upload.getvalue()
        signature = hashlib.sha256(upload.name.encode("utf-8") + b"\0" + content).hexdigest()
        # Streamlit reruns the script after most interactions. Only load a given
        # upload once; otherwise cleaning and navigation can be silently undone.
        if signature != st.session_state.upload_signature:
            try:
                uploaded_df = load_dataset(upload.name, content)
                set_dataset(uploaded_df, upload.name, signature)
                st.success(f"Loaded {upload.name}")
            except DataInputError as exc:
                # Remember the rejected payload so the same error is not repeated
                # on every unrelated rerun. The previous working dataset remains intact.
                st.session_state.upload_signature = signature
                st.error(str(exc))

    if st.button("Load built-in sample", use_container_width=True):
        sample_path = ROOT / "sample_data" / "sample_business_data.csv"
        try:
            sample_df = load_dataset(sample_path.name, sample_path.read_bytes())
            set_dataset(sample_df, sample_path.name)
        except (OSError, DataInputError) as exc:
            st.error(f"Could not load built-in sample: {exc}")

    if st.session_state.raw is not None and st.button("Reset to original"):
        st.session_state.active = st.session_state.raw.copy(deep=True)
        clear_derived_results()

    st.caption("Files are processed in this app session. Avoid uploading sensitive data to a public demo.")

if st.session_state.active is None:
    st.info("Upload a CSV/XLSX dataset or load the built-in sample to begin.")
    st.markdown("**Explore · Clean · Visualize · Model · Report**")
    st.stop()

df = st.session_state.active
p = profile_dataset(df)
tabs = st.tabs([
    "Overview", "Data quality", "Clean data", "Explore", "Relationships",
    "Predictive modeling", "Anomalies", "Export & report",
])

with tabs[0]:
    st.subheader(st.session_state.source)
    metric_columns = st.columns(5)
    labels = ["Rows", "Columns", "Missing cells", "Duplicate rows", "Quality score"]
    values = [
        f"{p['rows']:,}", p["columns"], f"{p['missing_cells']:,}",
        f"{p['duplicate_rows']:,}", f"{p['quality_score']:.1f}/100",
    ]
    for column, label, value in zip(metric_columns, labels, values):
        column.metric(label, value)
    st.markdown("#### Automated observations")
    for insight in generate_insights(df):
        st.write("• " + insight)
    st.dataframe(df.head(100), use_container_width=True)

with tabs[1]:
    st.caption("The quality score is a heuristic, not a certification or full domain-specific audit.")
    st.dataframe(column_profile(df), use_container_width=True)
    st.write("Missing values by column")
    if p["missing_by_column"].empty:
        st.success("No missing values.")
    else:
        st.bar_chart(p["missing_by_column"])

with tabs[2]:
    with st.form("clean"):
        dedupe = st.checkbox("Remove duplicate rows", True)
        strategy = st.selectbox(
            "Missing-value strategy",
            [
                "Keep as-is", "Drop rows with missing values",
                "Fill numeric with median", "Fill numeric with mean",
                "Fill numeric with zero", "Fill categorical with mode",
            ],
        )
        selected = st.multiselect("Apply to columns (empty = all eligible)", df.columns.tolist())
        submitted = st.form_submit_button("Apply cleaning")
    if submitted:
        try:
            cleaned = clean_dataset(
                df, drop_duplicates=dedupe, missing_strategy=strategy,
                selected_columns=selected or None,
            )
            st.session_state.active = cleaned
            clear_derived_results()
            st.success(f"Cleaning applied. {len(df) - len(cleaned):,} rows removed.")
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(f"Cleaning could not be applied: {exc}")

with tabs[3]:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    st.markdown("#### Numeric summary")
    st.dataframe(numeric_summary(df), use_container_width=True)

    if categorical:
        category_column = st.selectbox("Categorical variable", categorical)
        frequency = categorical_summary(df)[category_column]
        st.dataframe(frequency, use_container_width=True)
        st.plotly_chart(px.bar(frequency, x="value", y="count"), use_container_width=True)

    if numeric:
        distribution_column = st.selectbox("Distribution variable", numeric)
        st.plotly_chart(
            px.histogram(df, x=distribution_column, marginal="box"),
            use_container_width=True,
        )

    if categorical and numeric:
        group_col, measure_col = st.columns(2)
        group = group_col.selectbox("Group by", categorical)
        measure = measure_col.selectbox("Measure", numeric)
        aggregation = st.selectbox("Aggregation", ["mean", "median", "sum", "count", "min", "max"])
        grouped = grouped_summary(df, group, measure, aggregation)
        st.dataframe(grouped, use_container_width=True)
        st.plotly_chart(px.bar(grouped.head(30), x=group, y=measure), use_container_width=True)

    dates = infer_datetime_columns(df)
    if dates and numeric:
        date_col = st.selectbox("Date-like column", dates)
        trend_measure = st.selectbox("Trend measure", numeric)
        trend_aggregation = st.selectbox(
            "Trend aggregation", ["sum", "mean", "median", "count", "min", "max"]
        )
        trend = df[[date_col, trend_measure]].copy()
        trend[date_col] = pd.to_datetime(trend[date_col], errors="coerce")
        trend = trend.dropna(subset=[date_col]).set_index(date_col)[trend_measure]
        trend = trend.resample("MS").agg(trend_aggregation).reset_index()
        st.plotly_chart(px.line(trend, x=date_col, y=trend_measure, markers=True), use_container_width=True)

with tabs[4]:
    method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
    corr = correlation_matrix(df, method)
    if corr.empty:
        st.info("At least two numeric columns are required.")
    else:
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r"),
            use_container_width=True,
        )

with tabs[5]:
    target = st.selectbox("Target variable", df.columns.tolist())
    task = st.selectbox("Task", ["Auto-detect", "Regression", "Classification"])
    test_share = st.slider("Test set share", .1, .4, .2, .05)
    if st.button("Train baseline model"):
        try:
            st.session_state.model_result = train_predictive_model(df, target, task, test_share)
        except Exception as exc:
            st.session_state.model_result = None
            st.error(f"Model training failed: {exc}")

    result = st.session_state.model_result
    if result:
        st.write(result["task"], "· Random Forest baseline")
        metric_columns = st.columns(len(result["metrics"]))
        for column, (name, value) in zip(metric_columns, result["metrics"].items()):
            column.metric(name, f"{value:.4f}")
        st.dataframe(result["comparison"], use_container_width=True)
        if result.get("dropped_high_cardinality"):
            st.caption("Excluded high-cardinality features: " + ", ".join(result["dropped_high_cardinality"]))
        st.warning(
            "Baseline only. Random splits may be inappropriate for time-series, grouped "
            "or leakage-prone data. Metrics are not a guarantee of future performance."
        )

with tabs[6]:
    contamination = st.slider("Expected anomaly share", .01, .20, .05, .01)
    if st.button("Detect anomalies"):
        try:
            st.session_state.anomalies = detect_anomalies(df, contamination)
        except Exception as exc:
            st.session_state.anomalies = None
            st.error(f"Anomaly detection failed: {exc}")

    anomaly_output = st.session_state.anomalies
    if anomaly_output is not None:
        anomaly_count = int((anomaly_output.anomaly_flag == "Anomaly").sum())
        st.metric("Potential anomalies", anomaly_count)
        st.dataframe(
            anomaly_output.sort_values("anomaly_score").head(200),
            use_container_width=True,
        )
        st.download_button(
            "Download anomaly CSV", safe_csv_bytes(anomaly_output),
            "anomaly_results.csv", "text/csv",
        )
    st.caption("Anomaly flags are screening signals, not proof of errors or fraud.")

with tabs[7]:
    st.download_button(
        "Download active dataset (CSV)", safe_csv_bytes(df),
        "analysis_dataset.csv", "text/csv",
    )
    try:
        pdf = build_pdf_report(
            st.session_state.source, df, p, numeric_summary(df), generate_insights(df)
        )
        st.download_button(
            "Download PDF summary", pdf, "data_analysis_report.pdf", "application/pdf"
        )
    except Exception as exc:
        st.error(f"PDF report could not be generated: {exc}")
    st.caption("PDF includes dataset metrics, observations and numeric summary; it does not embed charts.")

st.divider()
st.caption("AI Data Analyst v1.0 · Local session processing · Correlation is not causation.")
