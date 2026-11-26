from pathlib import Path
import hashlib
import io
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from analytics import (
    DataInputError, load_dataset, profile_dataset, column_profile, clean_dataset, build_time_series_analysis, forecast_time_series_baseline, compare_datasets, build_geographic_summary, run_statistical_test,
    numeric_summary, categorical_summary, correlation_matrix, grouped_summary,
    infer_datetime_columns, generate_insights, generate_detailed_insights, build_data_story, safe_csv_bytes, build_cleaning_report, validate_analysis_readiness, column_quality_diagnostics,
)
from modeling import train_predictive_model, detect_anomalies
from reporting import build_pdf_report, build_text_report_pdf
from assistant import suggest_analyses, answer_dataset_question
from report_assistant import build_narrative_report
from smart_assistant import ask_smart_assistant, build_dataset_context, default_assistant_provider

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Universal Data Intelligence · Local-first analytics · Optional AI providers")


def excel_bytes(dataframe):
    """Return an Excel workbook without writing user data to disk."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Analysis data")
    return output.getvalue()


def excel_analysis_bytes(dataframe, profile, numeric, cleaning_report=None,
                         anomalies=None, model_result=None):
    """Build a multi-sheet workbook containing the active data and analysis outputs."""
    output = io.BytesIO()
    overview_keys = ["rows", "columns", "missing_cells", "missing_pct",
                     "duplicate_rows", "quality_score"]
    overview = pd.DataFrame(
        [{"metric": key.replace("_", " ").title(), "value": profile.get(key)}
         for key in overview_keys]
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Analysis data")
        overview.to_excel(writer, index=False, sheet_name="Dataset overview")
        numeric.to_excel(writer, sheet_name="Numeric summary")
        corr = correlation_matrix(dataframe, "pearson")
        if not corr.empty:
            corr.to_excel(writer, sheet_name="Correlation")
        if cleaning_report is not None:
            cleaning_report["per_column"].to_excel(
                writer, index=False, sheet_name="Cleaning audit"
            )
        if anomalies is not None:
            anomalies.to_excel(writer, index=False, sheet_name="Anomalies")
        if model_result:
            pd.DataFrame(
                [{"metric": key, "value": value}
                 for key, value in model_result.get("metrics", {}).items()]
            ).to_excel(writer, index=False, sheet_name="Model metrics")
    return output.getvalue()


def render_chart_downloads(fig, key):
    """Offer an interactive HTML export and on-demand static chart exports."""
    signature = hashlib.sha256(fig.to_json().encode("utf-8")).hexdigest()
    exports = st.session_state.chart_exports.get(key)
    if exports and exports.get("signature") != signature:
        st.session_state.chart_exports.pop(key, None)
        exports = None
    st.download_button(
        "Download interactive chart (HTML)",
        data=fig.to_html(full_html=True, include_plotlyjs=True),
        file_name=f"{key}.html",
        mime="text/html",
        key=f"{key}_html_download",
    )
    if st.button("Prepare PNG and SVG downloads", key=f"{key}_prepare_static"):
        try:
            st.session_state.chart_exports[key] = {
                "signature": signature,
                "png": fig.to_image(format="png", scale=2),
                "svg": fig.to_image(format="svg"),
            }
            st.success("Static chart files are ready.")
            exports = st.session_state.chart_exports[key]
        except Exception as exc:
            st.warning(
                "Static export is unavailable in this environment. "
                "The interactive HTML download is still available. "
                f"Export detail: {exc}"
            )
    if exports and exports.get("signature") == signature:
        png_col, svg_col = st.columns(2)
        png_col.download_button(
            "Download PNG (2× resolution)", data=exports["png"],
            file_name=f"{key}.png", mime="image/png",
            key=f"{key}_png_download",
        )
        svg_col.download_button(
            "Download SVG (vector)", data=exports["svg"],
            file_name=f"{key}.svg", mime="image/svg+xml",
            key=f"{key}_svg_download",
        )


def style_chart(fig, template):
    fig.update_layout(
        template=template, margin=dict(l=20, r=20, t=55, b=25),
        title_x=0.02, hovermode="x unified", legend_title_text="",
    )
    return fig


def initialise_state():
    defaults = {
        "raw": None,
        "active": None,
        "source": "",
        "upload_signature": None,
        "model_result": None,
        "anomalies": None,
        "model_config": None,
        "anomaly_config": None,
        "cleaning_report": None,
        "chart_exports": {},
        "report_artifacts": None,
        "narrative_report_artifacts": None,
        "chat_history": [],
        "chat_context": None,
        "smart_chat_history": [],
        "smart_chat_context": None,
        "smart_report_artifacts": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_derived_results():
    st.session_state.model_result = None
    st.session_state.anomalies = None
    st.session_state.model_config = None
    st.session_state.anomaly_config = None


def set_dataset(data, source, upload_signature=None):
    """Replace the working dataset atomically and invalidate dependent results."""
    st.session_state.raw = data.copy(deep=True)
    st.session_state.active = data.copy(deep=True)
    st.session_state.source = source
    st.session_state.cleaning_report = None
    st.session_state.chart_exports = {}
    st.session_state.report_artifacts = None
    st.session_state.narrative_report_artifacts = None
    st.session_state.chat_history = []
    st.session_state.chat_context = None
    st.session_state.smart_chat_history = []
    st.session_state.smart_chat_context = None
    st.session_state.smart_report_artifacts = None
    if upload_signature is not None:
        st.session_state.upload_signature = upload_signature
    clear_derived_results()


initialise_state()

with st.sidebar:
    st.header("Workspace")
    selected_theme = st.selectbox(
        "Appearance", ["Light", "Dark", "Green", "Ocean", "Purple", "Sand"],
        index=0, help="Choose a coordinated workspace and chart appearance.",
    )
    upload = st.file_uploader("Upload dataset", type=["csv", "xlsx"])
    reload_upload = st.button("Load selected file", use_container_width=True) if upload is not None else False
    if upload is not None:
        content = upload.getvalue()
        signature = hashlib.sha256(upload.name.encode("utf-8") + b"\0" + content).hexdigest()
        # Streamlit reruns the script after most interactions. Only load a given
        # upload once unless the user explicitly asks to load it again.
        if signature != st.session_state.upload_signature or reload_upload:
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
        st.session_state.cleaning_report = None

    st.caption("Files are processed in this app session. Avoid uploading sensitive data to a public demo.")

theme_css = {
    "Light": ("#f4f7fb", "#ffffff", "#172b4d", "#64748b", "#2563eb"),
    "Dark": ("#101827", "#182437", "#e5edf8", "#a8b6c9", "#60a5fa"),
    "Green": ("#eef6f0", "#ffffff", "#173b2b", "#557363", "#16845b"),
    "Ocean": ("#edf5fb", "#ffffff", "#15324b", "#58758d", "#087eae"),
    "Purple": ("#f4f0fb", "#ffffff", "#30234d", "#76678f", "#7652c8"),
    "Sand": ("#f8f4ec", "#fffdf8", "#43372b", "#847766", "#a66a32"),
}
page_bg, card_bg, text_color, muted_color, accent_color = theme_css[selected_theme]
st.markdown(
    f"""
    <style>
    :root {{ --app-bg: {page_bg}; --card-bg: {card_bg}; --app-text: {text_color};
             --muted-text: {muted_color}; --accent: {accent_color}; }}
    .stApp {{ background: var(--app-bg); color: var(--app-text); }}
    [data-testid="stSidebar"] {{ background: var(--card-bg); }}
    [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--card-bg); border: 1px solid color-mix(in srgb, var(--muted-text) 20%, transparent);
        border-radius: 14px;
    }}
    [data-testid="stMetric"] {{ padding: 14px 16px; }}
    [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {{ color: var(--muted-text); }}
    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px; border-color: var(--accent); transition: all .15s ease;
    }}
    .stButton > button[kind="primary"] {{ background: var(--accent); border-color: var(--accent); }}
    [data-baseweb="tab-list"] {{ gap: .35rem; }}
    [data-baseweb="tab"] {{ border-radius: 9px 9px 0 0; }}
    [data-testid="stChatMessage"] {{ background: var(--card-bg); border-radius: 14px; }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius: 12px; }}
    .block-container {{ padding-top: 1.5rem; }}
    h1, h2, h3 {{ letter-spacing: -0.025em; }}
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.active is None:
    st.info("Upload a CSV/XLSX dataset or load the built-in sample to begin.")
    st.markdown("**Explore · Clean · Visualize · Model · Report**")
    st.markdown("Choose a workspace appearance from the sidebar. The theme applies before a dataset is loaded.")
    st.stop()

df = st.session_state.active
p = profile_dataset(df)
theme_templates = {
    "Light": "plotly_white", "Dark": "plotly_dark", "Green": "plotly_white",
    "Ocean": "plotly_white", "Purple": "plotly_white", "Sand": "plotly_white",
}
theme_template = theme_templates[selected_theme]
px.defaults.template = theme_template
px.defaults.color_discrete_sequence = [accent_color, "#20a486", "#e09f3e", "#8b78c6", "#4c9bd6"]
tabs = st.tabs([
    "Dashboard", "Data quality", "Clean data", "Explore", "Relationships",
    "Predictive modeling", "Anomalies", "Export & report", "Chart Studio", "Automated insights", "Data story", "Time-series intelligence", "Dataset comparison", "Geographic intelligence", "Statistical testing",
    "Analysis suggestions", "Ask the data", "AI narrative report", "Smart AI assistant",
])

with tabs[0]:
    st.subheader(st.session_state.source)
    st.caption("Executive overview · Dataset health · Trends · Key patterns")
    metric_columns = st.columns(5)
    labels = ["Records", "Variables", "Missing cells", "Duplicate rows", "Quality score"]
    values = [
        f"{p['rows']:,}", p["columns"], f"{p['missing_cells']:,}",
        f"{p['duplicate_rows']:,}", f"{p['quality_score']:.1f}/100",
    ]
    for column, label, value in zip(metric_columns, labels, values):
        column.metric(label, value)

    insights = generate_insights(df)
    with st.container(border=True):
        st.markdown("#### Key findings")
        if insights:
            insight_cols = st.columns(min(2, len(insights)))
            for index, insight in enumerate(insights):
                insight_cols[index % len(insight_cols)].write("• " + insight)
        else:
            st.info("No automated observations are available for this dataset.")

    numeric_dashboard = df.select_dtypes(include=np.number).columns.tolist()
    time_candidates = infer_datetime_columns(df)
    year_candidates = [
        col for col in numeric_dashboard
        if str(col).strip().lower() in {"year", "yr", "yyyy"}
    ]
    time_options = time_candidates + [c for c in year_candidates if c not in time_candidates]

    if numeric_dashboard:
        st.markdown("#### Dataset at a glance")
        if time_options:
            trend_controls = st.columns([1, 2])
            time_col = trend_controls[0].selectbox(
                "Time variable", time_options, key="dashboard_time_column"
            )
            trend_col = trend_controls[1].selectbox(
                "Indicator", numeric_dashboard, key="dashboard_trend_measure"
            )
            trend_data = df[[time_col, trend_col]].copy()
            if time_col in time_candidates:
                trend_data[time_col] = pd.to_datetime(trend_data[time_col], errors="coerce")
            trend_data = trend_data.dropna(subset=[time_col, trend_col]).sort_values(time_col)
            trend_fig = px.line(
                trend_data, x=time_col, y=trend_col, markers=True,
                title=f"{trend_col} over time",
            )
            style_chart(trend_fig, theme_template)
            st.plotly_chart(trend_fig, use_container_width=True)
            render_chart_downloads(trend_fig, "dashboard_trend")

        left_chart, right_chart = st.columns(2)
        distribution_col = left_chart.selectbox(
            "Distribution variable", numeric_dashboard, key="dashboard_distribution"
        )
        distribution_fig = px.histogram(
            df, x=distribution_col, marginal="box",
            title=f"Distribution of {distribution_col}",
        )
        style_chart(distribution_fig, theme_template)
        left_chart.plotly_chart(distribution_fig, use_container_width=True)
        with left_chart:
            render_chart_downloads(distribution_fig, "dashboard_distribution")

        if len(numeric_dashboard) >= 2:
            corr = df[numeric_dashboard].corr(numeric_only=True)
            corr_fig = px.imshow(
                corr, text_auto=".2f", zmin=-1, zmax=1,
                color_continuous_scale="RdBu_r", title="Numeric correlation overview",
            )
            style_chart(corr_fig, theme_template)
            right_chart.plotly_chart(corr_fig, use_container_width=True)
            with right_chart:
                render_chart_downloads(corr_fig, "dashboard_correlation")
        else:
            right_chart.info("Add another numeric variable to display a correlation overview.")

    with st.expander("Preview active dataset"):
        st.dataframe(df.head(100), use_container_width=True)

with tabs[1]:
    st.caption("The overview quality score is a heuristic, not a certification or full domain-specific audit.")
    analysis_type = st.selectbox(
        "What do you plan to do with this dataset?",
        ["General exploratory analysis", "Correlation analysis", "Regression",
         "Classification", "Time-series analysis"],
    )
    target_column = None
    if analysis_type in {"Regression", "Classification"}:
        target_column = st.selectbox("Select target variable", df.columns.tolist())
    readiness = validate_analysis_readiness(df, analysis_type, target_column)
    st.markdown("#### Analysis-readiness validation")
    readiness_cols = st.columns(4)
    readiness_cols[0].metric("Readiness status", readiness["status"])
    readiness_cols[1].metric("Checklist score", f'{readiness["score"]:.1f}%')
    readiness_cols[2].metric("Checks passed", f'{readiness["passed"]}/{readiness["total"]}')
    readiness_cols[3].metric("Issues to review", readiness["blockers"] + readiness["warnings"])
    if readiness["status"] == "Ready":
        st.success(f"The dataset passed the selected {analysis_type.lower()} checklist.")
    elif readiness["status"] == "Ready with warnings":
        st.warning("The selected analysis can proceed cautiously, but review the warnings below.")
    else:
        st.error("The dataset has blocking issues for the selected analysis. Address them before proceeding.")
    st.caption("The percentage is the share of checks passed for the selected workflow. It is not a universal certification; suitability still depends on research design, sampling, measurement, assumptions and domain rules.")
    st.dataframe(readiness["checks"][["check", "status", "detail"]], use_container_width=True, hide_index=True)

    st.markdown("#### Column-level diagnostics")
    diagnostics = column_quality_diagnostics(df)
    st.dataframe(diagnostics, use_container_width=True, hide_index=True)
    st.caption("Missingness bands: none = 0%; low = up to 5%; moderate = over 5% to 20%; high = over 20%. Investigate the reason and pattern of missingness before choosing an imputation method.")
    st.dataframe(column_profile(df), use_container_width=True)
    st.write("Missing values by column")
    if p["missing_by_column"].empty:
        st.success("No missing values.")
    else:
        st.bar_chart(p["missing_by_column"])

with tabs[2]:
    with st.form("clean"):
        dedupe = st.checkbox("Remove duplicate rows", True)
        drop_empty = st.checkbox("Remove entirely empty columns", True)
        strategy = st.selectbox(
            "Missing-value strategy",
            [
                "Fill numeric with median and categorical with mode",
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
                df, drop_duplicates=dedupe, drop_empty_columns=drop_empty,
                missing_strategy=strategy, selected_columns=selected or None,
            )
            if cleaned.empty:
                raise ValueError("These cleaning settings would remove every row. Your dataset was not changed.")
            report = build_cleaning_report(
                df, cleaned, drop_duplicates=dedupe,
                missing_strategy=strategy, selected_columns=selected or None,
            )
            st.session_state.active = cleaned
            st.session_state.cleaning_report = report
            clear_derived_results()
            st.rerun()
        except (ValueError, TypeError) as exc:
            st.error(f"Cleaning could not be applied: {exc}")

    cleaning_report = st.session_state.cleaning_report
    st.markdown("#### Analysis dataset download")
    if cleaning_report is not None:
        st.success("Cleaning report generated. The updated working dataset is active.")
        st.download_button(
            "Download cleaned dataset (CSV)",
            safe_csv_bytes(df),
            "cleaned_dataset.csv",
            "text/csv",
            key="download_cleaned_dataset",
        )
        if cleaning_report["rows_removed"] == 0 and cleaning_report["missing_filled"] == 0 and not cleaning_report["per_column"]["column_removed"].any():
            st.info("The cleaning checks found no rows, missing values, or empty columns that needed changes.")
        else:
            st.success("Cleaning changes have been applied. Review the before-and-after audit below.")
        st.caption(
            f"Applied strategy: {cleaning_report['strategy']} · "
            f"Selected columns: {', '.join(cleaning_report['selected_columns']) if cleaning_report['selected_columns'] else 'All eligible columns'}"
        )
        report_metrics = st.columns(5)
        report_metrics[0].metric(
            "Rows", f"{cleaning_report['rows_before']:,} → {cleaning_report['rows_after']:,}",
            delta=f"-{cleaning_report['rows_removed']:,} removed" if cleaning_report["rows_removed"] else "No rows removed",
        )
        report_metrics[1].metric("Columns", f"{len(cleaning_report['per_column'])} → {len(df.columns)}")
        report_metrics[2].metric("Duplicates removed", f"{cleaning_report['duplicates_removed']:,}")
        report_metrics[3].metric("Missing cells filled", f"{cleaning_report['missing_filled']:,}")
        report_metrics[4].metric("Missing cells remaining", f"{cleaning_report['missing_after']:,}")
        st.write("**Data quality score (heuristic)**")
        st.metric(
            "Before → after",
            f"{cleaning_report['quality_before']:.1f} → {cleaning_report['quality_after']:.1f} / 100",
            delta=f"{cleaning_report['quality_after'] - cleaning_report['quality_before']:+.1f}",
        )
        st.write("**Column-by-column cleaning audit**")
        st.dataframe(cleaning_report["per_column"], use_container_width=True, hide_index=True)
        st.download_button(
            "Download cleaning report (CSV)",
            cleaning_report["per_column"].to_csv(index=False).encode("utf-8-sig"),
            "cleaning_report.csv", "text/csv",
        )
    else:
        st.info("No cleaning operation has been applied yet. You can download the current dataset or apply the recommended settings above.")
        st.download_button(
            "Download current dataset (CSV)",
            safe_csv_bytes(df),
            "analysis_dataset.csv",
            "text/csv",
            key="download_current_dataset_clean_tab",
        )

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
    model_config = (target, task, float(test_share))
    if st.session_state.model_config is not None and st.session_state.model_config != model_config:
        st.session_state.model_result = None
    st.session_state.model_config = model_config
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
    anomaly_config = float(contamination)
    if st.session_state.anomaly_config is not None and st.session_state.anomaly_config != anomaly_config:
        st.session_state.anomalies = None
    st.session_state.anomaly_config = anomaly_config
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
    st.subheader("Analysis Report Builder")
    st.caption(
        "Create a tailored PDF report and a multi-sheet Excel workbook from the active dataset. "
        "Add your interpretation and choose which analysis outputs to include."
    )
    numeric_report_columns = df.select_dtypes(include=np.number).columns.tolist()
    all_report_columns = df.columns.tolist()
    report_sections = [
        "Dataset overview", "Key findings", "Numeric summary", "Correlation matrix",
        "Cleaning audit", "Anomaly results", "Predictive model results", "Dataset sample",
    ]
    with st.form("analysis_report_builder"):
        report_title = st.text_input(
            "Report title", value="Data Analysis Report",
            help="Use a clear title suitable for sharing with colleagues or decision-makers.",
        )
        prepared_by = st.text_input("Prepared by / organization (optional)")
        executive_summary = st.text_area(
            "Executive summary (optional)",
            placeholder="Summarize the purpose, main results and implications...",
            height=110,
        )
        methodology = st.text_area(
            "Methodology and limitations (optional)",
            placeholder="Describe data sources, definitions, methods, assumptions and limitations...",
            height=110,
        )
        selected_sections = st.multiselect(
            "Include report sections", report_sections,
            default=["Dataset overview", "Key findings", "Numeric summary",
                     "Correlation matrix", "Dataset sample"],
        )
        include_report_chart = st.checkbox(
            "Include one chart in the PDF", value=bool(numeric_report_columns),
            disabled=not bool(numeric_report_columns),
        )
        if all_report_columns:
            chart_controls = st.columns(3)
            report_chart_type = chart_controls[0].selectbox(
                "Report chart type", ["Line", "Bar", "Scatter", "Histogram"]
            )
            report_chart_x = chart_controls[1].selectbox(
                "Chart X variable", all_report_columns
            )
            report_y_options = numeric_report_columns or all_report_columns
            report_chart_y = chart_controls[2].selectbox(
                "Chart Y variable", report_y_options,
                disabled=report_chart_type == "Histogram",
            )
        else:
            report_chart_type, report_chart_x, report_chart_y = "Histogram", None, None
        build_report = st.form_submit_button("Generate PDF and Excel report", type="primary")

    if build_report:
        chart_images = []
        chart_warning = None
        if include_report_chart and report_chart_x is not None:
            try:
                if report_chart_type == "Line":
                    report_fig = px.line(df, x=report_chart_x, y=report_chart_y,
                                         markers=True, title=report_title)
                elif report_chart_type == "Bar":
                    report_fig = px.bar(df, x=report_chart_x, y=report_chart_y,
                                        title=report_title)
                elif report_chart_type == "Scatter":
                    report_fig = px.scatter(df, x=report_chart_x, y=report_chart_y,
                                            title=report_title)
                else:
                    report_fig = px.histogram(df, x=report_chart_x, marginal="box",
                                              title=report_title)
                style_chart(report_fig, theme_template)
                chart_images.append((report_title, report_fig.to_image(format="png", scale=1.5)))
            except Exception as exc:
                chart_warning = (
                    "The PDF will be generated without the chart because static chart "
                    f"rendering was unavailable: {exc}"
                )

        try:
            report_pdf = build_pdf_report(
                st.session_state.source, df, p, numeric_summary(df),
                generate_insights(df),
                report_title=report_title,
                prepared_by=prepared_by,
                executive_summary=executive_summary,
                methodology=methodology,
                sections=selected_sections,
                cleaning_report=st.session_state.cleaning_report,
                anomalies=st.session_state.anomalies,
                model_result=st.session_state.model_result,
                chart_images=chart_images,
            )
            report_xlsx = excel_analysis_bytes(
                df, p, numeric_summary(df),
                cleaning_report=st.session_state.cleaning_report,
                anomalies=st.session_state.anomalies,
                model_result=st.session_state.model_result,
            )
            st.session_state.report_artifacts = {
                "pdf": report_pdf, "xlsx": report_xlsx,
                "title": report_title or "analysis_report",
            }
            st.success("Your report files are ready.")
            if chart_warning:
                st.warning(chart_warning)
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

    artifacts = st.session_state.report_artifacts
    if artifacts:
        safe_name = "".join(
            ch.lower() if ch.isalnum() else "_" for ch in artifacts["title"]
        ).strip("_") or "analysis_report"
        pdf_col, excel_col = st.columns(2)
        pdf_col.download_button(
            "Download professional PDF report",
            data=artifacts["pdf"], file_name=f"{safe_name}.pdf",
            mime="application/pdf", key="download_built_pdf_report",
            use_container_width=True,
        )
        excel_col.download_button(
            "Download analysis workbook (Excel)",
            data=artifacts["xlsx"], file_name=f"{safe_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_built_excel_report", use_container_width=True,
        )
        st.caption(
            "The Excel workbook contains the active dataset, dataset overview, numeric "
            "summary and available correlation, cleaning, anomaly and model outputs."
        )

    st.divider()
    st.markdown("#### Quick dataset exports")
    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "Download active dataset (CSV)", safe_csv_bytes(df),
        "analysis_dataset.csv", "text/csv",
    )
    download_col2.download_button(
        "Download active dataset (Excel)", data=excel_bytes(df),
        file_name="analysis_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with tabs[8]:
    st.subheader("Chart Studio")
    st.caption("Build a chart from the active dataset, customize its appearance, and export it.")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    all_columns = df.columns.tolist()
    if not all_columns:
        st.info("The active dataset has no columns to visualize.")
    else:
        controls = st.columns(3)
        chart_kind = controls[0].selectbox(
            "Chart type", ["Line", "Bar", "Scatter", "Area", "Histogram", "Box", "Violin"]
        )
        x_column = controls[1].selectbox("X variable", all_columns)
        y_options = numeric_columns or all_columns
        y_column = controls[2].selectbox(
            "Y variable", y_options, disabled=chart_kind == "Histogram"
        )
        color_options = ["None"] + all_columns
        color_column = st.selectbox("Color / group by (optional)", color_options)
        chart_title = st.text_input("Chart title", value=f"{chart_kind}: {y_column} by {x_column}")
        chart_height = st.slider("Chart height", 350, 850, 520, 50)

        color_arg = None if color_column == "None" else color_column
        try:
            if chart_kind == "Line":
                chart_fig = px.line(df, x=x_column, y=y_column, color=color_arg, markers=True, title=chart_title)
            elif chart_kind == "Bar":
                chart_fig = px.bar(df, x=x_column, y=y_column, color=color_arg, title=chart_title)
            elif chart_kind == "Scatter":
                has_numeric_xy = pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column])
                chart_fig = px.scatter(
                    df, x=x_column, y=y_column, color=color_arg, title=chart_title,
                    trendline="ols" if len(df) >= 3 and has_numeric_xy else None,
                )
            elif chart_kind == "Area":
                chart_fig = px.area(df, x=x_column, y=y_column, color=color_arg, title=chart_title)
            elif chart_kind == "Histogram":
                chart_fig = px.histogram(df, x=x_column, color=color_arg, marginal="box", title=chart_title)
            elif chart_kind == "Box":
                chart_fig = px.box(df, x=x_column, y=y_column, color=color_arg, title=chart_title)
            else:
                chart_fig = px.violin(df, x=x_column, y=y_column, color=color_arg, box=True, points="outliers", title=chart_title)
            style_chart(chart_fig, theme_template)
            chart_fig.update_layout(height=chart_height)
            st.plotly_chart(chart_fig, use_container_width=True)
            render_chart_downloads(chart_fig, "chart_studio_output")
        except Exception as exc:
            st.error(f"Could not create this chart with the selected variables: {exc}")

st.divider()
st.caption("AI Data Analyst v1.1 · Local session processing · Correlation is not causation.")


with tabs[9]:
    st.caption("Deterministic, data-backed observations generated locally from the active dataset. No external AI service is used.")
    detailed_findings = generate_detailed_insights(df)
    if not detailed_findings:
        st.info("No findings can be generated from an empty dataset.")
    else:
        st.markdown(f"#### {len(detailed_findings)} observations")
        findings_frame = pd.DataFrame(detailed_findings)
        for index, item in findings_frame.iterrows():
            with st.container(border=True):
                st.markdown(f"**{item['area']}**")
                st.write(item["finding"])
                st.caption("Evidence: " + item["evidence"])
                st.caption("Interpretation note: " + item["caveat"])
        st.download_button(
            "Download insights (CSV)",
            data=safe_csv_bytes(findings_frame),
            file_name="automated_data_insights.csv",
            mime="text/csv",
            key="download_detailed_insights",
        )
        st.download_button(
            "Download insights (Excel)",
            data=excel_bytes(findings_frame),
            file_name="automated_data_insights.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_detailed_insights_excel",
        )


with tabs[10]:
    st.subheader("Data Story Generator")
    st.caption("Create a structured, evidence-linked narrative from selected findings. This is template-based reporting, not generative AI.")
    story_title = st.text_input("Brief title", "Data Analysis Brief", key="story_title")
    audience = st.selectbox(
        "Intended audience",
        ["General", "Policy makers", "Researchers", "Executive leadership", "Development partners"],
        key="story_audience",
    )
    context = st.text_area(
        "Background or research question (optional)",
        placeholder="Describe the purpose of this analysis, its geographic or sector focus, and the question it addresses.",
        key="story_context",
    )
    story_findings = generate_detailed_insights(df)
    if story_findings:
        options = {
            f"{item['area']} — {item['finding']}": item
            for item in story_findings
        }
        selected_labels = st.multiselect(
            "Select findings to include",
            list(options.keys()),
            default=list(options.keys()),
            key="story_selected_findings",
        )
        selected_findings = [options[label] for label in selected_labels]
        story = build_data_story(
            selected_findings, title=story_title, audience=audience, context=context
        )
        st.text_area("Generated analytical narrative", value=story, height=520)
        st.download_button(
            "Download data story (TXT)",
            data=story.encode("utf-8"),
            file_name="data_analysis_brief.txt",
            mime="text/plain",
            key="download_data_story",
        )
        st.download_button(
            "Download selected evidence (CSV)",
            data=safe_csv_bytes(pd.DataFrame(selected_findings)),
            file_name="data_story_evidence.csv",
            mime="text/csv",
            key="download_data_story_evidence",
        )
    else:
        st.info("Upload or load a dataset with usable observations to generate a data story.")



with tabs[11]:
    st.subheader("Time-Series Intelligence")
    st.caption("Aggregate observations over time, compare periods, inspect moving averages and identify unusual changes.")
    inferred_dates = infer_datetime_columns(df)
    date_candidates = list(dict.fromkeys(
        inferred_dates + [
            c for c in df.columns
            if str(c).strip().lower() in {"year", "yr", "yyyy", "month", "date", "period"}
        ]
    ))
    numeric_candidates = df.select_dtypes(include=np.number).columns.tolist()
    if not date_candidates:
        st.info("No date/time column was identified. Add a recognizable date column (or a numeric Year column) to use this workspace.")
    elif not numeric_candidates:
        st.info("No numeric measure is available. Convert the indicator column to numeric values before analysis.")
    else:
        controls = st.columns(4)
        date_col = controls[0].selectbox("Date / period", date_candidates, key="ts_date_column")
        value_col = controls[1].selectbox("Numeric measure", numeric_candidates, key="ts_value_column")
        frequency_label = controls[2].selectbox("Period frequency", ["Month", "Quarter", "Year"], key="ts_frequency")
        aggregation = controls[3].selectbox("Aggregation", ["sum", "mean", "median", "count"], key="ts_aggregation")
        window = st.slider("Moving-average window (periods)", 2, 12, 3, key="ts_ma_window")
        freq_code = {"Month": "MS", "Quarter": "QS", "Year": "YS"}[frequency_label]
        try:
            ts_result = build_time_series_analysis(
                df, date_col, value_col, frequency=freq_code,
                aggregation=aggregation, moving_average_window=window,
            )
            ts_data = ts_result["data"]
            summary = ts_result["summary"]
            metrics = st.columns(4)
            metrics[0].metric("Periods", f"{summary['periods']:,}")
            metrics[1].metric("Valid observations", f"{summary['observations']:,}")
            metrics[2].metric("Trend direction", summary["trend_direction"])
            metrics[3].metric("Unusual changes", summary["unusual_periods"])
            st.caption(
                f"Aggregation: {aggregation} · Frequency: {frequency_label.lower()} · "
                f"Linear slope: {summary['trend_slope_per_period']:.4g} measure units per period"
            )

            trend_fig = px.line(
                ts_data, x="period", y="value", markers=True,
                title=f"{value_col} over time ({frequency_label.lower()}ly {aggregation})",
            )
            trend_fig.add_scatter(
                x=ts_data["period"], y=ts_data["moving_average"],
                mode="lines", name=f"{window}-period moving average",
            )
            trend_fig.add_scatter(
                x=ts_data["period"], y=ts_data["trend_line"],
                mode="lines", name="Linear trend (descriptive)",
                line=dict(dash="dash"),
            )
            unusual = ts_data[ts_data["unusual_change"]]
            if not unusual.empty:
                trend_fig.add_scatter(
                    x=unusual["period"], y=unusual["value"],
                    mode="markers", name="Unusual period change",
                    marker=dict(size=11, symbol="diamond"),
                )
            style_chart(trend_fig, theme_template)
            st.plotly_chart(trend_fig, use_container_width=True)
            render_chart_downloads(trend_fig, "time_series_intelligence")

            st.markdown("#### Period-by-period analysis")
            display_data = ts_data.copy()
            display_data["period"] = display_data["period"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_data, use_container_width=True, hide_index=True)
            st.download_button(
                "Download time-series results (CSV)",
                data=safe_csv_bytes(display_data),
                file_name="time_series_analysis.csv", mime="text/csv",
                key="download_time_series_csv",
            )
            st.download_button(
                "Download time-series results (Excel)",
                data=excel_bytes(display_data),
                file_name="time_series_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_time_series_excel",
            )
            with st.expander("Optional baseline forecast", expanded=False):
                st.caption("A simple last-observation baseline with chronological holdout evaluation; not a complex predictive model.")
                forecast_horizon = st.slider("Periods to forecast", 1, 12, 3, key="ts_forecast_horizon")
                if st.button("Evaluate and generate baseline forecast", key="run_ts_baseline"):
                    try:
                        forecast_result = forecast_time_series_baseline(
                            ts_data["value"], horizon=forecast_horizon
                        )
                        forecast_metrics = forecast_result["metrics"]
                        fcols = st.columns(3)
                        fcols[0].metric("Training periods", forecast_metrics["training_periods"])
                        fcols[1].metric("Holdout MAE", f"{forecast_metrics['mae']:.4g}")
                        fcols[2].metric("Holdout sMAPE", f"{forecast_metrics['smape_pct']:.2f}%")
                        st.markdown("**Chronological holdout results**")
                        st.dataframe(forecast_result["validation"], use_container_width=True, hide_index=True)
                        st.markdown("**Future baseline estimates**")
                        st.dataframe(forecast_result["forecast"], use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download baseline forecast (CSV)",
                            data=safe_csv_bytes(forecast_result["forecast"]),
                            file_name="time_series_baseline_forecast.csv",
                            mime="text/csv", key="download_ts_forecast",
                        )
                        for limitation in forecast_result["limitations"]:
                            st.caption(f"• {limitation}")
                    except ValueError as exc:
                        st.warning(str(exc))
            for warning in ts_result["warnings"]:
                st.warning(warning)
            st.caption(
                "Interpretation limits: trends are descriptive, not causal. Unusual-change flags use an "
                "IQR rule and are prompts for investigation, not proof of errors. Check date coverage, "
                "units, missing periods, seasonality and the meaning of the selected aggregation."
            )
        except (ValueError, TypeError) as exc:
            st.warning(f"Time-series analysis could not be completed: {exc}")



with tabs[12]:
    st.subheader("Dataset Comparison & Benchmarking")
    st.caption(
        "Compare the active working dataset with a second CSV/XLSX file. "
        "The comparison file is kept separate and never replaces your active dataset."
    )
    comparison_upload = st.file_uploader(
        "Upload comparison dataset", type=["csv", "xlsx"], key="comparison_file_upload"
    )
    if comparison_upload is None:
        st.info("Upload a second dataset to compare its coverage, schema, numeric indicators and categorical distributions with the active dataset.")
        st.caption("The active dataset is the reference. The uploaded file is the comparison dataset.")
    else:
        comparison_bytes = comparison_upload.getvalue()
        comparison_signature = hashlib.sha256(
            comparison_upload.name.encode("utf-8") + b"\\0" + comparison_bytes
        ).hexdigest()
        if st.session_state.get("comparison_signature") != comparison_signature:
            st.session_state.comparison_signature = comparison_signature
            st.session_state.comparison_result = None
        st.caption(f"Reference: {st.session_state.source} · Comparison: {comparison_upload.name}")
        if st.button("Run dataset comparison", type="primary", key="run_dataset_comparison"):
            try:
                comparison_df = load_dataset(comparison_upload.name, comparison_bytes)
                st.session_state.comparison_result = compare_datasets(df, comparison_df)
            except (DataInputError, ValueError, TypeError) as exc:
                st.session_state.comparison_result = None
                st.error(f"Comparison could not be completed: {exc}")

        comparison_result = st.session_state.get("comparison_result")
        if comparison_result is not None:
            result = comparison_result
            overview = result["summary"]
            overview_cols = st.columns(2)
            for index, row in overview.iterrows():
                overview_cols[index].markdown(f"**{row['dataset']} dataset**")
                overview_cols[index].metric("Rows", f"{row['rows']:,}")
                overview_cols[index].metric("Columns", f"{row['columns']:,}")
                overview_cols[index].metric("Missing cells", f"{row['missing_cells']:,}")
                overview_cols[index].metric("Duplicate rows", f"{row['duplicate_rows']:,}")
                overview_cols[index].caption(f"Missingness: {row['missing_pct']:.2f}%")

            st.markdown("#### Schema comparison")
            st.dataframe(result["schema"], use_container_width=True, hide_index=True)
            if not result["numeric"].empty:
                st.markdown("#### Numeric indicator comparison")
                st.dataframe(result["numeric"], use_container_width=True, hide_index=True)
                st.caption("Mean differences are comparison minus reference. Percentage differences use the absolute reference mean and are unavailable when that mean is zero.")
            else:
                st.info("No shared numeric columns were available for numeric summary comparison.")
            if not result["categorical"].empty:
                st.markdown("#### Categorical distribution comparison")
                st.dataframe(result["categorical"], use_container_width=True, hide_index=True)
                st.caption("Distribution differences are expressed in percentage points, not percent change.")
            else:
                st.info("No shared low-cardinality categorical columns were available for distribution comparison.")

            for warning in result["warnings"]:
                st.warning(warning)
            with st.expander("Interpretation limits", expanded=False):
                for limitation in result["limitations"]:
                    st.write("• " + limitation)

            workbook = io.BytesIO()
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                result["summary"].to_excel(writer, index=False, sheet_name="Overview")
                result["schema"].to_excel(writer, index=False, sheet_name="Schema")
                result["numeric"].to_excel(writer, index=False, sheet_name="Numeric comparison")
                result["categorical"].to_excel(writer, index=False, sheet_name="Category comparison")
                pd.DataFrame({"warning": result["warnings"]}).to_excel(
                    writer, index=False, sheet_name="Warnings"
                )
            export_cols = st.columns(2)
            export_cols[0].download_button(
                "Download comparison workbook (Excel)", workbook.getvalue(),
                "dataset_comparison.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_dataset_comparison_excel",
            )
            export_cols[1].download_button(
                "Download numeric comparison (CSV)",
                safe_csv_bytes(result["numeric"]), "numeric_comparison.csv",
                "text/csv", key="download_dataset_comparison_numeric",
            )
            st.download_button(
                "Download categorical comparison (CSV)",
                safe_csv_bytes(result["categorical"]), "categorical_comparison.csv",
                "text/csv", key="download_dataset_comparison_categorical",
            )



with tabs[13]:
    st.subheader("Geographic & Regional Intelligence")
    st.caption(
        "Aggregate and rank a numeric indicator across the geographic labels present in your dataset. "
        "Choose the geographic level and measure explicitly."
    )
    geo_candidates = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique(dropna=True) > 0
    ]
    geo_numeric = df.select_dtypes(include=np.number).columns.tolist()
    if not geo_candidates:
        st.info("No non-numeric geographic/category field was found. Add a country, region, state, district or other location label.")
    elif not geo_numeric:
        st.info("No numeric indicator is available for geographic aggregation.")
    else:
        geo_controls = st.columns([1.4, 1.4, 1, 1])
        geo_column = geo_controls[0].selectbox(
            "Geographic field", geo_candidates, key="geo_field"
        )
        geo_measure = geo_controls[1].selectbox(
            "Numeric indicator", geo_numeric, key="geo_measure"
        )
        geo_aggregation = geo_controls[2].selectbox(
            "Aggregation", ["sum", "mean", "median", "min", "max"],
            key="geo_aggregation",
        )
        geo_top_n = geo_controls[3].selectbox(
            "Show top", [10, 20, 25, 50, 100], index=2, key="geo_top_n"
        )
        try:
            geo_result = build_geographic_summary(
                df, geo_column, geo_measure,
                aggregation=geo_aggregation, top_n=geo_top_n,
            )
            coverage = geo_result["coverage"]
            geo_kpis = st.columns(4)
            geo_kpis[0].metric("Geographic groups", f"{coverage['geographic_groups']:,}")
            geo_kpis[1].metric("Included records", f"{coverage['included_rows']:,}")
            geo_kpis[2].metric("Excluded records", f"{coverage['excluded_rows']:,}")
            geo_kpis[3].metric("Aggregation", geo_aggregation.title())

            geo_data = geo_result["data"]
            st.markdown("#### Geographic ranking")
            st.dataframe(geo_data, use_container_width=True, hide_index=True)
            chart_title = f"{geo_measure} by {geo_column} ({geo_aggregation})"
            geo_fig = px.bar(
                geo_data.sort_values("value", ascending=True),
                x="value", y="geography", orientation="h",
                title=chart_title, hover_data=["rank", "observations"],
                labels={"value": geo_measure, "geography": geo_column},
            )
            style_chart(geo_fig, theme_template)
            st.plotly_chart(geo_fig, use_container_width=True)
            render_chart_downloads(geo_fig, "geographic_ranking")

            st.markdown("#### Coverage and interpretation")
            st.caption(
                f"Input: {coverage['input_rows']:,} rows · "
                f"Included: {coverage['included_rows']:,} · "
                f"Geographic field: {coverage['geographic_field']} · "
                f"Indicator: {coverage['measure']}"
            )
            for warning in geo_result["warnings"]:
                st.warning(warning)
            with st.expander("Methodology and limitations", expanded=False):
                for limitation in geo_result["limitations"]:
                    st.write("• " + limitation)

            geo_csv = safe_csv_bytes(geo_result["all_data"])
            geo_excel = excel_bytes(geo_result["all_data"])
            geo_downloads = st.columns(2)
            geo_downloads[0].download_button(
                "Download full geographic results (CSV)", geo_csv,
                "geographic_summary.csv", "text/csv",
                key="download_geographic_csv",
            )
            geo_downloads[1].download_button(
                "Download full geographic results (Excel)", geo_excel,
                "geographic_summary.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_geographic_excel",
            )
            st.caption(
                "Map-ready output is provided as a geographic ranking table. "
                "A reliable map requires standardized geographic codes, coordinates or a verified boundary dataset."
            )
        except ValueError as exc:
            st.warning(f"Geographic analysis could not be completed: {exc}")


with tabs[14]:
    st.subheader("Statistical Testing & Hypothesis Analysis")
    st.caption(
        "Run selected classical tests with effect sizes, confidence intervals where applicable, "
        "sample summaries and assumption warnings."
    )
    numeric_test_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_test_columns = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
        and 1 < df[col].nunique(dropna=True) <= 100
    ]
    test_type = st.selectbox(
        "Statistical test",
        [
            "One-sample t-test", "Independent-samples Welch t-test",
            "One-way ANOVA", "Chi-square test of independence",
        ],
        key="stat_test_type",
    )
    if test_type == "Chi-square test of independence":
        if len(categorical_test_columns) < 2:
            st.info("At least two categorical columns with two or more categories are required.")
        else:
            stat_controls = st.columns(2)
            stat_outcome = stat_controls[0].selectbox(
                "First categorical variable", categorical_test_columns,
                key="stat_chi_first",
            )
            second_options = [c for c in categorical_test_columns if c != stat_outcome]
            stat_group = stat_controls[1].selectbox(
                "Second categorical variable", second_options,
                key="stat_chi_second",
            )
            stat_alpha = st.selectbox(
                "Significance level (alpha)", [0.01, 0.05, 0.10],
                index=1, key="stat_chi_alpha",
            )
            run_stat_button = st.button(
                "Run chi-square test", type="primary", key="run_stat_chi"
            )
    elif not numeric_test_columns:
        st.info("No numeric outcome columns are available for these tests.")
    else:
        stat_controls = st.columns(3)
        stat_outcome = stat_controls[0].selectbox(
            "Numeric outcome", numeric_test_columns, key="stat_numeric_outcome"
        )
        stat_group = None
        if test_type in {"Independent-samples Welch t-test", "One-way ANOVA"}:
            if not categorical_test_columns:
                st.info("Add a categorical grouping column with at least two groups.")
                stat_group = None
            else:
                stat_group = stat_controls[1].selectbox(
                    "Grouping variable", categorical_test_columns,
                    key="stat_group_column",
                )
        if test_type == "One-sample t-test":
            hypothesized_mean = stat_controls[1].number_input(
                "Hypothesized population mean", value=0.0,
                key="stat_hypothesized_mean",
            )
        else:
            hypothesized_mean = 0.0
        stat_alpha = stat_controls[2 if test_type != "One-sample t-test" else 2].selectbox(
            "Significance level (alpha)", [0.01, 0.05, 0.10],
            index=1, key="stat_numeric_alpha",
        )
        run_stat_button = st.button(
            "Run statistical test", type="primary", key="run_stat_numeric"
        )
    if "run_stat_button" in locals() and run_stat_button:
        try:
            stat_result = run_statistical_test(
                df, test_type, stat_outcome, group_column=stat_group,
                hypothesized_mean=hypothesized_mean if test_type == "One-sample t-test" else 0.0,
                alpha=stat_alpha,
            )
            st.session_state.statistical_test_result = stat_result
        except (ValueError, TypeError) as exc:
            st.session_state.statistical_test_result = None
            st.error(f"Statistical test could not be completed: {exc}")

    stat_result = st.session_state.get("statistical_test_result")
    if stat_result is not None:
        st.markdown("#### Test results")
        stat_metrics = st.columns(4)
        stat_metrics[0].metric("Observations", f"{stat_result['n']:,}")
        stat_metrics[1].metric("Test statistic", f"{stat_result['statistic']:.5g}")
        stat_metrics[2].metric("p-value", f"{stat_result['p_value']:.5g}")
        stat_metrics[3].metric(
            stat_result["effect_size_name"] or "Effect size",
            f"{stat_result['effect_size']:.4g}" if np.isfinite(stat_result["effect_size"]) else "Unavailable",
        )
        st.write("**Decision:** " + stat_result["interpretation"])
        st.caption(
            f"Degrees of freedom: {stat_result['degrees_of_freedom']} · "
            f"Alpha: {stat_result['alpha']}"
        )
        if stat_result.get("confidence_interval") is not None:
            low, high = stat_result["confidence_interval"]
            st.write(
                f"**{stat_result['confidence_interval_label']} ({1-stat_result['alpha']:.0%}):** "
                f"{low:.5g} to {high:.5g}"
            )
        if not stat_result["group_summary"].empty:
            st.markdown("#### Group summaries")
            st.dataframe(stat_result["group_summary"], use_container_width=True, hide_index=True)
        if not stat_result.get("posthoc_comparisons", pd.DataFrame()).empty:
            st.markdown("#### Post-hoc pairwise comparisons (Tukey HSD)")
            st.dataframe(
                stat_result["posthoc_comparisons"],
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Tukey HSD adjusts pairwise p-values and confidence intervals to control "
                "the family-wise error rate. It is shown only when Levene's test does not "
                "indicate unequal variances; it does not replace checking the study design "
                "or independence assumptions."
            )
        if not stat_result["contingency"].empty:
            st.markdown("#### Observed contingency table")
            st.dataframe(stat_result["contingency"], use_container_width=True)
            st.markdown("#### Expected frequencies under independence")
            st.dataframe(stat_result["expected_frequencies"], use_container_width=True)
        for warning in stat_result["warnings"]:
            st.warning(warning)
        with st.expander("Methodology and interpretation limits", expanded=True):
            for limitation in stat_result["limitations"]:
                st.write("• " + limitation)
        stat_export = pd.DataFrame([{
            "test": stat_result["test"], "outcome": stat_result["outcome"],
            "group": stat_result["group"], "n": stat_result["n"],
            "statistic": stat_result["statistic"], "p_value": stat_result["p_value"],
            "degrees_of_freedom": str(stat_result["degrees_of_freedom"]),
            "alpha": stat_result["alpha"],
            "significant_at_alpha": stat_result["significant_at_alpha"],
            "effect_size_name": stat_result["effect_size_name"],
            "effect_size": stat_result["effect_size"],
            "confidence_interval": str(stat_result["confidence_interval"]),
            "interpretation": stat_result["interpretation"],
        }])
        stat_exports = st.columns(2)
        stat_exports[0].download_button(
            "Download test summary (CSV)", safe_csv_bytes(stat_export),
            "statistical_test_summary.csv", "text/csv",
            key="download_stat_test_csv",
        )
        stat_exports[1].download_button(
            "Download test summary (Excel)", excel_bytes(stat_export),
            "statistical_test_summary.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_stat_test_excel",
        )
    st.caption(
        "This workspace does not verify random sampling, study design, independence, "
        "measurement validity or pre-registration. Tukey HSD is available after one-way "
        "ANOVA when the equal-variance assumption is not flagged; other test families "
        "and repeated-analysis workflows do not receive automatic multiplicity correction."
    )
 
with tabs[15]:
    st.subheader("Recommended next steps")
    st.caption("Suggestions are generated from the active dataset's structure, missingness, duplicates and column types. They are deterministic and do not replace research judgement.")
    recommendations = suggest_analyses(df)
    for item in recommendations:
        with st.container(border=True):
            head = st.columns([4, 1])
            head[0].markdown("#### " + item["title"])
            head[1].markdown("**" + item["priority"] + " priority**")
            st.write(item["reason"])
            st.caption("Suggested action: " + item["action"])
    st.info("Recommendations are refreshed from the active dataset after upload or cleaning. Choose actions based on your research question, data provenance and assumptions.")

with tabs[16]:
    st.subheader("Ask your data")
    st.caption("A local-first assistant that answers supported questions using calculations on the active dataset. No external AI service or paid API is used.")
    st.markdown("Try: *What is the average sales?* · *Which region has the highest sales?* · *How many missing values?* · *What are the strongest correlations?*")
    quick_prompts = [
        ("Dataset size", "How many rows and columns?"),
        ("Missing values", "How many missing values?"),
        ("Correlations", "What are the strongest correlations?"),
        ("Next steps", "What should I do next?"),
    ]
    quick_cols = st.columns(4)
    selected_prompt = None
    for index, (label, question) in enumerate(quick_prompts):
        if quick_cols[index].button(label, key=f"chat_quick_{index}", use_container_width=True):
            selected_prompt = question
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            evidence = message.get("evidence")
            if isinstance(evidence, pd.DataFrame) and not evidence.empty:
                st.dataframe(evidence, use_container_width=True, hide_index=True)
    typed_prompt = st.chat_input("Ask a question about your active dataset…", key="dataset_chat_input")
    prompt = typed_prompt or selected_prompt
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        result = answer_dataset_question(df, prompt, context=st.session_state.chat_context)
        st.session_state.chat_context = result.get("context", st.session_state.chat_context)
        st.session_state.chat_history.append({
            "role": "assistant", "content": result["answer"],
            "evidence": result.get("evidence"),
        })
        st.rerun()
    if st.button("Clear conversation", key="clear_dataset_chat"):
        st.session_state.chat_history = []
        st.session_state.chat_context = None
        st.rerun()


with tabs[17]:
    st.subheader("AI-generated analytical report")
    st.caption(
        "Generate an evidence-led narrative from the active dataset. Findings are calculated "
        "locally; the report does not invent statistical significance, causality or forecasts."
    )
    with st.form("technical_report_settings"):
        technical_title = st.text_input("Report title", "Portfolio / Dataset Technical Analysis Report")
        technical_type = st.selectbox(
            "Report type",
            ["General technical report", "Financial / investment performance report",
             "Research and policy report", "Monitoring and evaluation report",
             "Market and business intelligence report"],
        )
        technical_author = st.text_input("Prepared by", "")
        technical_purpose = st.text_area(
            "Report purpose, background or analytical context (optional)",
            placeholder="Describe the decision, assignment, reporting period, benchmark or key context...",
            height=90,
        )
        generate_technical = st.form_submit_button("Generate professional technical report", type="primary")
    if generate_technical:
        try:
            narrative = build_narrative_report(
                df, st.session_state.source or "Active dataset",
                report_title=technical_title.strip() or "Technical Data Analysis Report",
                prepared_by=technical_author.strip(),
                report_type=technical_type,
                purpose=technical_purpose.strip(),
            )
            report_findings = narrative["findings"] + [
                "Recommendation: " + item for item in narrative["recommendations"]
            ]
            financial_metrics = narrative.get("financial_analysis", {}).get("metrics", {})
            if "portfolio_return_pct" in financial_metrics:
                report_findings.append(
                    "Total portfolio return: {:.2f}% (P&L: USD {:,.2f}).".format(
                        financial_metrics["portfolio_return_pct"],
                        financial_metrics["portfolio_pnl"],
                    )
                )
            if "invested_sleeve_return_pct" in financial_metrics:
                report_findings.append(
                    "Invested-sleeve return: {:.2f}% (P&L: USD {:,.2f}).".format(
                        financial_metrics["invested_sleeve_return_pct"],
                        financial_metrics["invested_sleeve_pnl"],
                    )
                )
            if "reported_fees" in financial_metrics:
                report_findings.append(
                    "Reported fees / brokerage: USD {:,.2f}; not deducted again.".format(
                        financial_metrics["reported_fees"]
                    )
                )
            report_sections = [
                "Dataset overview", "Key findings", "Numeric summary",
                "Correlation matrix",
            ]
            if "financial" in technical_type.lower() or "investment" in technical_type.lower():
                report_sections.append("Financial performance attribution")
            narrative_pdf = build_pdf_report(
                st.session_state.source, df, p, numeric_summary(df),
                report_findings,
                report_title=narrative["title"],
                prepared_by=narrative["prepared_by"],
                executive_summary=narrative["executive_summary"],
                methodology=narrative["methodology"],
                sections=report_sections,
            )
            narrative_excel = excel_analysis_bytes(df, p, numeric_summary(df))
            st.session_state.narrative_report_artifacts = {
                "markdown": narrative["markdown"], "pdf": narrative_pdf,
                "xlsx": narrative_excel, "title": narrative["title"],
            }
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Technical report could not be generated: {exc}")

    narrative_artifacts = st.session_state.get("narrative_report_artifacts")
    if narrative_artifacts:
        st.markdown(narrative_artifacts["markdown"])
        safe_report_name = "data_analysis_report"
        export_cols = st.columns(3)
        export_cols[0].download_button(
            "Download Markdown report", narrative_artifacts["markdown"],
            f"{safe_report_name}.md", "text/markdown",
            key="download_narrative_markdown", use_container_width=True,
        )
        export_cols[1].download_button(
            "Download PDF report", narrative_artifacts["pdf"],
            f"{safe_report_name}.pdf", "application/pdf",
            key="download_narrative_pdf", use_container_width=True,
        )
        export_cols[2].download_button(
            "Download Excel workbook", narrative_artifacts["xlsx"],
            f"{safe_report_name}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_narrative_excel", use_container_width=True,
        )
    else:
        st.info("Generate the report to preview its findings and export Markdown, PDF and Excel files.")


with tabs[18]:
    st.subheader("Smart AI Assistant")
    st.caption("Ask general questions, explore your dataset, interpret analytical results, and draft technical reports.")
    provider = st.selectbox(
        "Assistant engine",
        ["Dataset tools (no AI server)", "Ollama (local model)", "OpenAI-compatible API (opt-in)"],
        help="Ollama is the default no-API-key AI engine and runs an open-source model on a machine/server you control. Dataset tools are the offline fallback. The optional API provider is not required.",
        index=["Dataset tools (no AI server)", "Ollama (local model)", "OpenAI-compatible API (opt-in)"].index(default_assistant_provider()),
        key="smart_assistant_provider",
    )
    if provider == "Ollama (local model)":
        st.info("No API key is required. Install Ollama and download a model (for example, `ollama pull llama3.1:8b`). When running the app on the same computer, the default address is `http://localhost:11434`. For Streamlit Cloud, localhost is the cloud container: point OLLAMA_BASE_URL to an Ollama server reachable over the internet. This is a server URL, not an API key.")
    elif provider == "OpenAI-compatible API (opt-in)":
        st.warning("External processing is enabled only when you choose this option. The assistant sends your question, recent chat context, dataset column names and aggregate summaries—not raw dataset rows. Configure OPENAI_API_KEY and optional OPENAI_BASE_URL / OPENAI_MODEL as deployment secrets.")
    else:
        st.info("Dataset tools are the offline fallback and do not use a generative model. For general AI conversations without an API key, select Ollama and connect it to a local or self-hosted Ollama server. A cloud deployment needs a reachable OLLAMA_BASE_URL; no API key is needed.")

    smart_prompts = [
        "Explain the most important patterns in this dataset.",
        "What are the main data-quality risks?",
        "Help me interpret the correlation results.",
        "Draft a professional technical report outline for these data.",
    ]
    prompt_cols = st.columns(4)
    selected_smart_prompt = None
    for idx, suggestion in enumerate(smart_prompts):
        if prompt_cols[idx].button(suggestion, key=f"smart_prompt_{idx}", use_container_width=True):
            selected_smart_prompt = suggestion

    for message in st.session_state.smart_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            evidence = message.get("evidence")
            if isinstance(evidence, pd.DataFrame) and not evidence.empty:
                st.dataframe(evidence, use_container_width=True, hide_index=True)

    smart_input = st.chat_input("Ask anything: data, research, finance, policy, statistics, writing...", key="smart_assistant_chat_input")
    smart_question = smart_input or selected_smart_prompt
    if smart_question:
        st.session_state.smart_chat_history.append({"role": "user", "content": smart_question})
        try:
            dataset_context = build_dataset_context(df, st.session_state.source or "Active dataset") if provider != "Dataset tools (no AI server)" else ""
            result = ask_smart_assistant(
                smart_question,
                history=st.session_state.smart_chat_history[:-1],
                dataset_context=dataset_context,
                provider=provider,
                df=df,
                context=st.session_state.smart_chat_context,
            )
            st.session_state.smart_chat_context = result.get("context", st.session_state.smart_chat_context)
            st.session_state.smart_chat_history.append({
                "role": "assistant", "content": result["answer"],
                "evidence": result.get("evidence"),
                "provider": result.get("provider"),
            })
        except (RuntimeError, ValueError) as exc:
            st.session_state.smart_chat_history.append({
                "role": "assistant", "content": f"Assistant unavailable: {exc}",
            })
        st.rerun()

    latest_assistant = next(
        (item for item in reversed(st.session_state.smart_chat_history)
         if item.get("role") == "assistant" and item.get("content")),
        None,
    )
    if latest_assistant:
        st.divider()
        st.markdown("#### Export latest assistant response")
        export_md, export_pdf = st.columns(2)
        export_md.download_button(
            "Download response as Markdown",
            latest_assistant["content"],
            "smart_assistant_response.md",
            "text/markdown",
            key="download_smart_response_md",
            use_container_width=True,
        )
        try:
            smart_pdf = build_text_report_pdf(
                latest_assistant["content"],
                report_title="AI Smart Assistant | Technical Report",
            )
            export_pdf.download_button(
                "Download response as PDF",
                smart_pdf,
                "smart_assistant_technical_report.pdf",
                "application/pdf",
                key="download_smart_response_pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.caption(f"PDF export unavailable: {exc}")

    if st.button("Clear smart assistant conversation", key="clear_smart_assistant"):
        st.session_state.smart_chat_history = []
        st.session_state.smart_chat_context = None
        st.rerun()
