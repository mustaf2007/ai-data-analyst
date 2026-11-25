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
    DataInputError, load_dataset, profile_dataset, column_profile, clean_dataset, build_time_series_analysis,
    numeric_summary, categorical_summary, correlation_matrix, grouped_summary,
    infer_datetime_columns, generate_insights, generate_detailed_insights, build_data_story, safe_csv_bytes, build_cleaning_report, validate_analysis_readiness, column_quality_diagnostics,
)
from modeling import train_predictive_model, detect_anomalies
from reporting import build_pdf_report

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Universal Data Intelligence · Local-first analytics · Open-source Python · No paid APIs")


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
    if upload_signature is not None:
        st.session_state.upload_signature = upload_signature
    clear_derived_results()


initialise_state()

with st.sidebar:
    st.header("Workspace")
    selected_theme = st.selectbox(
        "Appearance", ["Light", "Dark", "Green"],
        index=0, help="Choose the visual theme for charts and the workspace.",
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

if st.session_state.active is None:
    st.info("Upload a CSV/XLSX dataset or load the built-in sample to begin.")
    st.markdown("**Explore · Clean · Visualize · Model · Report**")
    st.stop()

df = st.session_state.active
p = profile_dataset(df)
theme_templates = {"Light": "plotly_white", "Dark": "plotly_dark", "Green": "seaborn"}
theme_template = theme_templates[selected_theme]
px.defaults.template = theme_template
theme_css = {
    "Light": ("#f4f7fb", "#ffffff", "#172b4d", "#64748b"),
    "Dark": ("#101827", "#182437", "#e5edf8", "#a8b6c9"),
    "Green": ("#eef6f0", "#ffffff", "#173b2b", "#557363"),
}
page_bg, card_bg, text_color, muted_color = theme_css[selected_theme]
st.markdown(
    f"""
    <style>
    .stApp {{ background: {page_bg}; color: {text_color}; }}
    [data-testid="stMetric"] {{
        background: {card_bg}; border: 1px solid rgba(120,140,160,.20);
        padding: 14px 16px; border-radius: 14px;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px; }}
    .block-container {{ padding-top: 1.5rem; }}
    h1, h2, h3 {{ letter-spacing: -0.025em; }}
    </style>
    """,
    unsafe_allow_html=True,
)
tabs = st.tabs([
    "Dashboard", "Data quality", "Clean data", "Explore", "Relationships",
    "Predictive modeling", "Anomalies", "Export & report", "Chart Studio", "Automated insights", "Data story", "Time-series intelligence",
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
            for warning in ts_result["warnings"]:
                st.warning(warning)
            st.caption(
                "Interpretation limits: trends are descriptive, not causal. Unusual-change flags use an "
                "IQR rule and are prompts for investigation, not proof of errors. Check date coverage, "
                "units, missing periods, seasonality and the meaning of the selected aggregation."
            )
        except (ValueError, TypeError) as exc:
            st.warning(f"Time-series analysis could not be completed: {exc}")
