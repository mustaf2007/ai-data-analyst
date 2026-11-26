"""Question-led, evidence-based research report generation for tabular datasets.

All analytical results are computed locally. The research question guides variable
selection and test suggestions; it does not authorize causal interpretation.
"""
from __future__ import annotations

from io import BytesIO
import re

import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px

from analytics import profile_dataset


def _tokens(text):
    return {t for t in re.findall(r"[a-z0-9_]+", str(text).lower()) if len(t) > 2}


def _match_columns(question, columns):
    q = _tokens(question)
    ranked = []
    for col in columns:
        name = _tokens(col)
        overlap = len(q & name)
        if overlap:
            ranked.append((overlap, str(col)))
    return [name for _, name in sorted(ranked, reverse=True)]


def _fmt(value, digits=3):
    if value is None or pd.isna(value):
        return "not available"
    if isinstance(value, (int, float, np.number)):
        return f"{float(value):,.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _numeric_summary(df):
    rows = []
    for col in df.select_dtypes(include="number").columns[:30]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "Variable": str(col), "N": int(s.size), "Missing": int(df[col].isna().sum()),
            "Mean": s.mean(), "Median": s.median(), "Std. dev.": s.std(),
            "Minimum": s.min(), "Maximum": s.max(),
        })
    return pd.DataFrame(rows)


def _category_summary(df):
    rows = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique(dropna=True) not in range(2, 51):
            continue
        counts = df[col].astype("string").fillna("(Missing)").value_counts(dropna=False)
        for label, count in counts.head(15).items():
            rows.append({
                "Variable": str(col), "Category": str(label), "Count": int(count),
                "Share (%)": 100 * float(count) / max(len(df), 1),
            })
    return pd.DataFrame(rows)


def _correlations(df):
    cols = df.select_dtypes(include="number").columns[:30]
    if len(cols) < 2:
        return pd.DataFrame()
    corr = df[cols].corr(method="pearson", min_periods=3)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            n = int(df[[a, b]].dropna().shape[0])
            r = corr.loc[a, b]
            if pd.notna(r):
                rows.append({"Variable 1": str(a), "Variable 2": str(b),
                             "Pearson r": float(r), "N": n})
    return pd.DataFrame(rows).sort_values("Pearson r", key=lambda x: x.abs(), ascending=False) if rows else pd.DataFrame()


def _choose_test(df, question):
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                   and 1 < df[c].nunique(dropna=True) <= 20]
    q = _tokens(question)
    target_matches = _match_columns(question, numeric)
    if not target_matches:
        target_matches = _match_columns(question, categorical)
    target = target_matches[0] if target_matches else None
    group_matches = [c for c in _match_columns(question, categorical) if c != target]
    if not group_matches:
        group_matches = [c for c in categorical if c != target]
    group = group_matches[0] if group_matches else None

    result = {"test": "Descriptive analysis only", "outcome": target, "group": group,
              "n": 0, "statistic": None, "p_value": None, "effect_size": None,
              "effect_size_name": None, "interpretation": "No inferential test was selected.",
              "warnings": [], "method": "No inferential test was run; the available structure or question did not support a defensible automatic comparison."}
    if target in numeric and group:
        subset = df[[target, group]].dropna()
        levels = subset[group].unique()
        samples = [subset.loc[subset[group] == level, target].astype(float).values for level in levels]
        samples = [x for x in samples if len(x) >= 2]
        if len(samples) == 2:
            test = stats.ttest_ind(samples[0], samples[1], equal_var=False)
            pooled = np.sqrt((np.var(samples[0], ddof=1) + np.var(samples[1], ddof=1)) / 2)
            d = (np.mean(samples[0]) - np.mean(samples[1])) / pooled if pooled > 0 else np.nan
            result.update(test="Welch independent-samples t-test", n=len(subset),
                          statistic=float(test.statistic), p_value=float(test.pvalue),
                          effect_size=float(d) if np.isfinite(d) else None,
                          effect_size_name="Cohen's d (approx.)",
                          interpretation=("Evidence of a difference in group means at alpha=0.05; this is not proof of causation."
                                          if test.pvalue < .05 else
                                          "Insufficient evidence of a difference in group means at alpha=0.05."),
                          method="Welch's independent-samples t-test, two-sided; unequal variances are not assumed.")
            if min(map(len, samples)) < 5:
                result["warnings"].append("At least one comparison group has fewer than five observations.")
        elif len(samples) >= 3:
            test = stats.f_oneway(*samples)
            grand = subset[target].mean()
            ss_between = sum(len(x) * (np.mean(x) - grand) ** 2 for x in samples)
            ss_total = ((subset[target] - grand) ** 2).sum()
            eta = ss_between / ss_total if ss_total > 0 else np.nan
            result.update(test="One-way ANOVA", n=len(subset), statistic=float(test.statistic),
                          p_value=float(test.pvalue), effect_size=float(eta) if np.isfinite(eta) else None,
                          effect_size_name="Eta-squared",
                          interpretation=("Evidence that at least one group mean differs at alpha=0.05; follow-up comparisons are not performed."
                                          if test.pvalue < .05 else
                                          "Insufficient evidence that group means differ at alpha=0.05."),
                          method="One-way ANOVA; assumes independent observations and approximately normal residuals with comparable variances.")
            result["warnings"].append("Automatic ANOVA does not run post-hoc pairwise comparisons or verify independence/normality/homoscedasticity.")
        else:
            result["warnings"].append("Not enough groups with at least two observations for a group comparison.")
    elif target in categorical and group and target != group:
        subset = df[[target, group]].dropna()
        table = pd.crosstab(subset[group], subset[target])
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            test = stats.chi2_contingency(table, correction=False)
            n = int(table.to_numpy().sum())
            denom = min(table.shape[0] - 1, table.shape[1] - 1)
            v = np.sqrt(test.statistic / (n * denom)) if n and denom else np.nan
            result.update(test="Chi-square test of independence", n=n,
                          statistic=float(test.statistic), p_value=float(test.pvalue),
                          effect_size=float(v) if np.isfinite(v) else None,
                          effect_size_name="Cramer's V",
                          interpretation=("Evidence of an association between the categorical variables at alpha=0.05; this is not causation."
                                          if test.pvalue < .05 else
                                          "Insufficient evidence of an association at alpha=0.05."),
                          method="Pearson chi-square test of independence without continuity correction.")
            if (test.expected_freq < 5).any():
                result["warnings"].append("Some expected cell counts are below five; the chi-square approximation may be unreliable.")
        else:
            result["warnings"].append("The selected categorical variables do not form a usable contingency table.")
    elif len(numeric) >= 2 and any(x in q for x in {"relationship", "correlation", "associated", "association", "related"}):
        matches = _match_columns(question, numeric)
        a, b = (matches + numeric)[:2]
        pair = df[[a, b]].dropna()
        if len(pair) >= 3 and pair[a].nunique() > 1 and pair[b].nunique() > 1:
            test = stats.pearsonr(pair[a], pair[b])
            result.update(test="Pearson correlation test", outcome=a, group=b, n=len(pair),
                          statistic=float(test.statistic), p_value=float(test.pvalue),
                          effect_size=float(test.statistic), effect_size_name="Pearson r",
                          interpretation=("Evidence of a linear association at alpha=0.05; correlation does not establish causation."
                                          if test.pvalue < .05 else
                                          "Insufficient evidence of a linear association at alpha=0.05."),
                          method="Two-sided Pearson correlation test on complete pairs.")
    return result


def _charts(df, question):
    charts = []
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                   and 1 < df[c].nunique(dropna=True) <= 20]
    matched = _match_columns(question, numeric + categorical)
    if numeric:
        col = matched[0] if matched and matched[0] in numeric else numeric[0]
        fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
        charts.append({"title": f"Distribution of {col}", "figure": fig,
                       "data": df[[col]].dropna().copy()})
    if categorical:
        col = matched[0] if matched and matched[0] in categorical else categorical[0]
        counts = df[col].astype("string").fillna("(Missing)").value_counts().head(15).rename_axis("Category").reset_index(name="Count")
        fig = px.bar(counts, x="Category", y="Count", title=f"Frequency of {col}")
        charts.append({"title": f"Frequency of {col}", "figure": fig, "data": counts})
    if len(numeric) >= 2:
        preferred = [c for c in matched if c in numeric]
        x, y = (preferred + numeric)[:2]
        fig = px.scatter(df, x=x, y=y, title=f"{y} against {x}", trendline=None)
        charts.append({"title": f"{y} against {x}", "figure": fig,
                       "data": df[[x, y]].dropna().copy()})
    return charts


def _recommendations(profile, test, correlations, categories):
    recs = []
    if profile["missing_cells"]:
        recs.append({"Priority": "High", "Recommendation": "Investigate missingness patterns and causes before imputation or modelling.",
                     "Evidence": f'{profile["missing_cells"]:,} missing cells ({profile["missing_pct"]:.1f}% of all cells).'})
    if profile["duplicate_rows"]:
        recs.append({"Priority": "Medium", "Recommendation": "Review duplicated records against source identifiers and collection procedures before removing them.",
                     "Evidence": f'{profile["duplicate_rows"]:,} exact duplicate rows.'})
    if test.get("p_value") is not None:
        recs.append({"Priority": "High" if test["p_value"] < .05 else "Medium",
                     "Recommendation": "Validate the study design and assumptions, then investigate the observed group difference or association with domain context.",
                     "Evidence": f'{test["test"]}: p={test["p_value"]:.4g}; effect size {test.get("effect_size_name") or "not available"}={_fmt(test.get("effect_size"))}.'})
    if not recs:
        recs.append({"Priority": "Medium", "Recommendation": "Confirm variable definitions, units, sampling design and the decision the analysis must support.",
                     "Evidence": "The automated screening did not identify a specific data-quality issue."})
    if not correlations.empty:
        strongest = correlations.iloc[0]
        recs.append({"Priority": "Medium", "Recommendation": "Examine the strongest observed numeric relationship with a scatter plot and domain-informed checks; do not interpret it causally.",
                     "Evidence": f'{strongest["Variable 1"]} and {strongest["Variable 2"]}: r={strongest["Pearson r"]:.3f}, n={int(strongest["N"])}.'})
    return recs


def build_research_report(df: pd.DataFrame, research_question: str, *,
                          title="Research and Data Analysis Report",
                          objectives="", prepared_by="", source="Uploaded dataset",
                          alpha=0.05):
    """Build report sections and evidence from a question and an uploaded DataFrame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Upload a non-empty dataset before generating a report.")
    if not str(research_question).strip():
        raise ValueError("Enter a research question to guide the report.")
    if alpha not in (0.01, 0.05, 0.10):
        raise ValueError("Alpha must be 0.01, 0.05 or 0.10.")
    profile = profile_dataset(df)
    numeric = _numeric_summary(df)
    categories = _category_summary(df)
    correlations = _correlations(df)
    test = _choose_test(df, research_question)
    if test.get("p_value") is not None:
        test["significant_at_alpha"] = bool(test["p_value"] < alpha)
        test["alpha"] = alpha
        if test["p_value"] >= alpha:
            test["interpretation"] = test["interpretation"].replace("alpha=0.05", f"alpha={alpha:g}")
        else:
            test["interpretation"] = test["interpretation"].replace("alpha=0.05", f"alpha={alpha:g}")
    charts = _charts(df, research_question)
    recommendations = _recommendations(profile, test, correlations, categories)
    findings = [
        f"The dataset contains {profile['rows']:,} records and {profile['columns']:,} variables.",
        f"{profile['missing_cells']:,} cells are missing ({profile['missing_pct']:.1f}% of all cells); {profile['duplicate_rows']:,} exact duplicate rows were detected.",
    ]
    if not correlations.empty:
        r = correlations.iloc[0]
        findings.append(f"The strongest pairwise Pearson correlation is between {r['Variable 1']} and {r['Variable 2']} (r={r['Pearson r']:.3f}, n={int(r['N'])}); this is association, not causation.")
    if test.get("p_value") is not None:
        findings.append(f"{test['test']} returned statistic={test['statistic']:.4g}, p={test['p_value']:.4g}; {test['interpretation']}")
    else:
        findings.append("No inferential test was run automatically; the report prioritises descriptive evidence where the available variables do not support a defensible automatic test.")
    methodology = [
        f"The research question was: {research_question.strip()}",
        "The active uploaded dataset was analysed locally without sending raw records to an external AI service.",
        "Data screening included row and column counts, missing-cell percentage and exact duplicate-row counts.",
        "Numeric variables were summarised using non-missing observation counts, mean, median, standard deviation, minimum and maximum. Categorical variables were summarised using observed frequencies and shares.",
        "Pairwise Pearson correlations were calculated for numeric variables using complete pairs with at least three observations. Correlation is descriptive and does not establish causality.",
        test.get("method", "No inferential test was performed."),
        f"The decision threshold was alpha={alpha:g}. Statistical tests are exploratory and are not proof of a causal effect.",
        "The automated workflow cannot establish sampling representativeness, measurement validity, temporal ordering, independence, confounding control or data provenance. Review the data dictionary and study design before relying on the results.",
    ]
    limitations = [
        "Automatic variable selection uses question/column-name matching and dataset structure; review whether the selected variables genuinely represent the research question.",
        "No causal inference, sample weighting, missing-data imputation, multiple-testing correction or forecast is performed.",
        "Statistical assumptions and sampling design are not fully verified by this automated workflow.",
        "Results describe the uploaded data only; external generalisation requires a defensible sampling frame and study design.",
    ] + test.get("warnings", [])
    summary = (f"This report examines: {research_question.strip()} The uploaded dataset contains "
               f"{profile['rows']:,} rows and {profile['columns']:,} columns. "
               f"It identifies {len(findings)} evidence statements and {len(recommendations)} recommended next steps. "
               "Interpret all results in light of the stated methods and limitations.")
    report = {
        "title": title, "research_question": research_question.strip(), "objectives": objectives,
        "prepared_by": prepared_by, "source": source, "alpha": alpha,
        "profile": profile, "numeric_summary": numeric, "category_summary": categories,
        "correlations": correlations, "test": test, "findings": findings,
        "recommendations": pd.DataFrame(recommendations), "methodology": methodology,
        "limitations": limitations, "charts": charts, "executive_summary": summary,
    }
    report["markdown"] = _markdown_report(report)
    return report


def _markdown_report(r):
    lines = [f"# {r['title']}", f"**Source:** {r['source']}",
             f"**Prepared by:** {r['prepared_by'] or 'Not specified'}",
             "", "## Research question", r["research_question"]]
    if r["objectives"]:
        lines += ["", "## Objectives", r["objectives"]]
    lines += ["", "## Executive summary", r["executive_summary"],
              "", "## Dataset and data quality",
              f"- Records: {r['profile']['rows']:,}",
              f"- Variables: {r['profile']['columns']:,}",
              f"- Missing cells: {r['profile']['missing_cells']:,} ({r['profile']['missing_pct']:.2f}%)",
              f"- Exact duplicate rows: {r['profile']['duplicate_rows']:,}",
              "", "## Key findings"]
    lines += [f"- {x}" for x in r["findings"]]
    lines += ["", "## Statistical analysis",
              f"- Test: {r['test']['test']}",
              f"- Method: {r['test']['method']}",
              f"- Sample size: {r['test']['n']:,}",
              f"- Statistic: {_fmt(r['test']['statistic'])}",
              f"- p-value: {_fmt(r['test']['p_value'], 5)}",
              f"- Effect size: {r['test'].get('effect_size_name') or 'Not available'} = {_fmt(r['test'].get('effect_size'))}",
              f"- Interpretation: {r['test']['interpretation']}"]
    lines += ["", "## Methodology"] + [f"- {x}" for x in r["methodology"]]
    lines += ["", "## Recommendations"]
    for _, row in r["recommendations"].iterrows():
        lines.append(f"- **{row['Priority']} priority:** {row['Recommendation']} Evidence: {row['Evidence']}")
    lines += ["", "## Limitations"] + [f"- {x}" for x in r["limitations"]]
    return "\n".join(lines)


def build_research_excel(df, report):
    """Create a multi-sheet evidence workbook with analysis tables and source data."""
    output = BytesIO()
    overview = pd.DataFrame([
        {"Metric": "Research question", "Value": report["research_question"]},
        {"Metric": "Source", "Value": report["source"]},
        {"Metric": "Rows", "Value": report["profile"]["rows"]},
        {"Metric": "Columns", "Value": report["profile"]["columns"]},
        {"Metric": "Missing cells", "Value": report["profile"]["missing_cells"]},
        {"Metric": "Duplicate rows", "Value": report["profile"]["duplicate_rows"]},
        {"Metric": "Alpha", "Value": report["alpha"]},
    ])
    test = pd.DataFrame([{k: v for k, v in report["test"].items() if k not in {"warnings"}}])
    methodology = pd.DataFrame({"Methodology and limitations": report["methodology"] + report["limitations"]})
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="Report overview", index=False)
        pd.DataFrame({"Key findings": report["findings"]}).to_excel(writer, sheet_name="Key findings", index=False)
        report["numeric_summary"].to_excel(writer, sheet_name="Numeric summary", index=False)
        report["category_summary"].to_excel(writer, sheet_name="Category summary", index=False)
        report["correlations"].to_excel(writer, sheet_name="Correlations", index=False)
        test.to_excel(writer, sheet_name="Statistical test", index=False)
        report["recommendations"].to_excel(writer, sheet_name="Recommendations", index=False)
        methodology.to_excel(writer, sheet_name="Methodology", index=False)
        for i, chart in enumerate(report["charts"], 1):
            chart["data"].to_excel(writer, sheet_name=f"Chart data {i}", index=False)
        # Preserve the supplied data for reproducibility; truncate only at Excel's row limit.
        df.iloc[:1_048_575, :].to_excel(writer, sheet_name="Source data", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cells in ws.columns:
                letter = cells[0].column_letter
                width = min(max(max(len(str(c.value or "")) for c in cells[:200]) + 2, 12), 42)
                ws.column_dimensions[letter].width = width
    return output.getvalue()


def build_research_pdf(report):
    """Render the report, tables and automatically generated charts into a PDF."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from xml.sax.saxutils import escape

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ResearchBody", parent=styles["BodyText"], fontSize=8.5, leading=12, spaceAfter=5))
    styles.add(ParagraphStyle(name="ResearchSmall", parent=styles["BodyText"], fontSize=7, leading=9))
    story = [Paragraph(escape(report["title"]), styles["Title"]),
             Paragraph("Research question: " + escape(report["research_question"]), styles["ResearchBody"]),
             Paragraph("Source: " + escape(str(report["source"])) + " | Prepared by: " + escape(report["prepared_by"] or "Not specified"), styles["ResearchSmall"]),
             Spacer(1, 5*mm)]
    def para(text, style="ResearchBody"):
        story.append(Paragraph(escape(str(text)), styles[style]))
    def add_table(frame, max_rows=18):
        if frame is None or frame.empty:
            para("No applicable results were available.", "ResearchSmall")
            return
        data = [[str(c)[:24] for c in frame.columns]]
        data += [[str(v)[:48] for v in row] for row in frame.head(max_rows).fillna("").astype(str).values.tolist()]
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),
                                   ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                   ("GRID",(0,0),(-1,-1),.25,colors.HexColor("#D5DCE5")),
                                   ("FONTSIZE",(0,0),(-1,-1),6.5),("VALIGN",(0,0),(-1,-1),"TOP"),
                                   ("PADDING",(0,0),(-1,-1),4)]))
        story.append(table)
    para("Executive summary", "Heading2"); para(report["executive_summary"])
    para("Dataset and data quality", "Heading2")
    para(f"Rows: {report['profile']['rows']:,} | Columns: {report['profile']['columns']:,} | Missing: {report['profile']['missing_cells']:,} ({report['profile']['missing_pct']:.2f}%) | Exact duplicates: {report['profile']['duplicate_rows']:,}")
    para("Key findings", "Heading2")
    for item in report["findings"]: para("• " + item)
    para("Statistical analysis", "Heading2")
    for key in ["test", "method", "n", "statistic", "p_value", "effect_size_name", "effect_size", "interpretation"]:
        para(f"{key.replace('_',' ').title()}: {_fmt(report['test'].get(key))}")
    para("Numeric summary", "Heading2"); add_table(report["numeric_summary"])
    para("Category summary", "Heading2"); add_table(report["category_summary"])
    para("Correlation evidence", "Heading2"); add_table(report["correlations"])
    for chart in report["charts"]:
        para(chart["title"], "Heading2")
        try:
            image_bytes = chart["figure"].to_image(format="png", width=1000, height=520, scale=1)
            story.append(Image(BytesIO(image_bytes), width=170*mm, height=88*mm))
        except Exception:
            para("Static chart rendering was unavailable in this environment; the underlying chart data is included in the Excel workbook.", "ResearchSmall")
    para("Methodology", "Heading2")
    for item in report["methodology"]: para("• " + item)
    para("Recommendations", "Heading2")
    for _, row in report["recommendations"].iterrows():
        para(f"{row['Priority']} priority — {row['Recommendation']} Evidence: {row['Evidence']}")
    para("Limitations", "Heading2")
    for item in report["limitations"]: para("• " + item)
    doc.build(story)
    return buf.getvalue()
