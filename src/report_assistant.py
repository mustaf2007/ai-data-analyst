"""Deterministic, evidence-led narrative reports from tabular datasets."""
from __future__ import annotations

import pandas as pd

from analytics import profile_dataset


def _fmt(value):
    if pd.isna(value):
        return "not available"
    if isinstance(value, (int, float)):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def build_narrative_report(
    df: pd.DataFrame, source: str = "Uploaded dataset", *,
    report_title: str = "Technical Data Analysis Report",
    prepared_by: str = "", report_type: str = "General technical report",
    purpose: str = "",
) -> dict:
    """Create a structured Markdown report using calculations on the supplied data.

    The function does not infer causality or statistical significance. It reports
    descriptive evidence only and returns the underlying sections for export.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("A non-empty pandas DataFrame is required.")

    profile = profile_dataset(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and 1 < df[c].nunique(dropna=True) <= 100
    ]
    findings, recommendations, numeric_rows, category_rows = [], [], [], []

    if profile["missing_cells"]:
        recommendations.append(
            f"Investigate {profile['missing_cells']:,} missing cells "
            f"({profile['missing_pct']:.1f}% of all cells) before selecting a treatment."
        )
    if profile["duplicate_rows"]:
        recommendations.append(
            f"Review {profile['duplicate_rows']:,} fully duplicated rows; confirm whether "
            "they are genuine repeats before removing them."
        )
    if not profile["missing_cells"] and not profile["duplicate_rows"]:
        findings.append("No missing cells or fully duplicated rows were detected.")

    if numeric_cols:
        for col in numeric_cols[:12]:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            numeric_rows.append({
                "Variable": str(col), "N": int(series.size),
                "Mean": series.mean(), "Median": series.median(),
                "Std. dev.": series.std(), "Minimum": series.min(),
                "Maximum": series.max(),
            })
        if numeric_rows:
            top = max(numeric_rows, key=lambda x: abs(x["Mean"]) if pd.notna(x["Mean"]) else -1)
            findings.append(
                f"Among the numeric variables summarized, {top['Variable']} has a "
                f"mean of {_fmt(top['Mean'])} (n={top['N']:,})."
            )
    else:
        recommendations.append(
            "No numeric variables were detected. Review category distributions or "
            "derive valid numeric measures if appropriate for the research question."
        )

    for col in categorical_cols[:8]:
        counts = df[col].astype("string").fillna("(Missing)").value_counts(dropna=False)
        if counts.empty:
            continue
        category_rows.append({
            "Variable": str(col), "Most frequent category": str(counts.index[0]),
            "Count": int(counts.iloc[0]),
            "Share (%)": 100 * float(counts.iloc[0]) / max(len(df), 1),
            "Distinct values": int(df[col].nunique(dropna=True)),
        })
        findings.append(
            f"For {col}, the most frequent category is {str(counts.index[0])} "
            f"({int(counts.iloc[0]):,} records; {100*float(counts.iloc[0])/max(len(df),1):.1f}% of rows)."
        )

    correlation_rows = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(method="pearson", min_periods=3)
        pairs = []
        for i, first in enumerate(numeric_cols):
            for second in numeric_cols[i + 1:]:
                value = corr.loc[first, second]
                if pd.notna(value):
                    pairs.append((str(first), str(second), float(value)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for first, second, value in pairs[:5]:
            correlation_rows.append({
                "Variable 1": first, "Variable 2": second,
                "Pearson r": value, "Absolute r": abs(value),
            })
        if pairs:
            first, second, value = pairs[0]
            findings.append(
                f"The strongest observed Pearson association is between {first} and "
                f"{second} (r={value:.3f}); this is an association, not evidence of causation."
            )
        else:
            recommendations.append(
                "No usable pairwise correlations were available; inspect numeric variation "
                "and overlapping non-missing observations."
            )

    if len(df) < 30:
        recommendations.append(
            "The dataset contains fewer than 30 rows. Treat subgroup patterns and "
            "model estimates cautiously, and consider whether the sample represents the target population."
        )
    if not recommendations:
        recommendations.append(
            "Define the decision or research question, validate variable definitions and units, "
            "then select an analysis that matches the study design."
        )

    quality = (
        f"- Missing cells: {profile['missing_cells']:,} "
        f"({profile['missing_pct']:.2f}%)\n"
        f"- Fully duplicated rows: {profile['duplicate_rows']:,}\n"
        f"- Completeness: {profile['completeness_pct']:.1f}%\n"
        f"- Heuristic quality score: {profile['quality_score']:.1f}/100 "
        "(not a certification)"
    )
    numeric_table = pd.DataFrame(numeric_rows)
    category_table = pd.DataFrame(category_rows)
    correlation_table = pd.DataFrame(correlation_rows)

    def markdown_table(frame):
        if frame.empty:
            return "_No applicable results._"
        display = frame.copy()
        for col in display.select_dtypes(include="number").columns:
            display[col] = display[col].map(_fmt)
        headers = [str(column) for column in display.columns]
        rows = [[str(value) for value in row] for row in display.astype(str).itertuples(index=False, name=None)]
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(["| " + " | ".join(headers) + " |", separator] + body)

    executive = (
        f"The dataset **{source}** contains {profile['rows']:,} rows and "
        f"{profile['columns']:,} columns. It has {profile['missing_cells']:,} missing "
        f"cells ({profile['missing_pct']:.1f}%) and {profile['duplicate_rows']:,} "
        "fully duplicated rows. The findings below are descriptive and should be "
        "interpreted alongside the data's provenance, definitions and sampling design."
    )
    methodology = (
        "This report uses descriptive calculations on the active dataset: row/column "
        "counts, missingness, exact duplicate-row counts, numeric means/medians/standard "
        "deviations/ranges, category frequencies and pairwise Pearson correlations "
        "(minimum three overlapping observations). No inferential hypothesis test, "
        "causal model or forecast is run by this report generator. The quality score is "
        "a heuristic (70% completeness, 30% row uniqueness). Missingness mechanisms, "
        "sampling representativeness, units, measurement validity and data provenance "
        "are not established by these calculations."
    )
    md = [
        f"# {report_title}", f"**Report type:** {report_type}",
        f"**Source:** {source}", f"**Prepared by:** {prepared_by or 'Not specified'}",
        f"**Dataset size:** {profile['rows']:,} rows × {profile['columns']:,} columns",
        "", "## Executive summary", executive,
        "", "## Data quality and coverage", quality,
        "", "## Key findings",
    ]
    md.extend([f"- {item}" for item in findings] or ["- No descriptive findings could be calculated."])
    md += ["", "## Numeric summary", markdown_table(numeric_table),
           "", "## Category distributions", markdown_table(category_table),
           "", "## Strongest pairwise correlations", markdown_table(correlation_table),
           "", "## Recommended next steps"]
    md.extend([f"- {item}" for item in recommendations])
    md += ["", "## Methodology and limitations", methodology]
    return {
        "title": report_title, "source": source, "report_type": report_type,\n        "prepared_by": prepared_by, "purpose": purpose,
        "executive_summary": executive, "methodology": methodology,
        "findings": findings, "recommendations": recommendations,
        "numeric_summary": numeric_table, "category_summary": category_table,
        "correlations": correlation_table, "markdown": "\n".join(md),
        "profile": profile,
    }
