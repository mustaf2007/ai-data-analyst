"""Deterministic dataset recommendations and a local-first data Q&A assistant."""
from __future__ import annotations

import re
import pandas as pd


def suggest_analyses(df: pd.DataFrame) -> list[dict]:
    """Recommend useful next steps from observable dataset characteristics."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return [{
            "title": "Load a dataset",
            "reason": "An uploaded or sample dataset is needed before analysis can begin.",
            "action": "Upload a CSV/XLSX file or load the built-in sample.",
            "priority": "High",
        }]

    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and 1 < df[c].nunique(dropna=True) <= 100
    ]
    suggestions = []

    def add(title, reason, action, priority="Medium"):
        suggestions.append({
            "title": title, "reason": reason, "action": action,
            "priority": priority,
        })

    missing = int(df.isna().sum().sum())
    if missing:
        pct = missing / max(1, df.shape[0] * df.shape[1]) * 100
        add("Review missing values",
            f"{missing:,} missing cells ({pct:.1f}% of the dataset) were detected.",
            "Open Clean data, inspect missingness by column, then choose a justified strategy.",
            "High" if pct >= 10 else "Medium")

    duplicates = int(df.duplicated().sum())
    if duplicates:
        add("Check duplicate records",
            f"{duplicates:,} fully duplicated rows were detected.",
            "Review duplicates in Data quality before deciding whether they should be removed.",
            "High")

    if len(numeric) >= 2:
        add("Explore relationships",
            f"{len(numeric)} numeric variables are available for correlation analysis.",
            "Open Relationships and compare Pearson, Spearman or Kendall associations.")
    elif len(numeric) == 1:
        add("Explore the numeric distribution",
            f"{numeric[0]} is the only detected numeric variable.",
            f"Open Explore to inspect the distribution and outliers of {numeric[0]}.")

    if categorical and numeric:
        add("Compare outcomes across groups",
            f"{len(categorical)} categorical grouping variable(s) and {len(numeric)} numeric measure(s) are available.",
            "Use grouped summaries, geographic intelligence (if relevant), or statistical testing.")
    if len(categorical) >= 2:
        add("Test categorical associations",
            "At least two categorical variables are available.",
            "Open Statistical testing and consider a chi-square test of independence.")

    date_like = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_like.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(100)
            if len(sample) and pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.8:
                date_like.append(col)
    if date_like and numeric:
        add("Analyze changes over time",
            f"Possible time field(s): {', '.join(map(str, date_like[:3]))}.",
            "Open Time-series intelligence to aggregate observations and inspect trends.")
    if len(df) >= 30 and numeric:
        add("Explore unusual observations",
            f"The dataset contains {len(df):,} rows and numeric variables suitable for a baseline anomaly screen.",
            "Open Anomalies and review flagged observations; investigate them before treating them as errors.")
    if len(df) >= 50 and numeric and len(df.columns) >= 2:
        add("Consider a predictive baseline",
            "There are enough rows for an initial supervised-model feasibility check.",
            "Open Predictive modeling, select a target, and review the validation metrics and limitations.")

    if not suggestions:
        add("Inspect the dataset structure",
            "No strong automated next step was detected from column types and basic quality checks.",
            "Start with Data quality and Explore, then choose an analysis based on your research question.")
    return suggestions[:8]


def _find_column(question: str, columns) -> object | None:
    q = question.casefold()
    matches = [c for c in columns if str(c).casefold() in q]
    return max(matches, key=lambda c: len(str(c))) if matches else None


def _answer_dataset_question(df: pd.DataFrame, question: str) -> dict:
    """Answer common dataset questions with pandas calculations, never generated guesses."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"answer": "Load a non-empty dataset first.", "evidence": pd.DataFrame(), "intent": "empty"}
    q = str(question or "").strip()
    if not q:
        return {"answer": "Please enter a question about the active dataset.", "evidence": pd.DataFrame(), "intent": "empty"}
    low = q.casefold()
    if any(term in low for term in ("recommend", "suggest", "what should i do", "next step", "where should i start")):
        recommendations = suggest_analyses(df)
        evidence = pd.DataFrame(recommendations)
        first = recommendations[0]
        return {
            "answer": f"I found {len(recommendations)} suggested next steps. The first recommendation is {first['title']}: {first['reason']} {first['action']}",
            "evidence": evidence,
            "intent": "recommendations",
        }
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and 1 < df[c].nunique(dropna=True) <= 200
    ]
    evidence = pd.DataFrame()

    # A quality-risk question should produce a dataset-wide audit, not the
    # generic unsupported-question message. All checks are deterministic.
    quality_terms = (
        "data quality", "quality risk", "quality issue", "quality problem",
        "data risks", "data risk", "data issues", "data problems",
        "data-quality", "quality of the data",
    )
    if any(term in low for term in quality_terms):
        rows = []
        total_cells = max(1, df.shape[0] * df.shape[1])
        missing_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        blank_counts = {}
        for col in df.select_dtypes(include=["object", "string"]).columns:
            values = df[col].dropna().astype(str)
            blanks = int(values.str.strip().eq("").sum())
            if blanks:
                blank_counts[str(col)] = blanks

        def add_quality(risk, status, detail):
            rows.append({"check": risk, "status": status, "finding": detail})

        add_quality(
            "Missing values",
            "Review" if missing_count else "No issue detected",
            f"{missing_count:,} missing cells ({missing_count / total_cells * 100:.2f}% of all cells).",
        )
        add_quality(
            "Fully duplicated rows",
            "Review" if duplicate_count else "No issue detected",
            f"{duplicate_count:,} fully duplicated rows among {len(df):,} records. This does not detect repeated business keys.",
        )
        blank_total = sum(blank_counts.values())
        add_quality(
            "Blank text values",
            "Review" if blank_total else "No issue detected",
            f"{blank_total:,} whitespace-only values" +
            (f" in: {', '.join(blank_counts)}." if blank_counts else "."),
        )

        constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
        add_quality(
            "Constant or empty columns",
            "Review" if constant_cols else "No issue detected",
            "Review: " + ", ".join(constant_cols) if constant_cols else "No constant or entirely empty columns detected.",
        )

        numeric_text_cols = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            values = df[col].dropna().astype(str).str.strip()
            if len(values) >= 2 and pd.to_numeric(values, errors="coerce").notna().mean() >= 0.8:
                numeric_text_cols.append(str(col))
        add_quality(
            "Numeric values stored as text",
            "Review" if numeric_text_cols else "No issue detected",
            "Possible numeric columns stored as text: " + ", ".join(numeric_text_cols)
            if numeric_text_cols else "No likely numeric-as-text columns detected.",
        )

        infinite_cols = [
            str(col) for col in numeric
            if pd.Series(pd.to_numeric(df[col], errors="coerce")).isin([float("inf"), float("-inf")]).any()
        ]
        add_quality(
            "Infinite numeric values",
            "Review" if infinite_cols else "No issue detected",
            "Infinite values found in: " + ", ".join(infinite_cols)
            if infinite_cols else "No positive or negative infinite values detected.",
        )

        high_cardinality = [
            str(col) for col in df.columns
            if df[col].nunique(dropna=True) > max(50, int(len(df) * 0.8))
        ]
        add_quality(
            "High-cardinality fields",
            "Context check" if high_cardinality else "No issue detected",
            "Review whether these fields are identifiers rather than analysis categories: " + ", ".join(high_cardinality)
            if high_cardinality else "No unusually high-cardinality fields detected by the screening rule.",
        )

        evidence = pd.DataFrame(rows)
        flagged = evidence[evidence["status"] != "No issue detected"]
        if flagged.empty:
            answer = (
                f"The initial automated screen found no obvious structural data-quality risks in "
                f"this dataset ({len(df):,} rows × {len(df.columns):,} columns). "
                "This is not a guarantee of accuracy: verify units, definitions, source consistency, "
                "valid ranges and whether repeated records are legitimate."
            )
        else:
            findings = "; ".join(
                f"{row['check'].lower()}: {row['finding']}"
                for _, row in flagged.iterrows()
            )
            answer = (
                f"The initial screen identified {len(flagged)} item(s) to review in this "
                f"{len(df):,}-row, {len(df.columns)}-column dataset: {findings} "
                "These are screening flags, not proof that the underlying records are wrong. "
                "Also verify units, year/category consistency, definitions and source coverage."
            )
        return {"answer": answer, "evidence": evidence, "intent": "data_quality"}

    if any(term in low for term in ("missing", "null", "empty cells", "incomplete")):
        counts = df.isna().sum().sort_values(ascending=False)
        evidence = pd.DataFrame({"column": counts.index.astype(str), "missing": counts.values})
        total = int(counts.sum())
        pct = total / max(1, df.shape[0] * df.shape[1]) * 100
        return {"answer": f"The dataset has {total:,} missing cells ({pct:.1f}% of all cells). The table shows missing values by column.", "evidence": evidence, "intent": "missing"}

    if any(term in low for term in ("duplicate", "duplicated", "repeated rows")):
        count = int(df.duplicated().sum())
        return {"answer": f"I found {count:,} fully duplicated row(s) out of {len(df):,} records. This checks complete-row duplicates, not repeated IDs or domain-specific duplicates.", "evidence": pd.DataFrame({"metric": ["Rows", "Fully duplicated rows"], "value": [len(df), count]}), "intent": "duplicates"}

    if any(term in low for term in ("how many rows", "number of rows", "row count", "how many records", "dataset size", "how many columns")):
        return {"answer": f"The active dataset contains {len(df):,} rows and {len(df.columns):,} columns.", "evidence": pd.DataFrame({"column": df.columns.astype(str), "data_type": [str(t) for t in df.dtypes], "missing": df.isna().sum().values}), "intent": "overview"}

    if any(term in low for term in ("correlation", "correlated", "relationship between")):
        if len(numeric) < 2:
            return {"answer": "Correlation analysis needs at least two numeric columns. This dataset does not currently meet that requirement.", "evidence": pd.DataFrame(), "intent": "correlation"}
        method = "spearman" if "spearman" in low else "kendall" if "kendall" in low else "pearson"
        corr = df[numeric].corr(method=method)
        pairs = corr.where(pd.DataFrame(
            [[i < j for j in range(len(numeric))] for i in range(len(numeric))],
            index=numeric, columns=numeric
        )).stack().reset_index()
        pairs.columns = ["variable_1", "variable_2", "correlation"]
        evidence = pairs.reindex(pairs.correlation.abs().sort_values(ascending=False).index).head(10)
        if evidence.empty:
            return {"answer": "There are not enough valid paired numeric observations to calculate correlations.", "evidence": evidence, "intent": "correlation"}
        top = evidence.iloc[0]
        return {"answer": f"Using {method.title()} correlation, the strongest absolute association is between {top.variable_1} and {top.variable_2} (r = {top.correlation:.3f}). Correlation does not establish causation.", "evidence": evidence, "intent": "correlation"}

    measure = _find_column(q, numeric)
    group = _find_column(q, categorical)
    if measure is not None:
        series = pd.to_numeric(df[measure], errors="coerce").dropna()
        if series.empty:
            return {"answer": f"No valid numeric observations were found in {measure}.", "evidence": pd.DataFrame(), "intent": "numeric"}
        if group is not None and (any(x in low for x in (" by ", "across ", "among ", "per ")) or str(group).casefold() in low):
            work = df[[group, measure]].copy()
            work[measure] = pd.to_numeric(work[measure], errors="coerce")
            work = work.dropna()
            agg = "sum" if any(x in low for x in ("total", "sum", "overall")) else "median" if "median" in low else "count" if "count" in low else "mean"
            grouped = work.groupby(group, dropna=False)[measure].agg(agg).reset_index(name=agg)
            grouped = grouped.sort_values(agg, ascending=not any(x in low for x in ("highest", "largest", "top", "maximum", "most")))
            evidence = grouped.head(15)
            direction = "highest" if any(x in low for x in ("highest", "largest", "top", "maximum", "most")) else "lowest" if any(x in low for x in ("lowest", "smallest", "minimum", "least")) else "first in the sorted summary"
            first = evidence.iloc[0]
            return {"answer": f"Grouped {measure} by {group} using {agg}. The {direction} group is {first[group]} ({first[agg]:,.4g}). The table shows up to 15 groups.", "evidence": evidence, "intent": "grouped"}
        if any(x in low for x in ("highest", "maximum", "largest", "maximum value", "max")):
            idx = series.idxmax()
            return {"answer": f"The maximum {measure} is {series.loc[idx]:,.4g} (original row index {idx}).", "evidence": pd.DataFrame({"measure": [measure], "value": [series.loc[idx]], "row_index": [idx]}), "intent": "maximum"}
        if any(x in low for x in ("lowest", "minimum", "smallest", "min")):
            idx = series.idxmin()
            return {"answer": f"The minimum {measure} is {series.loc[idx]:,.4g} (original row index {idx}).", "evidence": pd.DataFrame({"measure": [measure], "value": [series.loc[idx]], "row_index": [idx]}), "intent": "minimum"}
        if any(x in low for x in ("sum", "total")):
            value = float(series.sum())
            return {"answer": f"The total of {measure} is {value:,.4g}, calculated from {len(series):,} valid observations.", "evidence": pd.DataFrame({"measure": [measure], "total": [value], "valid_observations": [len(series)]}), "intent": "sum"}
        if any(x in low for x in ("average", "mean", "typical")):
            value = float(series.mean())
            return {"answer": f"The average {measure} is {value:,.4g}, based on {len(series):,} valid observations.", "evidence": pd.DataFrame({"measure": [measure], "mean": [value], "valid_observations": [len(series)]}), "intent": "mean"}
        if any(x in low for x in ("distribution", "describe", "summary", "statistics", "stats")):
            evidence = series.describe().rename_axis("statistic").reset_index(name="value")
            return {"answer": f"Here is the descriptive summary for {measure}.", "evidence": evidence, "intent": "describe"}

    if any(x in low for x in ("categories", "category", "most common", "frequency", "distribution")):
        col = _find_column(q, categorical) or (categorical[0] if len(categorical) == 1 else None)
        if col is not None:
            counts = df[col].astype("string").fillna("(Missing)").value_counts().head(15)
            evidence = counts.rename_axis("category").reset_index(name="count")
            first = evidence.iloc[0]
            return {"answer": f"The most frequent value in {col} is {first['category']} ({int(first['count']):,} records). Showing up to 15 categories.", "evidence": evidence, "intent": "frequency"}

    return {
        "answer": "I can answer questions about dataset size, missing values, duplicate rows, numeric summaries, minimum/maximum/mean/total values, category frequencies, grouped comparisons, and numeric correlations. Please mention the column name in your question. Example: “Which region has the highest sales?”",
        "evidence": pd.DataFrame(), "intent": "unsupported",
    }



def answer_dataset_question(
    df: pd.DataFrame,
    question: str,
    context: dict | None = None,
) -> dict:
    """Answer a question and carry lightweight, dataset-scoped context across turns."""
    context = context or {}
    q = str(question or "").strip()
    low = q.casefold()
    numeric = df.select_dtypes(include="number").columns.tolist() if isinstance(df, pd.DataFrame) else []
    categorical = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and 1 < df[c].nunique(dropna=True) <= 200
    ] if isinstance(df, pd.DataFrame) else []

    previous_measure = context.get("measure")
    previous_group = context.get("group")
    follow_up = any(
        phrase in low for phrase in (
            "what about", "how about", "and ", "instead", "also", "then ",
            "for the same", "that group", "those groups",
        )
    ) or low in {"median", "mean", "average", "sum", "total", "maximum", "minimum"}

    # Resolve short follow-ups such as "what about median?" against the last measure.
    measure = _find_column(q, numeric)
    group = _find_column(q, categorical)
    if follow_up and measure is None and previous_measure in numeric:
        measure = previous_measure
    if follow_up and group is None and previous_group in categorical:
        group = previous_group

    # A group-value follow-up (e.g. "What about North?") computes that group's
    # statistic directly rather than repeating the full grouped ranking.
    if follow_up and measure in numeric and group in categorical:
        group_values = df[group].dropna().astype(str).unique().tolist()
        selected_value = next(
            (value for value in sorted(group_values, key=len, reverse=True)
             if value.casefold() in low),
            None,
        )
        if selected_value is not None:
            work = df.loc[df[group].astype(str).str.casefold() == selected_value.casefold(), measure]
            values = pd.to_numeric(work, errors="coerce").dropna()
            agg = (
                "median" if "median" in low else
                "sum" if any(term in low for term in ("sum", "total")) else
                "max" if any(term in low for term in ("highest", "maximum", "max")) else
                "min" if any(term in low for term in ("lowest", "minimum", "min")) else
                "mean"
            )
            if not values.empty:
                value = float(getattr(values, agg)())
                label = {"mean": "average", "median": "median", "sum": "total",
                         "max": "maximum", "min": "minimum"}[agg]
                result = {
                    "answer": f"For {selected_value} in {group}, the {label} {measure} is {value:,.4g}, based on {len(values):,} valid observations.",
                    "evidence": pd.DataFrame({
                        group: [selected_value], "measure": [measure],
                        "statistic": [label], "value": [value],
                        "valid_observations": [len(values)],
                    }),
                    "intent": "group_follow_up",
                }
            else:
                result = {
                    "answer": f"There are no valid numeric {measure} observations for {selected_value} in {group}.",
                    "evidence": pd.DataFrame(), "intent": "group_follow_up",
                }
            result["context"] = {"measure": measure, "group": group}
            return result

    resolved_question = q
    if follow_up and measure in numeric and _find_column(q, numeric) is None:
        resolved_question = f"{q} {measure}"
    if follow_up and group in categorical and _find_column(q, categorical) is None:
        resolved_question = f"{resolved_question} {group}"

    result = _answer_dataset_question(df, resolved_question)
    detected_measure = _find_column(resolved_question, numeric) or previous_measure
    detected_group = _find_column(resolved_question, categorical) or previous_group
    result["context"] = {
        "measure": detected_measure if detected_measure in numeric else None,
        "group": detected_group if detected_group in categorical else None,
    }
    return result
