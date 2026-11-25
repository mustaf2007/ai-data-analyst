"""Dataset loading, profiling, cleaning and exploration helpers."""
from io import BytesIO
from zipfile import BadZipFile, ZipFile
import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 250_000
MAX_COLUMNS = 150

class DataInputError(ValueError):
    """Invalid or unsafe dataset input."""

def load_dataset(filename: str, content: bytes) -> pd.DataFrame:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"csv", "xlsx"}: raise DataInputError("Upload a CSV or XLSX file.")
    if not content: raise DataInputError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES: raise DataInputError("File exceeds 20 MB.")
    try:
        if ext == "csv": df = pd.read_csv(BytesIO(content), low_memory=False)
        else:
            try:
                with ZipFile(BytesIO(content)) as z:
                    if len(z.infolist()) > 1000 or sum(x.file_size for x in z.infolist()) > 100*1024*1024:
                        raise DataInputError("Excel archive exceeds expanded-size limit.")
            except BadZipFile as exc: raise DataInputError("Invalid XLSX archive.") from exc
            df = pd.read_excel(BytesIO(content), engine="openpyxl")
    except DataInputError: raise
    except Exception as exc: raise DataInputError(f"Could not read file: {exc}") from exc
    if df.empty: raise DataInputError("The file contains no data rows.")
    if len(df) > MAX_ROWS or df.shape[1] > MAX_COLUMNS: raise DataInputError("Dataset exceeds row or column safeguards.")
    cols, seen = [], {}
    for raw in df.columns:
        name = str(raw).strip() or "unnamed"
        seen[name] = seen.get(name, 0) + 1
        cols.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    df.columns = cols
    return df

def profile_dataset(df: pd.DataFrame) -> dict:
    if df.empty: raise DataInputError("No rows to profile.")
    cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    completeness = 100 * (1 - missing / max(cells, 1))
    duplicate_rate = 100 * duplicates / max(len(df), 1)
    return {"rows":len(df), "columns":len(df.columns), "memory_mb":df.memory_usage(deep=True).sum()/1048576,
      "missing_cells":missing, "missing_pct":100*missing/max(cells,1), "duplicate_rows":duplicates,
      "completeness_pct":completeness, "quality_score":max(0,min(100,.7*completeness+.3*(100-duplicate_rate))),
      "numeric_columns":df.select_dtypes(include=np.number).columns.tolist(),
      "categorical_columns":df.select_dtypes(include=["object","category","bool"]).columns.tolist(),
      "datetime_columns":df.select_dtypes(include=["datetime","datetimetz"]).columns.tolist(),
      "missing_by_column":df.isna().sum().sort_values(ascending=False).loc[lambda s:s>0]}

def column_profile(df):
    return pd.DataFrame([{"column":str(c),"dtype":str(df[c].dtype),"non_null":int(df[c].notna().sum()),
      "missing":int(df[c].isna().sum()),"missing_%":round(df[c].isna().mean()*100,2),
      "unique":int(df[c].nunique()),"sample":" | ".join(df[c].dropna().astype(str).head(3))}
      for c in df.columns])

def clean_dataset(df, *, drop_duplicates=False, drop_empty_columns=False,
                  missing_strategy="Keep as-is", selected_columns=None):
    out=df.copy()
    if drop_duplicates:
        out=out.drop_duplicates()
    if drop_empty_columns:
        # Treat nulls and whitespace-only strings as empty; preserve columns
        # containing any substantive value, even when their header is "Unnamed".
        empty_columns = [
            c for c in out.columns
            if out[c].isna().all()
            or (
                (pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]))
                and out[c].dropna().astype(str).str.strip().eq("").all()
            )
        ]
        out = out.drop(columns=empty_columns)
    cols=selected_columns or out.columns.tolist()
    if missing_strategy=="Drop rows with missing values":
        out=out.dropna(subset=[c for c in cols if c in out.columns])
    elif missing_strategy in {"Fill numeric with median", "Fill numeric with mean", "Fill numeric with zero",
                              "Fill numeric with median and categorical with mode"}:
        method="median" if "median" in missing_strategy else "mean" if "mean" in missing_strategy else "zero"
        for c in out.select_dtypes(include=np.number):
            if c in cols:
                v=out[c].median() if method=="median" else out[c].mean() if method=="mean" else 0
                if pd.notna(v):
                    out[c]=out[c].fillna(v)
        if missing_strategy=="Fill numeric with median and categorical with mode":
            for c in out.select_dtypes(include=["object","category","bool"]):
                if c in cols and not out[c].mode().empty:
                    out[c]=out[c].fillna(out[c].mode().iloc[0])
    elif missing_strategy=="Fill categorical with mode":
        for c in out.select_dtypes(include=["object","category","bool"]):
            if c in cols and not out[c].mode().empty:
                out[c]=out[c].fillna(out[c].mode().iloc[0])
    elif missing_strategy!="Keep as-is":
        raise ValueError("Unknown missing-value strategy.")
    return out


def column_quality_diagnostics(df):
    """Summarize missingness severity and likely schema issues by column."""
    rows = []
    for column in df.columns:
        series = df[column]
        missing_pct = float(series.isna().mean() * 100) if len(series) else 0.0
        if missing_pct == 0:
            severity = "None"
        elif missing_pct <= 5:
            severity = "Low"
        elif missing_pct <= 20:
            severity = "Moderate"
        else:
            severity = "High"

        name = str(column)
        header_issue = name.strip() == "" or name.lower().startswith("unnamed")
        numeric_text = False
        if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            non_null = series.dropna().astype(str).str.strip()
            if len(non_null) >= 2:
                converted = pd.to_numeric(non_null, errors="coerce")
                numeric_text = converted.notna().mean() >= 0.8

        rows.append({
            "column": name,
            "missing": int(series.isna().sum()),
            "missing_%": round(missing_pct, 2),
            "missingness_level": severity,
            "header_review": "Review name" if header_issue else "OK",
            "type_review": "Possibly numeric stored as text" if numeric_text else "OK",
        })
    return pd.DataFrame(rows)


def validate_analysis_readiness(df, analysis_type="General exploratory analysis", target_column=None):
    """Run generic and selected-task checks; not a substitute for domain validation."""
    checks = []

    def add(name, passed, detail, severity="warning"):
        checks.append({
            "check": name,
            "status": "Passed" if passed else ("Blocked" if severity == "blocker" else "Needs attention"),
            "detail": detail,
            "passed": bool(passed),
            "severity": severity,
        })

    rows, columns = df.shape
    add("Dataset is not empty", rows > 0, f"{rows:,} rows found.", "blocker")
    add("Columns are present", columns > 0, f"{columns:,} columns found.", "blocker")

    if columns:
        all_null = [str(c) for c in df.columns if df[c].isna().all()]
        add("No entirely empty columns", not all_null,
            "No entirely empty columns." if not all_null else "Empty columns: " + ", ".join(all_null))
        constant = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
        add("Variables have usable variation", not constant,
            "All columns have variation." if not constant else "Constant or all-missing columns: " + ", ".join(constant))
        suspicious_headers = [str(c) for c in df.columns
                              if str(c).strip() == "" or str(c).lower().startswith("unnamed")]
        add("Column names are informative", not suspicious_headers,
            "No blank or automatically generated 'Unnamed' headers."
            if not suspicious_headers else "Review these headers: " + ", ".join(suspicious_headers))
    else:
        add("No entirely empty columns", False, "No columns are available.", "blocker")
        add("Variables have usable variation", False, "No columns are available.", "blocker")
        add("Column names are informative", False, "No columns are available.", "blocker")

    numeric = df.select_dtypes(include=np.number)
    infinite_columns = [str(c) for c in numeric.columns
                        if np.isinf(numeric[c].to_numpy(dtype=float, na_value=np.nan)).any()]
    add("No infinite numeric values", not infinite_columns,
        "No positive or negative infinity values." if not infinite_columns
        else "Infinite values found in: " + ", ".join(infinite_columns), "blocker")

    text_numeric = []
    for c in df.select_dtypes(include=["object", "string"]).columns:
        values = df[c].dropna().astype(str).str.strip()
        if len(values) >= 2 and pd.to_numeric(values, errors="coerce").notna().mean() >= 0.8:
            text_numeric.append(str(c))
    add("Numeric-looking text reviewed", not text_numeric,
        "No likely numeric-as-text columns detected." if not text_numeric
        else "Consider converting these columns to numeric: " + ", ".join(text_numeric))

    missing = int(df.isna().sum().sum())
    add("No missing cells", missing == 0,
        "No missing cells." if missing == 0 else f"{missing:,} missing cells remain.")
    duplicates = int(df.duplicated().sum()) if rows else 0
    add("No duplicate rows", duplicates == 0,
        "No fully duplicated rows." if duplicates == 0 else f"{duplicates:,} duplicate rows remain.")
    add("At least two observations", rows >= 2,
        f"{rows:,} observations available." if rows >= 2
        else "At least two observations are generally needed for meaningful analysis.", "blocker")

    if analysis_type == "Correlation analysis":
        passed = len(numeric.columns) >= 2
        add("Correlation has at least two numeric variables", passed,
            f"{len(numeric.columns)} numeric variables found; at least two are required.", "blocker")
    elif analysis_type in {"Regression", "Classification"}:
        target_exists = target_column in df.columns if target_column is not None else False
        add("Selected target column exists", target_exists,
            f"Target: {target_column}" if target_exists else "Select a valid target column.", "blocker")
        if target_exists:
            target = df[target_column]
            target_non_null = target.dropna()
            if analysis_type == "Regression":
                target_numeric = pd.api.types.is_numeric_dtype(target)
                add("Regression target is numeric", target_numeric,
                    "Target has a numeric data type." if target_numeric
                    else "Convert the selected target to numeric for regression.", "blocker")
            else:
                classes = int(target_non_null.nunique())
                add("Classification target has at least two classes", classes >= 2,
                    f"{classes} distinct non-missing classes found.", "blocker")
            features = [c for c in df.columns if c != target_column and df[c].nunique(dropna=True) > 1]
            add("At least one varying feature is available", bool(features),
                f"{len(features)} candidate feature(s) found." if features
                else "No other varying columns are available as features.", "blocker")
    elif analysis_type == "Time-series analysis":
        date_columns = infer_datetime_columns(df)
        add("A date/time column is identifiable", bool(date_columns),
            "Detected: " + ", ".join(date_columns) if date_columns
            else "No date-like column was identified. Check date formats or convert a date column.", "blocker")

    result = pd.DataFrame(checks)
    passed_count = int(result["passed"].sum()) if not result.empty else 0
    total = len(result)
    blockers = int((result["status"] == "Blocked").sum())
    warnings = int((result["status"] == "Needs attention").sum())
    score = round(100 * passed_count / total, 1) if total else 0.0
    status = "Not ready" if blockers else ("Ready" if warnings == 0 else "Ready with warnings")
    return {
        "status": status, "score": score, "passed": passed_count, "total": total,
        "blockers": blockers, "warnings": warnings, "checks": result,
        "analysis_type": analysis_type,
    }


def numeric_summary(df):
    n=df.select_dtypes(include=np.number)
    return pd.DataFrame() if n.empty else n.describe().T.rename_axis("variable").reset_index()

def categorical_summary(df, top_n=10):
    return {str(c):df[c].astype("string").fillna("(Missing)").value_counts().head(top_n).rename_axis("value").reset_index(name="count")
      for c in df.select_dtypes(include=["object","category","bool"])}

def correlation_matrix(df, method="pearson"):
    n=df.select_dtypes(include=np.number)
    return pd.DataFrame() if n.shape[1]<2 else n.corr(method=method)

def grouped_summary(df, group_column, value_column, aggregation="mean"):
    if group_column not in df or value_column not in df: raise ValueError("Select valid columns.")
    if aggregation not in {"mean","median","sum","count","min","max"}: raise ValueError("Unsupported aggregation.")
    return df.groupby(group_column,dropna=False)[value_column].agg(aggregation).reset_index().sort_values(value_column,ascending=False)

def infer_datetime_columns(df, threshold=.8):
    result=[]
    for c in df.select_dtypes(include=["object","string"]):
        s=df[c].dropna().head(100)
        if not s.empty and pd.to_datetime(s,errors="coerce").notna().mean()>=threshold: result.append(str(c))
    return result

def generate_insights(df):
    if df.empty: return ["The active dataset contains no rows."]
    out=[]
    miss=df.isna().mean().mean()*100
    out.append("The active dataset has no missing cells." if miss==0 else f"Overall missingness is {miss:.1f}%.")
    d=int(df.duplicated().sum())
    if d: out.append(f"{d:,} fully duplicated rows are present.")
    n=df.select_dtypes(include=np.number)
    if n.shape[1]>=2:
        c=n.corr()
        pairs=c.where(np.triu(np.ones(c.shape),1).astype(bool)).stack().dropna()
        if not pairs.empty:
            p=pairs.abs().idxmax(); out.append(f"Strongest numeric correlation: {p[0]} and {p[1]} (r={c.loc[p[0],p[1]]:.2f}); association is not causation.")
    return out[:8]


def generate_detailed_insights(df, max_findings=30):
    """Create deterministic, evidence-backed observations without causal claims."""
    findings = []
    if df is None or df.empty:
        return findings

    rows, columns = df.shape
    findings.append({
        "area": "Dataset", "finding": f"The active dataset contains {rows:,} rows and {columns:,} columns.",
        "evidence": f"{rows:,} observations; {columns:,} variables.",
        "caveat": "Confirm that the rows represent the intended unit of analysis.",
    })

    missing = df.isna().sum()
    missing_total = int(missing.sum())
    if missing_total:
        worst = missing[missing > 0].sort_values(ascending=False).head(3)
        details = "; ".join(f"{col}: {int(value):,} ({value / rows:.1%})" for col, value in worst.items())
        findings.append({
            "area": "Data quality", "finding": f"{missing_total:,} missing cells ({missing_total / (rows * columns):.1%} of all cells).",
            "evidence": "Highest missingness: " + details,
            "caveat": "Investigate why values are missing before choosing deletion or imputation.",
        })

    duplicates = int(df.duplicated().sum())
    if duplicates:
        findings.append({
            "area": "Data quality", "finding": f"{duplicates:,} fully duplicated rows ({duplicates / rows:.1%} of records).",
            "evidence": f"{duplicates:,} duplicate rows detected using all columns.",
            "caveat": "Repeated records may be valid in some datasets; verify the record key before removal.",
        })

    numeric = df.select_dtypes(include=np.number)
    for col in numeric.columns:
        series = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 3 or series.nunique() < 2:
            continue
        mean, median = float(series.mean()), float(series.median())
        std = float(series.std(ddof=1))
        findings.append({
            "area": "Distribution",
            "finding": f"{col}: mean {mean:,.3g}; median {median:,.3g}.",
            "evidence": f"n={len(series):,}; standard deviation={std:,.3g}; min={series.min():,.3g}; max={series.max():,.3g}.",
            "caveat": "Large mean–median differences can indicate skew or influential observations; inspect the distribution.",
        })

    if numeric.shape[1] >= 2:
        corr = numeric.replace([np.inf, -np.inf], np.nan).corr(min_periods=3)
        pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().dropna()
        if not pairs.empty:
            pair = pairs.abs().sort_values(ascending=False).index[0]
            value = float(corr.loc[pair[0], pair[1]])
            findings.append({
                "area": "Relationships",
                "finding": f"{pair[0]} and {pair[1]} have the strongest observed absolute linear correlation (r={value:.2f}).",
                "evidence": f"Pearson correlation calculated from available paired observations; r={value:.4f}.",
                "caveat": "Correlation is not causation; check sample size, outliers, nonlinear patterns and confounding.",
            })

    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        counts = df[col].value_counts(dropna=True)
        if len(counts) >= 2:
            top = counts.index[0]
            findings.append({
                "area": "Categories",
                "finding": f"{col}: '{top}' is the most frequent category ({int(counts.iloc[0]):,} records, {counts.iloc[0] / rows:.1%} of all rows).",
                "evidence": f"{len(counts):,} observed non-missing categories; top category count={int(counts.iloc[0]):,}.",
                "caveat": "Category frequencies are descriptive and depend on how categories and missing values are coded.",
            })

    return findings[:max(0, int(max_findings))]


def safe_csv_bytes(df):
    out=df.copy()
    for c in out.select_dtypes(include=["object","string"]):
        out[c]=out[c].map(lambda v:"'"+v if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@","\t","\r")) else v)
    return out.to_csv(index=False).encode("utf-8-sig")


def build_cleaning_report(before, after, *, drop_duplicates=False,
                          missing_strategy="Keep as-is", selected_columns=None):
    """Return auditable before/after cleaning metrics without mutating either frame."""
    deduplicated = before.drop_duplicates() if drop_duplicates else before
    selected = list(selected_columns) if selected_columns else deduplicated.columns.tolist()
    missing_before = int(before.isna().sum().sum())
    missing_after_dedup = int(deduplicated.isna().sum().sum())
    missing_after = int(after.isna().sum().sum())

    filled_cells = 0
    rows_removed_for_missing = 0
    if missing_strategy.startswith("Fill "):
        eligible = set(selected)
        if missing_strategy.startswith("Fill numeric"):
            eligible &= set(deduplicated.select_dtypes(include=np.number).columns)
            if missing_strategy == "Fill numeric with median and categorical with mode":
                eligible |= (set(selected) & set(deduplicated.select_dtypes(include=["object", "category", "bool"]).columns))
        elif missing_strategy == "Fill categorical with mode":
            eligible &= set(deduplicated.select_dtypes(include=["object", "category", "bool"]).columns)
        eligible &= set(after.columns)
        filled_cells = int(deduplicated[[c for c in eligible if c in deduplicated.columns]].isna().sum().sum()
                           - after[[c for c in eligible if c in after.columns]].isna().sum().sum()) if eligible else 0
    elif missing_strategy == "Drop rows with missing values":
        rows_removed_for_missing = max(0, len(deduplicated) - len(after))

    before_profile = profile_dataset(before)
    after_profile = profile_dataset(after)
    per_column = pd.DataFrame({
        "column": before.columns.astype(str),
        "missing_before": [int(before[c].isna().sum()) for c in before.columns],
        "missing_after": [int(after[c].isna().sum()) if c in after.columns else 0 for c in before.columns],
        "column_removed": [c not in after.columns for c in before.columns],
    })
    per_column["missing_resolved"] = per_column["missing_before"] - per_column["missing_after"]

    return {
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "rows_removed": int(len(before) - len(after)),
        "duplicates_removed": int(len(before) - len(deduplicated)),
        "rows_removed_for_missing": int(rows_removed_for_missing),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "missing_filled": int(filled_cells),
        "quality_before": float(before_profile["quality_score"]),
        "quality_after": float(after_profile["quality_score"]),
        "per_column": per_column,
        "strategy": missing_strategy,
        "selected_columns": list(selected),
    }


def build_data_story(findings, *, title="Data analysis brief", audience="General", context=""):
    """Turn selected structured findings into a cautious, evidence-linked narrative."""
    if findings is None:
        findings = []
    records = findings.to_dict("records") if isinstance(findings, pd.DataFrame) else list(findings)
    records = [
        item for item in records
        if isinstance(item, dict) and item.get("finding")
    ]
    if not records:
        return f"{title.strip() or 'Data analysis brief'}\n\nNo supported findings were selected. Generate insights or select findings before drafting."

    groups = {}
    for item in records:
        groups.setdefault(str(item.get("area", "Other")), []).append(item)

    lines = [
        (title.strip() or "Data analysis brief"),
        f"Audience: {audience.strip() or 'General'}",
        "",
        "Purpose and scope",
        context.strip() or "This brief summarizes descriptive observations generated from the active dataset. It does not establish causal relationships.",
        "",
        "Key findings",
    ]
    for area, items in groups.items():
        lines.extend(["", area])
        for item in items:
            lines.append(f"- {str(item['finding']).strip()}")
            evidence = str(item.get("evidence", "")).strip()
            caveat = str(item.get("caveat", "")).strip()
            if evidence:
                lines.append(f"  Evidence: {evidence}")
            if caveat:
                lines.append(f"  Interpretation note: {caveat}")

    lines.extend([
        "",
        "Interpretation and limitations",
        "These findings are descriptive and depend on the coverage, definitions, units, sampling and quality of the supplied data. Correlation does not demonstrate causation. Check missingness, outliers and subgroup sizes before drawing conclusions.",
        "",
        "Recommended next steps",
        "1. Verify source definitions, time coverage, units and the observational unit.",
        "2. Investigate data-quality issues and document any cleaning decisions.",
        "3. Test important patterns with an appropriate statistical method and domain knowledge.",
        "4. Compare results with independent sources before using them to guide policy or operational decisions.",
    ])
    return "\n".join(lines)


def build_time_series_analysis(
    df, date_column, value_column, *, frequency="MS", aggregation="sum",
    moving_average_window=3,
):
    """Aggregate a numeric measure over time and calculate transparent trend diagnostics."""
    if date_column not in df.columns or value_column not in df.columns:
        raise ValueError("Select date and measure columns that exist in the dataset.")
    if date_column == value_column:
        raise ValueError("Date and measure columns must be different.")
    if frequency not in {"MS", "QS", "YS"}:
        raise ValueError("Frequency must be month, quarter or year.")
    if aggregation not in {"sum", "mean", "median", "count"}:
        raise ValueError("Unsupported aggregation method.")
    if not isinstance(moving_average_window, (int, np.integer)) or moving_average_window < 2:
        raise ValueError("Moving-average window must be an integer of at least 2.")

    date_values = df[date_column]
    if (
        pd.api.types.is_numeric_dtype(date_values)
        and str(date_column).strip().lower() in {"year", "yr", "yyyy"}
    ):
        numeric_years = pd.to_numeric(date_values, errors="coerce")
        dates = pd.to_datetime(
            numeric_years.where(numeric_years.between(1000, 9999)).round().astype("Int64").astype("string") + "-01-01",
            errors="coerce",
        )
    else:
        dates = pd.to_datetime(date_values, errors="coerce")
    values = pd.to_numeric(df[value_column], errors="coerce")
    valid = dates.notna() & values.notna() & np.isfinite(values)
    working = pd.DataFrame({"period": dates[valid], "value": values[valid]})
    warnings = []
    if len(working) < 2:
        raise ValueError("At least two valid date-and-measure observations are required.")

    grouped = working.set_index("period").resample(frequency)["value"]
    if aggregation == "count":
        result = grouped.count()
    else:
        result = getattr(grouped, aggregation)()
    result = result.dropna().sort_index()
    result = result[result.index.notna()]
    if len(result) < 2:
        raise ValueError("The selected frequency produces fewer than two non-empty periods.")

    output = pd.DataFrame({
        "period": result.index,
        "value": result.to_numpy(dtype=float),
    })
    output["period_change"] = output["value"].diff()
    previous = output["value"].shift(1)
    output["period_change_pct"] = np.where(
        previous.ne(0), output["period_change"] / previous.abs() * 100, np.nan
    )
    output["moving_average"] = output["value"].rolling(
        window=moving_average_window, min_periods=moving_average_window
    ).mean()

    x = np.arange(len(output), dtype=float)
    slope = float(np.polyfit(x, output["value"].to_numpy(dtype=float), 1)[0])
    if slope > 0:
        direction = "Increasing"
    elif slope < 0:
        direction = "Decreasing"
    else:
        direction = "No linear change"
    output["trend_line"] = np.polyval(
        np.polyfit(x, output["value"].to_numpy(dtype=float), 1), x
    )

    output["unusual_change"] = False
    changes = output["period_change"].dropna()
    if len(output) >= 6 and len(changes) >= 5:
        q1, q3 = changes.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr > 0:
            lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            output.loc[output["period_change"].notna(), "unusual_change"] = (
                (output.loc[output["period_change"].notna(), "period_change"] < lower)
                | (output.loc[output["period_change"].notna(), "period_change"] > upper)
            )
        else:
            warnings.append("Unusual-change flags were not calculated because period changes have no IQR spread.")
    else:
        warnings.append("At least six aggregated periods are required before unusual-change flags are assessed.")

    if output["period_change_pct"].isna().any():
        warnings.append("Percentage change is unavailable where the previous period is zero or missing.")
    if len(working) < len(df):
        warnings.append(f"{len(df) - len(working):,} rows were excluded because the date or measure was invalid or missing.")
    if len(output) < moving_average_window:
        warnings.append("There are too few periods to calculate the selected moving average.")

    summary = {
        "periods": int(len(output)),
        "observations": int(len(working)),
        "aggregation": aggregation,
        "frequency": frequency,
        "trend_direction": direction,
        "trend_slope_per_period": slope,
        "unusual_periods": int(output["unusual_change"].sum()),
    }
    return {"data": output, "summary": summary, "warnings": warnings}



def forecast_time_series_baseline(series, *, horizon=3, holdout_periods=None):
    """Evaluate a last-observation baseline on a chronological holdout, then forecast."""
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float).reset_index(drop=True)
    if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 12:
        raise ValueError("Forecast horizon must be an integer from 1 to 12.")
    if len(values) < 8:
        raise ValueError("At least eight observed periods are required for baseline forecasting.")

    if holdout_periods is None:
        holdout_periods = max(2, int(np.ceil(len(values) * 0.2)))
    if not isinstance(holdout_periods, (int, np.integer)) or holdout_periods < 1:
        raise ValueError("Holdout periods must be a positive integer.")
    if len(values) - holdout_periods < 3:
        raise ValueError("The chronological training sample must contain at least three periods.")

    split = len(values) - holdout_periods
    train = values.iloc[:split]
    actual = values.iloc[split:]
    validation_prediction = float(train.iloc[-1])
    errors = actual.to_numpy() - validation_prediction
    mae = float(np.mean(np.abs(errors)))
    denominator = np.abs(actual.to_numpy()) + abs(validation_prediction)
    smape_terms = np.divide(
        2 * np.abs(errors), denominator,
        out=np.zeros_like(errors, dtype=float), where=denominator != 0,
    )
    smape = float(np.mean(smape_terms) * 100)

    future = pd.DataFrame({
        "step_ahead": np.arange(1, horizon + 1, dtype=int),
        "forecast": np.repeat(float(values.iloc[-1]), horizon),
        "method": "Last-observation baseline",
    })
    validation = pd.DataFrame({
        "step": np.arange(1, len(actual) + 1, dtype=int),
        "actual": actual.to_numpy(),
        "predicted": np.repeat(validation_prediction, len(actual)),
    })
    return {
        "forecast": future,
        "validation": validation,
        "metrics": {
            "training_periods": int(len(train)),
            "holdout_periods": int(len(actual)),
            "mae": mae,
            "smape_pct": smape,
        },
        "limitations": [
            "This is a last-observation baseline, not a fitted forecasting model.",
            "The forecast assumes the latest observed value remains constant.",
            "The holdout evaluates one historical window and may not represent future conditions.",
            "Seasonality, structural breaks and external drivers are not modeled.",
        ],
    }



def compare_datasets(reference, comparison, *, categorical_limit=30):
    """Compare two datasets at schema and summary-statistic level without assuming row alignment."""
    if not isinstance(reference, pd.DataFrame) or not isinstance(comparison, pd.DataFrame):
        raise ValueError("Both inputs must be pandas DataFrames.")
    if reference.empty or comparison.empty:
        raise ValueError("both datasets must contain at least one data row.")
    if len(reference.columns) == 0 or len(comparison.columns) == 0:
        raise ValueError("Both datasets must contain at least one column.")
    if not isinstance(categorical_limit, int) or categorical_limit < 1:
        raise ValueError("Categorical limit must be a positive integer.")

    def overview(label, frame):
        cells = int(frame.shape[0] * frame.shape[1])
        missing = int(frame.isna().sum().sum())
        return {
            "dataset": label,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "missing_cells": missing,
            "missing_pct": round(100 * missing / max(cells, 1), 2),
            "duplicate_rows": int(frame.duplicated().sum()),
        }

    reference_columns = list(reference.columns)
    comparison_columns = list(comparison.columns)
    shared = [column for column in reference_columns if column in comparison_columns]
    all_columns = list(dict.fromkeys(reference_columns + comparison_columns))
    schema_rows = []
    for column in all_columns:
        in_reference, in_comparison = column in reference.columns, column in comparison.columns
        schema_rows.append({
            "column": str(column),
            "reference_present": in_reference,
            "comparison_present": in_comparison,
            "reference_dtype": str(reference[column].dtype) if in_reference else None,
            "comparison_dtype": str(comparison[column].dtype) if in_comparison else None,
            "type_matches": (
                str(reference[column].dtype) == str(comparison[column].dtype)
                if in_reference and in_comparison else False
            ),
        })
    schema = pd.DataFrame(schema_rows)

    numeric_rows = []
    categorical_frames = []
    warnings = []
    for column in shared:
        left, right = reference[column], comparison[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = pd.to_numeric(left, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            right_values = pd.to_numeric(right, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if left_values.empty or right_values.empty:
                warnings.append(f"{column}: numeric comparison skipped because one dataset has no finite values.")
                continue
            left_mean, right_mean = float(left_values.mean()), float(right_values.mean())
            mean_delta = right_mean - left_mean
            numeric_rows.append({
                "column": str(column),
                "reference_n": int(len(left_values)),
                "comparison_n": int(len(right_values)),
                "reference_mean": left_mean,
                "comparison_mean": right_mean,
                "mean_difference": mean_delta,
                "mean_difference_pct": (
                    mean_delta / abs(left_mean) * 100 if left_mean != 0 else np.nan
                ),
                "reference_median": float(left_values.median()),
                "comparison_median": float(right_values.median()),
                "reference_std": float(left_values.std(ddof=1)) if len(left_values) > 1 else np.nan,
                "comparison_std": float(right_values.std(ddof=1)) if len(right_values) > 1 else np.nan,
                "reference_min": float(left_values.min()),
                "comparison_min": float(right_values.min()),
                "reference_max": float(left_values.max()),
                "comparison_max": float(right_values.max()),
            })
        elif (
            pd.api.types.is_object_dtype(left)
            or pd.api.types.is_string_dtype(left)
            or pd.api.types.is_categorical_dtype(left)
            or pd.api.types.is_bool_dtype(left)
        ) and (
            pd.api.types.is_object_dtype(right)
            or pd.api.types.is_string_dtype(right)
            or pd.api.types.is_categorical_dtype(right)
            or pd.api.types.is_bool_dtype(right)
        ):
            left_counts = left.astype("string").fillna("(Missing)").value_counts(dropna=False)
            right_counts = right.astype("string").fillna("(Missing)").value_counts(dropna=False)
            categories = list(dict.fromkeys(
                left_counts.index.tolist() + right_counts.index.tolist()
            ))
            if len(categories) > categorical_limit:
                warnings.append(
                    f"{column}: categorical distribution skipped ({len(categories)} categories; "
                    f"limit is {categorical_limit})."
                )
                continue
            left_pct = left_counts / max(len(left), 1) * 100
            right_pct = right_counts / max(len(right), 1) * 100
            for category in categories:
                a_pct = float(left_pct.get(category, 0.0))
                b_pct = float(right_pct.get(category, 0.0))
                categorical_frames.append({
                    "column": str(column),
                    "category": str(category),
                    "reference_count": int(left_counts.get(category, 0)),
                    "comparison_count": int(right_counts.get(category, 0)),
                    "reference_pct": a_pct,
                    "comparison_pct": b_pct,
                    "difference_percentage_points": b_pct - a_pct,
                })

    numeric = pd.DataFrame(numeric_rows)
    categorical = pd.DataFrame(categorical_frames)
    summary = pd.DataFrame([
        overview("Reference", reference),
        overview("Comparison", comparison),
    ])
    if not shared:
        warnings.append("The datasets have no shared column names; only dataset-level and schema comparisons are available.")
    return {
        "summary": summary,
        "schema": schema,
        "numeric": numeric,
        "categorical": categorical,
        "shared_columns": shared,
        "warnings": warnings,
        "limitations": [
            "This compares dataset-level summaries and matching column names; it does not assume rows represent the same entities or periods.",
            "Differences may reflect different coverage, definitions, units, sampling or time windows.",
            "A percentage difference is unavailable when the reference mean is zero.",
            "Summary differences are descriptive and do not establish causes or statistical significance.",
        ],
    }



def build_geographic_summary(
    df, geographic_column, measure_column, *, aggregation="sum", top_n=25
):
    """Aggregate a numeric measure by a user-selected geographic/category field."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The dataset must contain at least one data row.")
    if geographic_column not in df.columns or measure_column not in df.columns:
        raise ValueError("Selected geographic and measure columns must exist.")
    if geographic_column == measure_column:
        raise ValueError("Choose different columns for geography and measure.")
    if aggregation not in {"sum", "mean", "median", "min", "max"}:
        raise ValueError("Aggregation must be sum, mean, median, min or max.")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("Top N must be a positive integer.")

    work = df[[geographic_column, measure_column]].copy()
    work["_geography"] = work[geographic_column].astype("string").str.strip()
    work["_measure"] = pd.to_numeric(work[measure_column], errors="coerce")
    work["_measure"] = work["_measure"].replace([np.inf, -np.inf], np.nan)
    missing_geo = work["_geography"].isna() | work["_geography"].eq("")
    invalid_measure = work["_measure"].isna()
    valid = work.loc[~missing_geo & ~invalid_measure, ["_geography", "_measure"]]
    if valid.empty:
        raise ValueError("No rows contain both a valid geographic label and a finite numeric measure.")

    grouped = valid.groupby("_geography", dropna=True)["_measure"]
    result = grouped.agg(aggregation).reset_index()
    result.columns = ["geography", "value"]
    counts = valid.groupby("_geography")["_measure"].size().rename("observations")
    result = result.merge(counts, left_on="geography", right_index=True, how="left")
    result = result.sort_values(["value", "geography"], ascending=[False, False], kind="stable").reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    total = float(result["value"].sum())
    if aggregation == "sum" and total > 0 and (result["value"] >= 0).all():
        result["share_pct"] = result["value"] / total * 100
    else:
        result["share_pct"] = np.nan

    warnings = []
    if missing_geo.any():
        warnings.append(f"{int(missing_geo.sum())} rows were excluded because the geographic label was missing or blank.")
    if invalid_measure.any():
        warnings.append(f"{int(invalid_measure.sum())} rows were excluded because the selected measure was missing or non-finite.")
    if result["geography"].nunique() == 1:
        warnings.append("Only one geographic group is represented; a regional ranking is not meaningful.")
    if aggregation == "sum" and not result["value"].ge(0).all():
        warnings.append("Share percentages are omitted because one or more aggregated values are negative.")
    elif aggregation == "sum" and total <= 0:
        warnings.append("Share percentages are omitted because the aggregate total is zero or negative.")

    return {
        "data": result.head(top_n).reset_index(drop=True),
        "all_data": result,
        "coverage": {
            "input_rows": int(len(df)),
            "included_rows": int(len(valid)),
            "excluded_rows": int(len(df) - len(valid)),
            "geographic_groups": int(result["geography"].nunique()),
            "measure": str(measure_column),
            "geographic_field": str(geographic_column),
            "aggregation": aggregation,
        },
        "warnings": warnings,
        "limitations": [
            "The selected column is treated as a geographic label; the function does not verify official country, region, district or boundary codes.",
            "Different geographic levels should not be mixed in one comparison.",
            "Results depend on the selected aggregation, units, time period, coverage and underlying observation grain.",
            "A ranking is descriptive and does not explain why geographic values differ.",
            "Map visualization requires valid geographic identifiers or coordinates; names alone may be ambiguous.",
        ],
    }


def run_statistical_test(
    df, test_type, outcome_column, *, group_column=None,
    hypothesized_mean=0.0, alpha=0.05
):
    """Run a small set of assumption-aware classical tests on independent observations."""
    from scipy import stats

    supported = {
        "One-sample t-test", "Independent-samples Welch t-test",
        "One-way ANOVA", "Chi-square test of independence",
    }
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Provide a non-empty pandas DataFrame.")
    if test_type not in supported:
        raise ValueError("Choose a supported statistical test.")
    if outcome_column not in df.columns:
        raise ValueError("The selected outcome column must exist.")
    if not 0 < float(alpha) < 1:
        raise ValueError("Alpha must be strictly between 0 and 1.")

    warnings = []
    result = {
        "test": test_type, "outcome": str(outcome_column),
        "group": str(group_column) if group_column is not None else None,
        "alpha": float(alpha), "n": 0, "statistic": np.nan,
        "p_value": np.nan, "degrees_of_freedom": None,
        "effect_size_name": None, "effect_size": np.nan,
        "confidence_interval": None, "group_summary": pd.DataFrame(),
        "contingency": pd.DataFrame(), "expected_frequencies": pd.DataFrame(),
        "warnings": warnings,
    }

    if test_type in {"One-sample t-test", "Independent-samples Welch t-test", "One-way ANOVA"}:
        values = pd.to_numeric(df[outcome_column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if test_type == "One-sample t-test":
            clean = values.dropna().astype(float)
            if len(clean) < 2:
                raise ValueError("A one-sample t-test requires at least two finite observations.")
            if clean.nunique() < 2:
                raise ValueError("The outcome has zero variance; a t statistic cannot be estimated.")
            mu = float(hypothesized_mean)
            if not np.isfinite(mu):
                raise ValueError("The hypothesized mean must be finite.")
            test = stats.ttest_1samp(clean, popmean=mu)
            n = len(clean)
            mean = float(clean.mean())
            sd = float(clean.std(ddof=1))
            se = sd / np.sqrt(n)
            critical = float(stats.t.ppf(1 - alpha / 2, n - 1))
            result.update({
                "n": int(n), "statistic": float(test.statistic),
                "p_value": float(test.pvalue), "degrees_of_freedom": int(n - 1),
                "effect_size_name": "Cohen's d (one-sample)",
                "effect_size": float((mean - mu) / sd),
                "confidence_interval": [mean - critical * se, mean + critical * se],
                "group_summary": pd.DataFrame([{
                    "group": "All observations", "n": n, "mean": mean, "std": sd
                }]),
            })
            result["comparison"] = f"Sample mean compared with {mu:g}"
            result["confidence_interval_label"] = "Confidence interval for the sample mean"
        else:
            if group_column is None or group_column not in df.columns:
                raise ValueError("Select a valid grouping column.")
            work = pd.DataFrame({
                "_value": values,
                "_group": df[group_column].astype("string").str.strip(),
            })
            work = work.dropna(subset=["_value", "_group"])
            work = work[work["_group"].ne("")]
            grouped = [
                (str(label), part["_value"].astype(float).to_numpy())
                for label, part in work.groupby("_group", sort=True)
            ]
            if len(grouped) < 2:
                raise ValueError("At least two non-empty groups are required.")
            if any(len(arr) < 2 for _, arr in grouped):
                raise ValueError("Each group must contain at least two finite observations.")
            if any(np.var(arr, ddof=1) == 0 for _, arr in grouped):
                warnings.append("At least one group has zero within-group variance; interpret the test cautiously.")
            summary = pd.DataFrame([
                {"group": label, "n": int(len(arr)), "mean": float(np.mean(arr)),
                 "std": float(np.std(arr, ddof=1))}
                for label, arr in grouped
            ])
            result["n"] = int(sum(len(arr) for _, arr in grouped))
            result["group_summary"] = summary
            if test_type == "Independent-samples Welch t-test":
                if len(grouped) != 2:
                    raise ValueError("The Welch t-test requires exactly two groups; use one-way ANOVA for more.")
                (label_a, x), (label_b, y) = grouped
                if np.var(x, ddof=1) == 0 and np.var(y, ddof=1) == 0:
                    raise ValueError("Both groups have zero variance; the test cannot be estimated.")
                test = stats.ttest_ind(x, y, equal_var=False)
                vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
                se2 = vx / len(x) + vy / len(y)
                if se2 <= 0 or not np.isfinite(se2):
                    raise ValueError("The standard error could not be estimated.")
                df_welch = se2 ** 2 / (
                    (vx / len(x)) ** 2 / (len(x) - 1)
                    + (vy / len(y)) ** 2 / (len(y) - 1)
                )
                se = np.sqrt(se2)
                critical = float(stats.t.ppf(1 - alpha / 2, df_welch))
                pooled_var = (
                    (len(x) - 1) * vx + (len(y) - 1) * vy
                ) / (len(x) + len(y) - 2)
                pooled_sd = np.sqrt(pooled_var)
                effect = (float(np.mean(x)) - float(np.mean(y))) / pooled_sd if pooled_sd > 0 else np.nan
                difference = float(np.mean(x) - np.mean(y))
                result.update({
                    "statistic": float(test.statistic), "p_value": float(test.pvalue),
                    "degrees_of_freedom": float(df_welch),
                    "effect_size_name": "Cohen's d (pooled-SD standardization)",
                    "effect_size": float(effect),
                    "confidence_interval": [
                        difference - critical * se, difference + critical * se
                    ],
                    "comparison": f"{label_a} minus {label_b}",
                    "confidence_interval_label": "Confidence interval for the mean difference",
                })
                if not np.isfinite(effect):
                    warnings.append("Cohen's d is unavailable because pooled within-group variance is zero.")
            else:
                if len(grouped) < 2:
                    raise ValueError("One-way ANOVA requires at least two groups.")
                test = stats.f_oneway(*(arr for _, arr in grouped))
                grand_mean = float(work["_value"].mean())
                ss_between = sum(
                    len(arr) * (float(np.mean(arr)) - grand_mean) ** 2
                    for _, arr in grouped
                )
                ss_total = float(((work["_value"] - grand_mean) ** 2).sum())
                eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
                levene = stats.levene(*(arr for _, arr in grouped), center="median")
                if np.isfinite(levene.pvalue) and levene.pvalue < alpha:
                    warnings.append("Levene's test suggests unequal group variances; classical one-way ANOVA may be unreliable. Consider Welch ANOVA or a robust alternative.")
                result.update({
                    "statistic": float(test.statistic), "p_value": float(test.pvalue),
                    "degrees_of_freedom": [len(grouped) - 1, result["n"] - len(grouped)],
                    "effect_size_name": "Eta-squared",
                    "effect_size": float(eta_squared),
                    "comparison": "At least one group mean differs",
                    "confidence_interval_label": None,
                })
                if not np.isfinite(eta_squared):
                    warnings.append("Eta-squared is unavailable because total outcome variance is zero.")
        if values.isna().any():
            warnings.append("Rows with missing, non-numeric or non-finite outcome values were excluded.")
        if len(result["group_summary"]) and (result["group_summary"]["n"] < 5).any():
            warnings.append("At least one group has fewer than five observations; distributional assumptions are difficult to assess.")
        warnings.append("The t-test and ANOVA assume independent observations; t-based inference also relies on an approximately normal sampling distribution, especially for small samples.")
    else:
        if group_column is None or group_column not in df.columns:
            raise ValueError("Select a valid second categorical column for the chi-square test.")
        if outcome_column == group_column:
            raise ValueError("Choose two different categorical columns.")
        left = df[outcome_column].astype("string").str.strip()
        right = df[group_column].astype("string").str.strip()
        valid = left.notna() & right.notna() & left.ne("") & right.ne("")
        if not valid.all():
            warnings.append("Rows with missing or blank categories were excluded.")
        table = pd.crosstab(left[valid], right[valid])
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError("The chi-square test requires at least two categories in each variable.")
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
        n = int(table.to_numpy().sum())
        denom = min(table.shape[0] - 1, table.shape[1] - 1)
        cramer_v = np.sqrt(chi2 / (n * denom)) if n > 0 and denom > 0 else np.nan
        expected_frame = pd.DataFrame(
            expected, index=table.index, columns=table.columns
        )
        low_expected = int((expected < 5).sum())
        if (expected < 1).any() or low_expected / expected.size > 0.2:
            warnings.append(
                f"{low_expected} of {expected.size} expected cell counts are below 5 or at least one is below 1; the chi-square approximation may be unreliable."
            )
        result.update({
            "n": n, "statistic": float(chi2), "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "effect_size_name": "Cramer's V", "effect_size": float(cramer_v),
            "contingency": table, "expected_frequencies": expected_frame,
            "comparison": "The two categorical variables are associated",
            "confidence_interval_label": None,
        })
        warnings.append("The chi-square test assumes independent observations and mutually exclusive categories; sparse expected counts can invalidate the approximation.")

    result["significant_at_alpha"] = bool(
        np.isfinite(result["p_value"]) and result["p_value"] < alpha
    )
    result["interpretation"] = (
        "The result is statistically significant at the selected alpha level."
        if result["significant_at_alpha"]
        else "The result is not statistically significant at the selected alpha level."
    )
    result["limitations"] = [
        "Statistical significance does not measure practical importance; interpret the effect size and confidence interval where available.",
        "A p-value does not establish causation or the probability that the null hypothesis is true.",
        "The validity of inference depends on study design, sampling, measurement quality and independent observations.",
        "Multiple testing increases false-positive risk; adjust the analysis plan when many hypotheses are tested.",
    ]
    return result
