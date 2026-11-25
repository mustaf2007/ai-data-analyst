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

def clean_dataset(df, *, drop_duplicates=False, missing_strategy="Keep as-is", selected_columns=None):
    out=df.copy()
    if drop_duplicates: out=out.drop_duplicates()
    cols=selected_columns or out.columns.tolist()
    if missing_strategy=="Drop rows with missing values": out=out.dropna(subset=cols)
    elif missing_strategy.startswith("Fill numeric"):
        method=missing_strategy.replace("Fill numeric with ","")
        for c in out.select_dtypes(include=np.number):
            if c in cols:
                v=out[c].median() if method=="median" else out[c].mean() if method=="mean" else 0
                if pd.notna(v): out[c]=out[c].fillna(v)
    elif missing_strategy=="Fill categorical with mode":
        for c in out.select_dtypes(include=["object","category","bool"]):
            if c in cols and not out[c].mode().empty: out[c]=out[c].fillna(out[c].mode().iloc[0])
    elif missing_strategy!="Keep as-is": raise ValueError("Unknown missing-value strategy.")
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
        elif missing_strategy == "Fill categorical with mode":
            eligible &= set(deduplicated.select_dtypes(include=["object", "category", "bool"]).columns)
        filled_cells = int(deduplicated[list(eligible)].isna().sum().sum()
                           - after[list(eligible)].isna().sum().sum()) if eligible else 0
    elif missing_strategy == "Drop rows with missing values":
        rows_removed_for_missing = max(0, len(deduplicated) - len(after))

    before_profile = profile_dataset(before)
    after_profile = profile_dataset(after)
    per_column = pd.DataFrame({
        "column": before.columns.astype(str),
        "missing_before": [int(before[c].isna().sum()) for c in before.columns],
        "missing_after": [int(after[c].isna().sum()) for c in after.columns],
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
