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
