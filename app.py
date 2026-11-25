from pathlib import Path
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/"src"))
from analytics import (DataInputError,load_dataset,profile_dataset,column_profile,clean_dataset,
    numeric_summary,categorical_summary,correlation_matrix,grouped_summary,infer_datetime_columns,
    generate_insights,safe_csv_bytes)
from modeling import train_predictive_model,detect_anomalies
from reporting import build_pdf_report

st.set_page_config(page_title="AI Data Analyst",page_icon="📊",layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Universal Data Intelligence · Local-first analytics · Open-source Python · No paid APIs")
if "raw" not in st.session_state: st.session_state.raw=None
if "active" not in st.session_state: st.session_state.active=None
if "source" not in st.session_state: st.session_state.source=""

with st.sidebar:
    st.header("Workspace")
    upload=st.file_uploader("Upload dataset",type=["csv","xlsx"])
    if upload:
        try:
            df=load_dataset(upload.name,upload.getvalue())
            st.session_state.raw=df; st.session_state.active=df.copy(); st.session_state.source=upload.name
        except DataInputError as e: st.error(str(e))
    if st.button("Load built-in sample",use_container_width=True):
        path=ROOT/"sample_data"/"sample_business_data.csv"
        df=pd.read_csv(path); st.session_state.raw=df; st.session_state.active=df.copy(); st.session_state.source=path.name
    if st.session_state.raw is not None and st.button("Reset to original"):
        st.session_state.active=st.session_state.raw.copy()
    st.caption("Files are processed in this app session. Avoid uploading sensitive data to a public demo.")

if st.session_state.active is None:
    st.info("Upload a CSV/XLSX dataset or load the built-in sample to begin.")
    st.markdown("**Explore · Clean · Visualize · Model · Report**")
    st.stop()

df=st.session_state.active
p=profile_dataset(df)
tabs=st.tabs(["Overview","Data quality","Clean data","Explore","Relationships","Predictive modeling","Anomalies","Export & report"])
with tabs[0]:
    st.subheader(st.session_state.source)
    a=st.columns(5)
    for col,label,value in zip(a,["Rows","Columns","Missing cells","Duplicate rows","Quality score"],
        [f"{p['rows']:,}",p["columns"],f"{p['missing_cells']:,}",p["duplicate_rows"],f"{p['quality_score']:.1f}/100"]): col.metric(label,value)
    st.markdown("#### Automated observations")
    for insight in generate_insights(df): st.write("• "+insight)
    st.dataframe(df.head(100),use_container_width=True)
with tabs[1]:
    st.caption("The quality score is a heuristic, not a certification or full domain-specific audit.")
    st.dataframe(column_profile(df),use_container_width=True)
    st.write("Missing values by column")
    if p["missing_by_column"].empty: st.success("No missing values.")
    else: st.bar_chart(p["missing_by_column"])
with tabs[2]:
    with st.form("clean"):
        dedupe=st.checkbox("Remove duplicate rows",True)
        strategy=st.selectbox("Missing-value strategy",["Keep as-is","Drop rows with missing values","Fill numeric with median","Fill numeric with mean","Fill numeric with zero","Fill categorical with mode"])
        selected=st.multiselect("Apply to columns (empty = all eligible)",df.columns.tolist())
        submitted=st.form_submit_button("Apply cleaning")
    if submitted:
        st.session_state.active=clean_dataset(df,drop_duplicates=dedupe,missing_strategy=strategy,selected_columns=selected or None)
        st.success("Cleaning applied to working copy."); st.rerun()
with tabs[3]:
    numeric=df.select_dtypes(include=np.number).columns.tolist()
    categorical=df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    st.markdown("#### Numeric summary"); st.dataframe(numeric_summary(df),use_container_width=True)
    if categorical:
        c=st.selectbox("Categorical variable",categorical)
        freq=categorical_summary(df)[c]; st.dataframe(freq,use_container_width=True)
        st.plotly_chart(px.bar(freq,x="value",y="count"),use_container_width=True)
    if numeric:
        n=st.selectbox("Distribution variable",numeric)
        st.plotly_chart(px.histogram(df,x=n,marginal="box"),use_container_width=True)
    if categorical and numeric:
        x,y=st.columns(2); group=x.selectbox("Group by",categorical); measure=y.selectbox("Measure",numeric)
        agg=st.selectbox("Aggregation",["mean","median","sum","count","min","max"])
        result=grouped_summary(df,group,measure,agg); st.dataframe(result,use_container_width=True)
        st.plotly_chart(px.bar(result.head(30),x=group,y=measure),use_container_width=True)
    dates=infer_datetime_columns(df)
    if dates and numeric:
        date=st.selectbox("Date-like column",dates); measure=st.selectbox("Trend measure",numeric)
        trend=df[[date,measure]].copy(); trend[date]=pd.to_datetime(trend[date],errors="coerce")
        trend=trend.dropna(subset=[date]).set_index(date)[measure].resample("MS").sum().reset_index()
        st.plotly_chart(px.line(trend,x=date,y=measure,markers=True),use_container_width=True)
with tabs[4]:
    method=st.selectbox("Correlation method",["pearson","spearman","kendall"])
    corr=correlation_matrix(df,method)
    if corr.empty: st.info("At least two numeric columns are required.")
    else: st.plotly_chart(px.imshow(corr,text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="RdBu_r"),use_container_width=True)
with tabs[5]:
    target=st.selectbox("Target variable",df.columns.tolist()); task=st.selectbox("Task",["Auto-detect","Regression","Classification"])
    test=st.slider("Test set share",.1,.4,.2,.05)
    if st.button("Train baseline model"):
        try: st.session_state.model_result=train_predictive_model(df,target,task,test)
        except Exception as e: st.error(str(e)); st.session_state.model_result=None
    result=st.session_state.get("model_result")
    if result:
        st.write(result["task"],"· Random Forest baseline")
        for c,(name,value) in zip(st.columns(len(result["metrics"])),result["metrics"].items()): c.metric(name,f"{value:.4f}")
        st.dataframe(result["comparison"],use_container_width=True)
        st.warning("Baseline only. Random splits may be inappropriate for time-series, grouped or leakage-prone data.")
with tabs[6]:
    share=st.slider("Expected anomaly share",.01,.20,.05,.01)
    if st.button("Detect anomalies"):
        try: st.session_state.anomalies=detect_anomalies(df,share)
        except Exception as e: st.error(str(e))
    if st.session_state.get("anomalies") is not None:
        out=st.session_state.anomalies
        st.metric("Potential anomalies",int((out.anomaly_flag=="Anomaly").sum()))
        st.dataframe(out.sort_values("anomaly_score").head(200),use_container_width=True)
        st.download_button("Download anomaly CSV",safe_csv_bytes(out),"anomaly_results.csv","text/csv")
    st.caption("Anomaly flags are screening signals, not proof of errors or fraud.")
with tabs[7]:
    st.download_button("Download active dataset (CSV)",safe_csv_bytes(df),"analysis_dataset.csv","text/csv")
    pdf=build_pdf_report(st.session_state.source,df,p,numeric_summary(df),generate_insights(df))
    st.download_button("Download PDF summary",pdf,"data_analysis_report.pdf","application/pdf")
    st.caption("PDF includes dataset metrics, observations and numeric summary; it does not embed charts.")
st.divider()
st.caption("AI Data Analyst v1.0 · Local session processing · Correlation is not causation.")
