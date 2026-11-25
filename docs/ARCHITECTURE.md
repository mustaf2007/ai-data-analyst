# Architecture

The Streamlit application runs in one Python process. Uploaded CSV/XLSX bytes are parsed into pandas DataFrames. The raw and active working datasets are held in Streamlit session state; cleaning creates a working copy.

- `app.py`: interface, navigation, charts and downloads.
- `src/analytics.py`: validation, profiling, cleaning, summaries, correlations and CSV serialization.
- `src/modeling.py`: preprocessing pipelines, Random Forest models and Isolation Forest.
- `src/reporting.py`: PDF report generation.

No paid AI endpoint, external connector or database is used. This is not a multi-tenant enterprise service: authentication, authorization, audit logging, durable storage, model registry and monitoring are not included.
