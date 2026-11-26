# Architecture

The application is a local-first Streamlit workspace. Uploaded CSV/XLSX files are parsed into pandas DataFrames; the original and active working copies are held in Streamlit session state.

## Interface

`app.py` provides six top-level workspaces:

- Overview
- Prepare
- Analyze
- Models
- Reports
- Assistant

Related workflows are nested under these workspaces. When a dataset is replaced or cleaned, derived results are invalidated to avoid showing outputs calculated from stale data.

## Modules

- `src/analytics.py`: file validation, profiling, cleaning, descriptive summaries, correlation, time-series, geographic summaries, statistical tests and insight generation.
- `src/modeling.py`: preprocessing pipelines, Random Forest regression/classification and Isolation Forest.
- `src/reporting.py`: configurable PDF report generation and assistant-response PDF rendering.
- `src/research_reporting.py`: research-question-guided variable matching, descriptive evidence, cautious statistical test selection, chart construction and PDF/Excel/Markdown research exports.
- `src/financial_reporting.py`: conservative financial performance metrics when required portfolio fields are explicit and unambiguous.
- `src/assistant.py`: deterministic analysis suggestions and dataset question answering.
- `src/smart_assistant.py`: dataset context preparation and optional Ollama/OpenAI-compatible assistant calls.

## Data and privacy

Core analytics and research-report calculations run locally in the application process. No database or durable dataset storage is included. The optional external AI mode transmits a question, recent conversation and aggregate dataset context to the configured provider; raw dataset rows are not included in that context.

This is not a multi-tenant enterprise service. Authentication, authorisation, tenant isolation, audit logging, durable storage, model registry and monitoring are not included.
