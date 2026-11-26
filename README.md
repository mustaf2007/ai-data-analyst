# AI Data Analyst — Universal Data Intelligence

A local-first data analysis workspace built with Python, Streamlit and open-source libraries. Upload a CSV or XLSX file, inspect data quality, clean a working copy, explore distributions and relationships, train baseline predictive models, detect unusual observations, and export CSV/PDF outputs.

**Local-first by default.** Core data analysis uses deterministic Python/statistical/ML processing with no API key. The Smart AI Assistant can optionally connect to a locally hosted Ollama model or an explicitly configured OpenAI-compatible API for general-purpose questions and report drafting.

## Current capabilities

- **Upload & preview:** CSV and XLSX, file-size and dataset-shape safeguards, duplicate column-name normalization.
- **Overview:** dataset dimensions, missingness, duplicate-row count, heuristic quality score, automated observations and data preview.
- **Data quality & readiness:** column-level data types, non-null counts, missingness, unique counts and sample values, plus analysis-specific readiness checks for exploratory analysis, correlation, regression, classification and time-series workflows. Column diagnostics classify missingness severity and flag suspicious headers and numeric-looking text columns.
- **Cleaning:** remove duplicate rows; drop entirely empty columns; drop rows with missing values; impute numeric values with median, mean or zero; impute categorical values with mode; or use the combined recommended option (numeric median plus categorical mode). After applying changes, the app displays an automatic before/after report with row counts, duplicates removed, missing cells filled/remaining, a per-column missingness table, and heuristic quality-score change. The per-column report can be downloaded as CSV, and the cleaned active dataset is available as `cleaned_dataset.csv`. The download is shown directly in the Clean data tab after cleaning, and the current dataset remains downloadable even before cleaning. The original upload remains available for reset.
- **Exploration:** numeric descriptive statistics, categorical frequency tables, histograms, box plots, grouped aggregations and date trends with selectable aggregation.
- **Relationships:** Pearson, Spearman and Kendall correlation matrices plus strongest pairwise associations.
- **Predictive modeling:** local Random Forest regression/classification baseline with hold-out metrics and actual-vs-predicted comparison.
- **Anomaly detection:** local Isolation Forest with configurable expected anomaly share.
- **Exports:** active dataset CSV, Excel workbooks, PDF analysis reports and downloadable charts.\n- **Smart AI Assistant:** conversational general-purpose Q&A through Ollama or an opt-in OpenAI-compatible endpoint; dataset-only deterministic mode remains available without a model.\n- **Professional technical reports:** configurable report title, report type, author and purpose; executive summary, data quality, quantified findings, descriptive tables, correlations, recommendations, methodology and limitations; Markdown, PDF and Excel exports. Financial/investment reports additionally detect common portfolio fields and calculate total-portfolio and invested-sleeve returns separately, summarize holdings, and report fees without deducting them twice. Metrics are omitted when required fields are unavailable or ambiguous.\n- **Privacy controls:** model context uses dataset schema and aggregate summaries, not raw record rows. External API use is opt-in and clearly identified in the interface.
- **Engineering:** pytest suite, GitHub Actions CI, dependency files, Streamlit configuration and security notes.

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/mustaf2007/ai-data-analyst.git
cd ai-data-analyst
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Select **Load built-in sample** to try the application without supplying a file.

## Tests and checks

```bash
pytest
python -m compileall -q app.py src tests
```

The GitHub Actions workflow runs the test suite and compile check on Python 3.11 and 3.12 for pushes and pull requests to `main`.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create an app and select this repository, branch `main`, and entrypoint `app.py`.
4. Select Python 3.12 in advanced settings.
5. Deploy. No secrets or API keys are required for the core application. For general-purpose AI answers, optionally add `OPENAI_API_KEY` and `OPENAI_MODEL` under Advanced settings → Secrets. You may also set `OPENAI_BASE_URL` for a compatible endpoint. Alternatively, configure `OLLAMA_BASE_URL` and `OLLAMA_MODEL` to reach a separately hosted Ollama server.

Community Cloud is suitable for a public demo and lightweight use. It is not a substitute for a secured private deployment for confidential client datasets. Resource limits and platform policies may change; test realistic file sizes and workloads before client use.

## Data handling and limitations

- The app has no database and does not intentionally persist uploaded datasets. The optional external AI mode transmits the user question, recent conversation and compact dataset schema/aggregate summaries to the configured provider; raw record rows are not included in that context. Do not enable external AI for confidential datasets unless permitted by your data-governance rules. Data is held in Streamlit session state and in process memory while the session is active. Hosting providers may still process application traffic and runtime data; consult the host's current privacy terms.
- Do not upload personal, confidential, regulated or commercially sensitive data to a public demo.
- The 20 MB, 250,000-row and 150-column limits are application safeguards, not guarantees of successful processing under every hosting resource limit.
- The overview quality score is a **heuristic**: 70% completeness and 30% row uniqueness. The analysis-readiness percentage is the share of checks passed for the selected workflow; it is not a certification or a guarantee of suitability. Review warnings and apply domain-specific validation.
- Correlation does not imply causation.
- Predictive modeling is a baseline workflow. The random hold-out split is not appropriate for every time-series, panel, grouped or leakage-prone dataset. It does not perform automated feature leakage detection, temporal validation, hyperparameter optimization or model governance.
- Anomaly detection identifies unusual patterns, not confirmed errors, fraud or misconduct.
- PDF export currently contains summary tables and observations, not chart images.
- Excel support is `.xlsx`; legacy `.xls` is not supported.

## Repository layout

```text
app.py
src/
  analytics.py
  modeling.py
  reporting.py
tests/
  test_analytics.py
  test_modeling.py
  test_reporting.py
sample_data/
  sample_business_data.csv
.streamlit/config.toml
.github/workflows/ci.yml
requirements.txt
requirements-dev.txt
pytest.ini
README.md
SECURITY.md
CONTRIBUTING.md
CHANGELOG.md
```

## License

MIT. See [LICENSE](LICENSE).
