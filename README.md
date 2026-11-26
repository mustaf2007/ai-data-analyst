# AI Data Analyst — Universal Data Intelligence

A local-first analytics workspace for CSV and Excel datasets. Prepare data, explore patterns, test relationships, build baseline models, and produce evidence-based reports from one streamlined interface.

**Core analysis, public open-data connections and the browser-based AI assistant do not require an API key.** The Smart AI Assistant can run open-source Qwen models directly in the visitor’s browser using WebLLM and WebGPU. Optional hosted providers and Ollama remain available as alternatives.

## Workspace guide

The interface is organised into six workspaces:

| Workspace | Main tools |
|---|---|
| **Overview** | Dataset health, key findings, trend and distribution snapshots |
| **Prepare** | Data-quality checks, analysis readiness, cleaning and cleaning audit |
| **Analyze** | Descriptive exploration, correlations, Chart Studio, time series, statistical tests, geographic summaries, dataset comparison and automated insights |
| **Models** | Baseline Random Forest regression/classification and Isolation Forest anomaly screening |
| **Reports** | Research Report Builder, specialist technical/financial reports, and configurable PDF/Excel exports |
| **Assistant** | Deterministic Ask the Data and optional generative Smart AI Assistant |

Related tools are grouped into secondary tabs. Specialist report types remain available alongside the research report and general-purpose exports.

## Main capabilities

- **Safe data intake:** CSV/XLSX upload, duplicate-header normalization, and safeguards of 20 MB, 250,000 rows and 150 columns. The sidebar also connects directly to curated World Bank Open Data and Our World in Data datasets without API keys, then loads the retrieved table into the same analysis workspace.
- **Data quality and cleaning:** missingness and duplicate checks, column diagnostics, analysis-readiness checklists, controlled cleaning options, before/after audit and cleaned-data export. The original upload remains available for reset.
- **Exploration and visualisation:** descriptive statistics, category frequencies, histograms, box plots, grouped summaries, date trends, correlation matrices and custom charts with HTML/PNG/SVG export where supported.
- **Statistical analysis:** one-sample and Welch t-tests, one-way ANOVA, chi-square tests, effect sizes, confidence intervals and assumption/sparse-count warnings where applicable.
- **Time-series and geographic analysis:** period aggregation and baseline forecasts, plus geographic summaries when usable geographic fields are present.
- **Dataset comparison:** compare a second uploaded dataset with the active dataset, including schema and indicator/distribution differences.
- **Baseline modelling:** local Random Forest regression/classification and Isolation Forest screening. These are exploratory baselines, not production model governance.
- **Research Report Builder:** enter a research question and objectives to produce descriptive evidence, question-guided statistical screening, effect sizes, automatically generated charts, methodology, limitations and prioritised recommendations. Download a formatted PDF, Markdown report and multi-sheet Excel evidence workbook with source data and chart-ready tables. The workflow uses local calculations and does not require an AI API key.
- **Specialist reports:** generate technical, financial/investment, research/policy, monitoring/evaluation or market/business reports. When explicit portfolio fields are available, financial metrics and holdings attribution are included in PDF and Excel outputs.
- **Financial report support:** when explicit, unambiguous portfolio fields are present, financial reporting can distinguish total-portfolio and invested-sleeve performance and report fees without deducting them twice.
- **Assistant options:** Ask the Data provides deterministic answers from supported dataset calculations. Browser AI runs Qwen 3.5 open models locally in the browser with WebLLM/WebGPU and requires no API key or inference server. The first use downloads model/runtime files and caches them in browser storage where supported. Groq, OpenRouter, Gemini, Ollama and OpenAI-compatible endpoints remain optional alternatives.

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/mustaf2007/ai-data-analyst.git
cd ai-data-analyst
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Choose **Load built-in sample** from the sidebar to explore the app without uploading a file, or expand **Connect public open data** to retrieve a public World Bank or Our World in Data dataset without credentials.

## Public open-data connections\n\nUse **Connect public open data** in the sidebar to load curated indicators and datasets from the World Bank API or Our World in Data CSV downloads. Current catalogue examples include GDP, population, inflation, unemployment, agriculture value added, CO₂ emissions, internet use, electricity access and life expectancy. Connections are made on demand; the app records the provider, retrieval date and source link in the Overview workspace. No API key or account is required for these catalogue entries.\n\nThe public source must be reachable from the app host. Availability, update frequency, licensing and attribution are controlled by each provider; public endpoints may change or impose fair-use limits. Downloads are capped at 35 MB. The connector deliberately uses a reviewed catalogue rather than accepting arbitrary URLs.\n\n## Browser-based AI assistant

The Smart AI Assistant defaults to **Browser AI (open-source · no API key)**. It uses WebLLM with the Qwen 3.5 0.8B model by default, with a 2B option for more capable devices. Model inference runs inside the visitor’s browser through WebGPU; model/runtime files are downloaded from public sources on first use and cached by the browser where supported. Use a recent Chrome or Edge browser on a device with WebGPU enabled. First-time downloads can be large and performance depends on available GPU memory. If WebGPU is unavailable in the browser or embedded frame, use one of the optional providers below.

## Optional hosted/local AI setup

The analytical features and report builder work without an API key. To use generative responses with Ollama:

```bash
ollama pull llama3.1:8b
```

Run Ollama on the same computer as the local Streamlit app; the default endpoint is `http://localhost:11434`. For a hosted Streamlit deployment, `localhost` refers to the cloud container, not your computer. Configure `OLLAMA_BASE_URL` to a reachable Ollama server and optionally set `OLLAMA_MODEL`. This is a server URL, not an API key; hosting a reachable model may have infrastructure costs. An OpenAI-compatible API is optional and requires provider credentials.

## Tests

```bash
pytest
python -m compileall -q app.py src tests
```

GitHub Actions runs the test suite and compile check on Python 3.11 and 3.12.

## Data handling and limitations

- The app does not intentionally persist uploaded datasets in a database. Data is held in Streamlit session state and process memory while the session is active.
- Browser AI performs inference on the user’s device; prompts and included dataset summaries are not sent to an inference API. The browser must download the WebLLM runtime and model weights from public CDNs/model hosting on first use, and WebGPU-capable hardware/browser support is required. The optional hosted AI modes send the question, recent conversation and compact schema/aggregate summaries to the configured provider; raw record rows are not included in that context. Do not use hosted modes with confidential data unless your governance rules permit the transfer.
- Do not upload personal, confidential, regulated or commercially sensitive information to a public demo.
- Upload limits are application safeguards, not guarantees that every permitted file will fit the hosting platform's memory or execution limits.
- The dataset quality score is a heuristic, not a certification. Analysis-readiness scores are workflow-specific checklists, not guarantees of statistical suitability.
- Research report variable selection uses question/column-name matching and data structure. Review the selected variables and statistical method before relying on the report. Sampling design, assumptions, representativeness, measurement validity and confounding are not fully verified automatically.
- Correlation does not imply causation. Statistical significance is not proof of practical importance or causal effect.
- Predictive modelling uses baseline hold-out evaluation; it does not automatically address temporal validation, leakage, hyperparameter optimisation or model governance.
- Anomaly flags identify unusual patterns, not confirmed errors, fraud or misconduct.
- PDF chart images depend on static Plotly rendering being available; chart data is included in the Excel evidence workbook.
- Excel export uses `.xlsx`; legacy `.xls` is not supported.
- The public demo is not a multi-tenant enterprise service. Authentication, authorisation, tenant isolation, audit logging and durable retention controls are not included.

## Repository layout

```text
app.py
src/
  analytics.py
  assistant.py
  financial_reporting.py
  modeling.py
  reporting.py
  research_reporting.py
  smart_assistant.py
tests/
docs/
sample_data/
.streamlit/
.github/workflows/
requirements.txt
requirements-dev.txt
```

## License

MIT. See [LICENSE](LICENSE).

## Hybrid dataset chatbot

The **Ask your data** workspace offers two modes:

- **Hybrid AI (choose provider):** a configured hosted model translates a natural-language question into a constrained JSON analysis plan. The app validates the plan and performs the calculations with pandas. Supported operations include schema inspection, descriptive summaries, aggregations, grouped rankings, filtering, correlations, chronological trends and basic data-quality checks.
- **Model-free:** the existing deterministic assistant remains available without an AI provider or API key.

The hybrid planner receives the question, recent chat context and dataset schema metadata (column names, data types, missing counts and distinct counts). It does not receive raw dataset rows. The application never executes model-generated Python, SQL or shell commands. Column names, operations, aggregation functions, filter operators and limits are validated before execution. A table of calculated evidence is returned with each supported answer.

### Configure a provider

Choose **Free hosted AI (automatic fallback)** and enter a Groq, OpenRouter or Google Gemini API key in the app, or configure `GROQ_API_KEY`, `OPENROUTER_API_KEY` or `GEMINI_API_KEY` as Streamlit secrets. Alternatively choose **OpenAI-compatible API (opt-in)** and configure `OPENAI_API_KEY`, with optional `OPENAI_BASE_URL` and `OPENAI_MODEL`.

Hosted providers receive the user's question, recent conversation and schema metadata. Review the selected provider's privacy terms and usage limits before sending questions. API charges or free-tier quotas may apply. The model-free assistant remains available if no provider is configured or a provider is unavailable.

The chatbot can answer a broad range of questions **that can be supported by the loaded table**; it is not a guarantee that every possible question is answerable. Ambiguous, unsupported or unavailable information should be clarified rather than fabricated. Recent turns are retained within the active Streamlit session to help resolve follow-up questions; loading a different dataset clears the chat.
