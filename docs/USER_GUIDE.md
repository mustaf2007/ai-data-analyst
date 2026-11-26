# User guide

## Start a session

1. Install dependencies with `pip install -r requirements-dev.txt`.
2. Run `streamlit run app.py`.
3. Upload a CSV/XLSX file from the sidebar, or choose **Load built-in sample**.
4. Choose one of the six workspaces: **Overview**, **Prepare**, **Analyze**, **Models**, **Reports** or **Assistant**.
5. Choose the relevant secondary tab inside each workspace. The sidebar also provides six appearance themes.

## Workspace map

- **Overview — Dashboard:** review dataset dimensions, missingness, duplicates, the heuristic quality score, automated observations and quick charts.
- **Prepare — Data quality:** select an intended analysis workflow, inspect readiness checks and column diagnostics. Read the warnings; a readiness score is not a statistical certification.
- **Prepare — Clean data:** clean a working copy. Review the before/after audit and download the cleaned dataset. The original upload can be restored with **Reset to original**.
- **Analyze — Explore / Relationships / Chart Studio:** inspect descriptive statistics, category frequencies, grouped summaries, distributions, date trends and numeric associations; create and export custom charts.
- **Analyze — Time series / Statistical tests / Geographic / Compare datasets:** use these only when the active data contains appropriate time, group, geographic or comparison fields. Review test assumptions and coverage limitations.
- **Analyze — Insights / Data story / Recommendations:** review deterministic findings and evidence, prepare audience-oriented narratives and see suggested next analytical steps.
- **Models — Predictive model / Anomalies:** run exploratory Random Forest baselines or anomaly screening when the data meets the stated requirements. Validate results independently before operational use.
- **Reports — Research report:** enter a research question, optional objectives, report title and significance threshold. Review the selected test, variables, sample size, p-value, effect size, charts, methodology and limitations. Download PDF, Markdown and Excel evidence outputs.
- **Reports — Specialist report:** prepare a technical, financial/investment, research/policy, monitoring/evaluation or market/business report. Financial metrics are calculated only when required fields are explicit and unambiguous; PDF and Excel outputs can include financial metrics and holdings attribution.
- **Reports — PDF & Excel export:** create a configurable general analysis report and export the active dataset.
- **Assistant — Ask the data:** ask supported questions answered through deterministic calculations on the active dataset.
- **Assistant — AI assistant:** select dataset tools for offline calculations, Ollama for generative AI without an API key (requires a local or reachable self-hosted model), or an optional OpenAI-compatible endpoint.

## Cleaning and result freshness

The app keeps the original upload and a separate active working dataset. Applying cleaning replaces the active dataset and clears model, anomaly, chart, report and assistant outputs that would otherwise refer to an earlier version. Resetting the original dataset also clears derived outputs.

## Research report workflow

1. Open **Reports → Research report**.
2. Write one focused research question that names the outcome and, where relevant, the comparison or explanatory factor.
3. Add objectives and a report title if useful.
4. Generate the report.
5. Review which variables and statistical method were selected. The current matching is heuristic, so correct interpretation requires subject-matter review.
6. Check the descriptive tables, test assumptions, warnings, effect size, recommendations and limitations.
7. Download the PDF for presentation and the Excel workbook for inspection and reproducibility.

The report is evidence-driven and calculated locally. It is not an automated literature review, does not verify external sources, and does not establish causal relationships.

## Interpretation and privacy

- Quality scores are heuristic and readiness checks are workflow-specific.
- Correlation is not causation. A p-value alone does not establish practical importance.
- The predictive and anomaly workflows are baselines, not production decision systems.
- The optional external AI provider receives questions and aggregate dataset context; do not send confidential data unless permitted.
- Public hosting is not a secure environment for confidential client records.
