# User guide

1. Install dependencies with `pip install -r requirements-dev.txt`.
2. Start with `streamlit run app.py`.
3. Upload CSV/XLSX or load the built-in sample.
4. Review the overview and column-level data quality profile.
5. Review the general analysis-readiness checklist in **Data quality**. It reports passed checks, warnings, blockers and a transparent checklist percentage; it is not a universal certification.
6. Apply optional cleaning to a working copy. After the operation, the Clean data tab automatically displays a result notification and before/after report: rows retained/removed, duplicates removed, rows removed for missing values, missing cells filled/remaining, per-column missingness, and the heuristic quality score change. Download the per-column cleaning report and the cleaned dataset (`cleaned_dataset.csv`) as CSV. If the selected operation would remove every row, it is rejected and the active dataset is preserved. The original uploaded dataset remains available through **Reset to original**.
7. Explore descriptive statistics, category frequencies, distributions, grouped summaries and date trends. Date trends let you select the aggregation method.
8. Inspect numeric correlations using Pearson, Spearman or Kendall.
9. Train a baseline regression/classification model or run Isolation Forest.
10. Export the active dataset as CSV, anomaly results as CSV, or create a PDF summary.

The uploaded dataset is loaded only when the file content changes, so normal Streamlit reruns should not undo cleaning. Replacing or cleaning the active dataset clears model and anomaly results so that stale outputs are not presented as current.

Correlation is not causation. Quality scores are heuristic. Model results require validation, especially for time-series, grouped or leakage-prone data. Anomaly flags are not proof of error or fraud.
