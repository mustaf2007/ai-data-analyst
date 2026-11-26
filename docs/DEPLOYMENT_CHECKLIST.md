# Deployment checklist

- [ ] Review committed files; exclude secrets and real client data.
- [ ] Run `pytest` and `python -m compileall -q app.py src tests`.
- [ ] Confirm GitHub Actions passes on Python 3.11 and 3.12.
- [ ] Verify the six top-level workspaces and their secondary tabs render correctly on desktop and mobile.
- [ ] Test CSV and XLSX uploads, duplicate headers, missing values, malformed files and upload-size limits.
- [ ] Verify cleaning persists after navigation and reruns; verify **Reset to original** restores the original upload.
- [ ] Verify cleaning and dataset replacement clear stale model, anomaly, chart, report and assistant outputs.
- [ ] Test numeric summaries, categorical frequencies, charts, grouped summaries, date trends and Pearson/Spearman/Kendall correlations.
- [ ] Test regression and classification on suitable datasets; test anomaly detection with numeric data.
- [ ] Test statistical tests with valid and sparse group sizes; confirm warnings appear where assumptions or expected counts are problematic.
- [ ] Test the Research Report Builder with numeric, categorical-only, missing-value and grouped datasets. Review variable selection, test selection, p-values, effect sizes, methodology and limitations.
- [ ] Download and reopen research PDF, Markdown and Excel evidence workbook; verify source data and chart data sheets.
- [ ] Test specialist report types, including a portfolio dataset with explicit starting/ending values, cost basis, market value and fees. Confirm financial metrics and holdings attribution appear in PDF/Excel only when supported.
- [ ] Test general PDF/Excel exports, chart downloads, active dataset CSV and anomaly CSV.
- [ ] Test deterministic Ask the Data; test Ollama only when a local or reachable server is configured.
- [ ] Connect the repository to Streamlit Community Cloud, select branch `main`, entrypoint `app.py`, and Python 3.12.
- [ ] Deploy and repeat the end-to-end workflow with a non-sensitive test dataset.

Do not process confidential, personal, regulated or commercially sensitive data on a public demo. Authentication, authorisation, tenant isolation, audit logging and retention controls are not implemented. A passing unit-test suite does not replace live end-to-end testing.
