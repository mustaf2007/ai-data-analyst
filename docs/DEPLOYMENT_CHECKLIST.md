# Deployment checklist

- [ ] Review all committed files; exclude secrets and real client data.
- [ ] Run `pytest` and `python -m compileall -q app.py src tests`.
- [ ] Confirm GitHub Actions passes on Python 3.11 and 3.12.
- [ ] Test CSV and XLSX uploads, including duplicate headers, missing values and malformed files.
- [ ] Verify cleaning persists after navigation and reruns; verify **Reset to original** restores the original upload.
- [ ] Verify changing datasets clears previous model and anomaly results.
- [ ] Test numeric summaries, categorical frequencies, charts, grouping, date trends and all correlation methods.
- [ ] Test regression and classification on suitable datasets; test anomaly detection with numeric data.
- [ ] Download and reopen active CSV, anomaly CSV and PDF outputs.
- [ ] Connect this GitHub repository to Streamlit Community Cloud.
- [ ] Select branch `main`, entrypoint `app.py`, and Python 3.12.
- [ ] Deploy and repeat the end-to-end workflow using a non-sensitive test dataset.

Do not process confidential, personal, regulated or commercially sensitive data on a public demo. Authentication, authorization, tenant isolation, audit logging and retention controls are not implemented. A passing unit-test suite does not replace live end-to-end testing.
