# Deployment checklist

- [ ] Review all committed files; exclude secrets and real client data.
- [ ] Run `pytest` and `python -m compileall -q app.py src tests`.
- [ ] Test representative CSV and XLSX datasets.
- [ ] Connect this GitHub repository to Streamlit Community Cloud.
- [ ] Select branch `main`, entrypoint `app.py`, and Python 3.12.
- [ ] Deploy and test sample data, navigation, charts and downloads.

Do not process confidential, personal, regulated or commercially sensitive data on a public demo. Authentication, authorization, tenant isolation, audit logging and retention controls are not implemented.
