# Contributing

- Keep processing local and deterministic; do not add paid APIs or external telemetry.
- Add pytest coverage for behavior changes.
- Run `pytest` and `python -m compileall -q app.py src tests` before pull requests.
- Document limitations of statistical and ML outputs.
- Never commit client data, secrets or credentials.
