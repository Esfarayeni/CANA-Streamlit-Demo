# Canalization Atlas
An interactive Streamlit application for exploring Boolean network models from Cell Collective using the CANA Python library.

## Cell Collective metadata

`model_metadata.json` is a version-controlled catalogue of the public Cell Collective metadata and linked references for every bundled Cell Collective model. The dashboard reads it first, avoiding a network request during normal use; the public API remains a fallback for a missing record.

Refresh the catalogue deliberately after model updates with:

```bash
python scripts/refresh_model_metadata.py
```
