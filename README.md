# Document Processing & Verification System

This repository contains a production-ready document processing pipeline that uses OCR, vision models, and LLMs to extract and validate fields from documents. The project includes multi-engine OCR fallback, Donut and spaCy integrations, audit trails, and a FastAPI-based API.

## Quickstart
1. Clone the repo
2. Create a virtual env and install dependencies

```powershell
cd E:\n
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Start local server

```powershell
python -m uvicorn app.main:app --reload
```

4. Health check

```powershell
curl http://localhost:8000/health
```

## What to include in GitHub
- `app/` - source code (API & services)
- `doc_archive/` - docs & internal notes
- `scripts/` - helper scripts
- `requirements.txt` - dependencies
- `setup.py` - packaging metadata
- `.github/workflows` - CI configuration

## What NOT to include
- `venv/` - virtual environment (add to .gitignore)
- `cache/` - embedding caches and intermediate files
- `documents/` - uploaded documents and demo PDFs
- `vector_store/` - generated embeddings & vector DB (store externally or use LFS)

Use Git LFS for large artifacts (models, ONNX, large datasets).

## Repo Structure suggestion
```
/ (root)
├── app/                # application source
├── doc_archive/        # docs and architecture
├── scripts/            # utilities
├── requirements.txt
├── .github/workflows/  # CI
├── README.md
└── LICENSE
```

## Contributing
Please see `CONTRIBUTING.md` for contribution guidelines.

---
If you want, I can add a minimal `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`, and optionally create a `tests/` skeleton and add a CodeQL analysis workflow or a test matrix.
