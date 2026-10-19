# Streamlit UI starter for CMS Medicare Claims GenAI

Copy `streamlit_app.py` and `requirements-streamlit.txt` into the root of your existing repository. This UI calls your existing FastAPI `/ask` endpoint; it does not replace or modify your retrieval/LLM pipeline.

## Local launch (two terminal windows)

From the project root, activate the existing virtual environment and install the UI dependencies:

```bash
python -m pip install -r requirements-streamlit.txt
```

Terminal 1: run the existing API (after preparing your Chroma database and setting OPENAI_API_KEY in your local `.env` as described in the main README):

```bash
python -m uvicorn src.api.main:app --reload
```

Terminal 2:

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL displayed by Streamlit (typically http://localhost:8501).

## Deployment caveat

This is a local-first starter. Deploying only the Streamlit UI to Streamlit Community Cloud will NOT make the RAG model work: `localhost:8000` on Streamlit Cloud is not your laptop's API. You must either deploy the FastAPI backend to an accessible HTTPS service and set the Streamlit secret `RAG_API_URL` to that service URL, or integrate the backend directly into Streamlit and initialize Chroma/model files on the host. For a public backend, implement authentication and rate limits first to avoid unlimited API spending. Never put OPENAI_API_KEY in source control.

## Security

Keep `.env` and `.streamlit/secrets.toml` out of Git. Never expose API keys through the browser. If any key was committed previously, revoke and replace it.
