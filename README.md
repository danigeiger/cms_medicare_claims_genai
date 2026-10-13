# CMS Medicare Claims GenAI — RAG Codebook Assistant

A Retrieval-Augmented Generation (RAG) application that helps users identify and understand variables in the **CMS Medicare Fee-for-Service (FFS) Claims Codebook** using natural-language questions.

The application combines exact vector retrieval over Chroma-stored embeddings, transformer-based reranking, reciprocal rank fusion (RRF), OpenAI language models, and a FastAPI interface to retrieve relevant CMS variable definitions and generate structured answers.

## Project Overview

CMS Medicare claims datasets contain hundreds of variables with specialized names, descriptions, and coding conventions.

Finding the correct variable can require manually searching lengthy technical documentation.

This project addresses that problem by allowing users to ask questions such as:

> What variable identifies the patient's race?

The system retrieves relevant definitions from the CMS codebook and returns a structured response identifying the most appropriate variable.

**Example response:**

```json
{
  "variable": "BENE_RACE_CD",
  "label": "Beneficiary Race Code",
  "explanation": "Identifies the beneficiary's race code."
}
```

## Architecture

The system consists of two primary workflows: document ingestion and question answering.

### 1. Document Ingestion

```text
CMS Medicare FFS Codebook (PDF)
             |
             v
     parse_codebook.py
             |
             v
    Structured JSON (302 variables)
             |
             v
    build_vector_store.py
             |
             v
  SentenceTransformer Embeddings
             |
             v
    Persistent Chroma Database
```

The CMS codebook is parsed into structured variable records. Each record is converted into a vector embedding and stored in Chroma for semantic retrieval.

### 2. RAG Question-Answering Pipeline

```text
        User Question
             |
             v
        FastAPI /ask
             |
             v
          rag.py
             |
             v
      MiniLM query embedding
             |
             v
     Chroma persistent store
     + exact NumPy search
        Top 20 candidates
             |
             v
       CrossEncoder reranks
       same 20 candidates
             |
             v
      Reciprocal Rank Fusion
       (exact + reranked)
             |
             v
         Top 5 documents
             |
             v
        GPT-4.1-mini
             |
             v
       Structured CMSAnswer
             |
             v
        FastAPI JSON response
```

### Retrieval, Reranking, and Fusion

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional embeddings for CMS variable documents and user questions.

The current main RAG pipeline calls `retrieve_exact()` to compare the query embedding against all **302 stored vectors** using NumPy squared Euclidean distance. **ChromaDB remains the persistent vector/document store**; its indexed `retrieve()` search is retained for experiments and comparison. Exact retrieval supplies the top **20** candidates.

**Reranking model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` independently scores those same 20 candidates against the question. **Reciprocal Rank Fusion (RRF)** combines the exact-search and CrossEncoder rankings; the top **five** fused candidates are provided as context to the LLM.

This design arose from experiments investigating intermittent omissions in Chroma indexed retrieval and cases where reranking excluded the expected variable from the final five. The underlying cause of the intermittent indexed-search behavior is not established.

### Answer Generation

The five highest-ranked CMS variable definitions are supplied as context to OpenAI's `gpt-4.1-mini` model.

The application uses Pydantic structured outputs to return:

- `variable`: CMS variable identifier, or `null` if no appropriate match is established.
- `label`: Human-readable variable label, or `null`.
- `explanation`: Explanation of the selected variable.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| Document ingestion | pypdf |
| Embeddings | SentenceTransformers |
| Vector database | ChromaDB |
| Reranking | Hugging Face CrossEncoder |
| LLM | OpenAI GPT-4.1-mini |
| Structured outputs | Pydantic |
| API framework | FastAPI |
| ASGI server | Uvicorn |
| Automated testing | pytest, HTTPX |

## Project Structure

```text
cms_medicare_claims_genai/
|
├── docs/
│   ├── cms_codebook.pdf
│   ├── retrieval_experiments.md
│   └── retrieval_experiments_integrated.md
|
├── knowledge_base/
│   └── processed/
│       └── cms_codebook.json
|
├── src/
│   ├── ingestion/
│   │   └── parse_codebook.py
│   |
│   ├── vector_store/
│   │   └── build_vector_store.py
│   |
│   ├── retrieval/
│   │   ├── retrieval.py
│   │   ├── reranker.py
│   │   ├── evaluate_retrieval.py
│   │   ├── evaluate_candidate_recall.py
│   │   └── evaluate_rrf.py
│   |
│   ├── rag/
│   │   ├── rag.py
│   │   ├── evaluate_exact_rag.py
│   │   └── test_rrf_answer.py
│   |
│   └── api/
│       └── main.py
|
├── tests/
│   ├── test_api.py
│   └── test_retrieval_regression.py
|
├── cms_codebook_vectors/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The vector database, generated artifacts, and environment secrets are maintained locally and excluded from version control where appropriate.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/danigeiger/cms_medicare_claims_genai.git
cd cms_medicare_claims_genai
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the OpenAI API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit the `.env` file to GitHub.

### 5. Prepare the vector database

The application requires the locally generated CMS codebook Chroma database.

After placing the CMS codebook PDF in the expected location, run the ingestion and vector-store scripts:

```bash
python -m src.ingestion.parse_codebook
python -m src.vector_store.build_vector_store
```

These scripts prepare the structured CMS variables and vector embeddings used during retrieval.

## Running the Application

Start the FastAPI development server:

```bash
python -m uvicorn src.api.main:app --reload
```

The application will be available at:

`http://127.0.0.1:8000`

### Interactive API Documentation

FastAPI automatically generates interactive Swagger documentation:

`http://127.0.0.1:8000/docs`

### API Endpoints

**GET `/health`**

Checks API availability.

Example response:

```json
{
  "status": "healthy"
}
```

**POST `/ask`**

Accepts a natural-language question about CMS claims variables.

Example request:

```json
{
  "query": "What is the patient's race?"
}
```

Example response:

```json
{
  "variable": "BENE_RACE_CD",
  "label": "Beneficiary Race Code",
  "explanation": "Identifies the beneficiary's race code."
}
```

## Evaluation Results

All reported scores below come from **20-question development evaluations**, not a held-out test set. The questions and pipeline were refined during experimentation; these results do not establish accuracy on unseen queries. Retrieval top-k coverage and final LLM answer accuracy are different metrics.

### Historical Retrieval Experiments

| Configuration | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Initial Chroma baseline | 60% | 80% | 95% |
| Label-emphasized embeddings | 60% | 90% | 100% |
| Early CrossEncoder experiment | 80% | 100% | 100% |

These figures are preserved from earlier configurations; they are **not** measurements of the current exact-search + RRF pipeline.

### Updated Candidate Coverage

On the revised 20-question development benchmark:

| Candidate ranking | Expected variable in top five |
|---|---:|
| CrossEncoder reranking | 19/20 (95%) |
| Reciprocal Rank Fusion | **20/20 (100%)** |

### End-to-End RAG Answers

| Recorded evaluation | Correct selected variables |
|---|---:|
| Earlier pipeline | 19/20 (95%) |
| Current exact NumPy + CrossEncoder + RRF + GPT-4.1-mini | **20/20 (100%)** |

The latest run (`python -m src.rag.evaluate_exact_rag`) completed all **20 API requests**, with **20 correct selected identifiers** and **zero errors**. The development benchmark includes a clarified service-location question, so this before-and-after result is not an isolated causal estimate of RRF's effect.

### Experiment Documentation

- [Historical retrieval experiments](docs/retrieval_experiments.md) — original baseline, label emphasis, reranking, and RAG/RRF experiments, preserved in their historical context.
- [Integrated retrieval experiments](docs/retrieval_experiments_integrated.md) — detailed RRF, exact-search, reliability investigations, engineering decisions, and current end-to-end evaluation.

Run the latest evaluation with `python -m src.rag.evaluate_exact_rag` (requires a configured OpenAI API key and incurs API usage).

## Automated API Testing

Automated tests use pytest and FastAPI's TestClient.

Run:

```bash
python -m pytest tests/test_api.py -v
```

The test suite verifies:

1. The `/health` endpoint responds successfully.
2. The `/ask` endpoint accepts questions and returns structured answers.
3. Requests missing the required query are rejected with HTTP 422.

The `/ask` test uses mocking to avoid live OpenAI API requests.

**Previously recorded API test result: 3 passed.**

Additional regression tests for retrieval, fusion, and mocked RAG integration: `python -m pytest tests/test_retrieval_regression.py -v` (**5 passed** in the latest recorded run).

These tests validate API behavior rather than the accuracy of the underlying retrieval or language model.

## Current Status

Implemented:

- PDF codebook ingestion and structured variable extraction.
- SentenceTransformer embedding generation.
- Persistent Chroma vector storage.
- Exact NumPy candidate retrieval from persisted Chroma vectors; indexed Chroma retrieval retained for comparison.
- CrossEncoder reranking and Reciprocal Rank Fusion (RRF).
- Retrieval and RAG evaluation scripts.
- OpenAI-based structured answer generation.
- FastAPI endpoints.
- Automated API tests.

### Planned Improvements

- Evaluate on a new, independent set of unseen questions.
- Benchmark exact-retrieval latency and memory use; continue investigating intermittent indexed-search discrepancies.
- Add error handling and logging.
- Containerize the application with Docker.
- Prepare for cloud deployment.

## Limitations

- The system retrieves variable definitions from a CMS codebook; it does not analyze actual patient claims.
- LLM-generated answers may be incorrect and should be verified against official CMS documentation.
- The current evaluation dataset is small.
- The application requires an OpenAI API key and a locally available Chroma vector database.
- The API is a development implementation and has not yet been hardened for public deployment.

## Data Source

Centers for Medicare & Medicaid Services (CMS), Chronic Conditions Data Warehouse (CCW), Medicare Fee-for-Service Claims Codebook, Version L, August 2026 (V1.17).

## Author

Jessica D. Geiger

M.S. Data Science

[GitHub](https://github.com/danigeiger) | [LinkedIn](https://www.linkedin.com/in/jessica-d-geiger/)

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
