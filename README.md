# CMS Medicare Claims GenAI — RAG Codebook Assistant

A Retrieval-Augmented Generation (RAG) application that helps users identify and understand variables in the **CMS Medicare Fee-for-Service (FFS) Claims Codebook** using natural-language questions.

The application combines semantic search, transformer-based reranking, OpenAI language models, and a FastAPI interface to retrieve relevant CMS variable definitions and generate structured answers.

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
        retrieval.py
             |
             v
       Chroma Database
       Top 20 Candidates
             |
             v
         reranker.py
     CrossEncoder Reranking
             |
             v
       Top 5 Candidates
             |
             v
       OpenAI GPT-4.1-mini
             |
             v
     Structured CMSAnswer
             |
             v
       FastAPI JSON Response
```

### Retrieval and Reranking

The system uses two complementary transformer models:

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`

- Generates 384-dimensional embeddings.
- Represents CMS variable descriptions and user questions as vectors.
- Enables semantic similarity search in Chroma.

**Reranking model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

- Evaluates the relevance of retrieved candidates against the original question.
- Reorders the top 20 retrieved candidates.
- Selects the five highest-scoring candidates for answer generation.

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
│   └── retrieval_experiments.md
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
│   │   └── evaluate_retrieval.py
│   |
│   ├── rag/
│   │   ├── rag.py
│   │   └── evaluate_rag.py
│   |
│   └── api/
│       └── main.py
|
├── tests/
│   └── test_api.py
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

The retrieval system was evaluated using 20 natural-language questions with expected CMS variable identifiers.

### Retrieval Experiments

| Experiment | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Initial Chroma retrieval | 60% | 80% | 95% |
| Improved document labels | 60% | 90% | 100% |
| CrossEncoder reranking (initial 5 candidates) | 80% | 100% | 100% |

Top-k accuracy measures whether the expected CMS variable appears within the first k ranked results.

The current implementation retrieves 20 candidates before reranking, but the table above reflects the earlier evaluated configurations. The revised 20-candidate configuration requires a separate full evaluation.

### RAG Answer Evaluation

The structured RAG pipeline was evaluated against 20 questions using exact variable-identifier matching.

**Observed result: 19/20 correct (95%).**

This result was recorded before the most recent retrieval expansion. Subsequent testing successfully identified `BENE_RACE_CD` for a previously unsuccessful race-related question, but a complete updated evaluation has not yet been performed.

These results are preliminary and based on a small test set; they do not establish production-level accuracy.

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

**Latest test result: 3 passed.**

These tests validate API behavior rather than the accuracy of the underlying retrieval or language model.

## Current Status

Implemented:

- PDF codebook ingestion and structured variable extraction.
- SentenceTransformer embedding generation.
- Persistent Chroma vector storage.
- Semantic retrieval.
- CrossEncoder reranking.
- Retrieval and RAG evaluation scripts.
- OpenAI-based structured answer generation.
- FastAPI endpoints.
- Automated API tests.

### Planned Improvements

- Reevaluate the expanded retrieval and reranking pipeline.
- Expand the evaluation dataset and investigate inconsistent retrieval results.
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

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
