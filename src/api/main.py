
from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.rag import CMSAnswer, generate_cms_answer


# Create the FastAPI application
app = FastAPI(
    title="CMS Medicare Claims RAG API",
    description="Semantic search and AI-powered answers for CMS Medicare claims variables.",
    version="1.0.0",
)


# Define the format of an incoming question
class CMSQuery(BaseModel):
    query: str


# Define a basic health-check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Define the CMS question-answering endpoint
@app.post("/ask", response_model=CMSAnswer)
def ask_cms_question(request: CMSQuery):
    """Retrieve CMS codebook context and generate an answer."""

    cms_answer = generate_cms_answer(request.query)

    return cms_answer
