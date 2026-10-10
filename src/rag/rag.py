
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.retrieval.reranker import rerank


# Load environment variables from .env
load_dotenv()


class CMSAnswer(BaseModel):
    """Structured answer identifying a CMS codebook variable."""

    variable: str | None
    label: str | None
    explanation: str


def get_cms_context(query: str) -> list[dict]:
    """Retrieve and rerank CMS codebook candidates."""

    reranked_cms_candidates = rerank(
        query,
        rerank_function_top_k=5,
    )

    return reranked_cms_candidates


def format_cms_context(
    reranked_cms_candidates: list[dict],
) -> str:
    """Convert reranked CMS documents into readable LLM context."""

    cms_document_texts = []

    for candidate in reranked_cms_candidates:
        cms_document_texts.append(
            candidate["document"]
        )

    formatted_cms_context = "\n\n".join(
        cms_document_texts
    )

    return formatted_cms_context


def generate_cms_answer(query: str) -> CMSAnswer:
    """Generate a structured answer using CMS codebook context."""

    # Step 1: Retrieve and rerank CMS variables
    reranked_cms_candidates = get_cms_context(query)

    # Step 2: Format the retrieved CMS documents
    formatted_cms_context = format_cms_context(
        reranked_cms_candidates
    )

    # Step 3: Initialize the OpenAI client
    openai_client = OpenAI()

    # Step 4: Generate a structured CMS answer
    llm_response = openai_client.responses.parse(
        model="gpt-4.1-mini",
        instructions=(
            "You are an assistant specializing in CMS Medicare "
            "claims data dictionaries. "
            "Answer the user's question using only the supplied "
            "CMS codebook context. "
            "Select the single most relevant CMS variable. "
            "Return its exact variable identifier, label, "
            "and a concise explanation of why it matches. "
            "Do not invent CMS variables or definitions. "
            "If the context does not contain sufficient information "
            "to identify the correct variable, return null for "
            "variable and label, and explain why."
        ),
        input=(
            f"CMS CODEBOOK CONTEXT:\n"
            f"{formatted_cms_context}\n\n"
            f"USER QUESTION:\n"
            f"{query}"
        ),
        text_format=CMSAnswer,
    )

    # Step 5: Extract the structured response
    cms_answer = llm_response.output_parsed

    if cms_answer is None:
        raise ValueError(
            "The LLM did not return a structured CMS answer."
        )

    return cms_answer


if __name__ == "__main__":

    query = "What is the specialty of the attending physician?"

    print("\n=== Structured CMS Medicare RAG Test ===\n")
    print(f"Question: {query}\n")

    cms_answer = generate_cms_answer(query)

    print(f"Variable: {cms_answer.variable}")
    print(f"Label: {cms_answer.label}")
    print(f"Explanation: {cms_answer.explanation}")
