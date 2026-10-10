
import os

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from src.retrieval.reranker import rerank


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
    """Convert CMS candidate documents into readable LLM context."""

    cms_document_texts = []

    for candidate in reranked_cms_candidates:
        cms_document_texts.append(
            candidate["document"]
        )

    formatted_cms_context = "\n\n".join(
        cms_document_texts
    )

    return formatted_cms_context


def generate_cms_answer(query: str) -> str:
    """Generate an answer using retrieved CMS codebook context."""

    # Step 1: Retrieve and rerank CMS variables
    reranked_cms_candidates = get_cms_context(query)

    # Step 2: Format the retrieved documents
    formatted_cms_context = format_cms_context(
        reranked_cms_candidates
    )

    # Step 3: Connect to the OpenAI API
    openai_client = OpenAI()

    # Step 4: Send the question and CMS context to the LLM
    llm_response = openai_client.responses.create(
        model="gpt-4.1-mini",
        instructions=(
            "You are an assistant specializing in CMS Medicare "
            "claims data dictionaries. Answer the user's question "
            "using only the provided CMS codebook context. "
            "Identify the most relevant CMS variable and explain "
            "why it matches. Do not invent CMS variables or "
            "definitions. If the context is insufficient, say so."
        ),
        input=(
            f"CMS CODEBOOK CONTEXT:\n"
            f"{formatted_cms_context}\n\n"
            f"USER QUESTION:\n{query}"
        ),
    )

    # Step 5: Extract the generated answer
    cms_answer = llm_response.output_text

    return cms_answer


if __name__ == "__main__":
    query = "What is the specialty of the attending physician?"

    print("\n=== CMS Medicare RAG Test ===\n")
    print(f"Question: {query}\n")

    answer = generate_cms_answer(query)

    print("Generated answer:\n")
    print(answer)
