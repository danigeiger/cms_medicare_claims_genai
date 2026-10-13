from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from src.retrieval.retrieval import retrieve_exact
from src.retrieval.reranker import rerank_candidates
from src.retrieval.evaluate_rrf import calculate_rrf

load_dotenv()


class CMSAnswer(BaseModel):
    variable: str | None
    label: str | None
    explanation: str


def get_cms_context(query: str):
    """
    Retrieve CMS variables using exact similarity search,
    CrossEncoder reranking, and Reciprocal Rank Fusion.
    """

    # Step 1: Retrieve the 20 closest CMS variables.
    retrieval_results = retrieve_exact(
        query,
        retrieve_function_top_k=20,
    )

    retrieval_ids = retrieval_results["ids"][0]

    # Step 2: Rerank all 20 candidates with CrossEncoder.
    reranked_candidates = rerank_candidates(
        query,
        retrieval_results,
        top_k=20,
    )

    reranked_ids = [
        candidate["variable"]
        for candidate in reranked_candidates
    ]

    # Step 3: Combine both rankings using RRF.
    rrf_ids = calculate_rrf(
        retrieval_ids,
        reranked_ids,
    )

    # Step 4: Look up the candidate documents.
    candidate_lookup = {
        candidate["variable"]: candidate
        for candidate in reranked_candidates
    }

    # Step 5: Select the five highest-ranked RRF candidates.
    top_five = [
        candidate_lookup[variable]
        for variable in rrf_ids[:5]
    ]

    return top_five


def format_cms_context(candidates):
    """
    Format retrieved CMS codebook entries for the LLM.
    """

    return "\n\n".join(
        candidate["document"]
        for candidate in candidates
    )


def generate_cms_answer(query: str) -> CMSAnswer:
    """
    Generate a structured CMS variable answer using GPT-4.1-mini.
    """

    candidates = get_cms_context(query)
    context = format_cms_context(candidates)

    client = OpenAI()

    response = client.responses.parse(
        model="gpt-4.1-mini",
        instructions=(
            "You are an assistant that identifies CMS Medicare "
            "claims codebook variables. Use only the supplied "
            "codebook context. Select the exact variable that "
            "best answers the user's question. Do not invent "
            "variable names. If no variable matches, return "
            "null for variable and label."
        ),
        input=(
            f"CMS codebook context:\n{context}\n\n"
            f"Question: {query}"
        ),
        text_format=CMSAnswer,
    )

    return response.output_parsed


if __name__ == "__main__":
    question = (
        "What code identifies the type of location where an "
        "individual claim-line service was performed?"
    )

    answer = generate_cms_answer(question)

    print(f"Variable: {answer.variable}")
    print(f"Label: {answer.label}")
    print(f"Explanation: {answer.explanation}")
