
from openai import OpenAI

from src.retrieval.retrieval import retrieve_exact
from src.retrieval.reranker import rerank_candidates
from src.retrieval.evaluate_rrf import calculate_rrf
from src.rag.rag import CMSAnswer, format_cms_context


QUERY = (
    "What code identifies the type of location where an "
    "individual claim-line service was performed?"
)

EXPECTED = "LINE_PLACE_OF_SRVC_CD"


def main():
    # 1. Retrieve the 20 nearest CMS variables.
    results = retrieve_exact(
        QUERY,
        retrieve_function_top_k=20,
    )

    # 2. Rerank those same 20 candidates.
    reranked = rerank_candidates(
        QUERY,
        results,
        top_k=20,
    )

    # 3. Combine the two rankings using RRF.
    rrf_ids = calculate_rrf(
        results["ids"][0],
        [item["variable"] for item in reranked],
    )

    # 4. Build a lookup of candidate details.
    candidate_lookup = {
        item["variable"]: item
        for item in reranked
    }

    # 5. Select the top five RRF candidates.
    top_five = [
        candidate_lookup[variable]
        for variable in rrf_ids[:5]
    ]

    print("\nRRF TOP FIVE CANDIDATES")
    print("=" * 55)

    for rank, candidate in enumerate(top_five, start=1):
        print(f"{rank}. {candidate['variable']}")

    # 6. Format the same context used by the main RAG pipeline.
    context = format_cms_context(top_five)

    # 7. Ask GPT-4.1-mini to select a variable.
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
            f"Question: {QUERY}"
        ),
        text_format=CMSAnswer,
    )

    answer = response.output_parsed

    print("\nGPT-4.1-mini RESULT")
    print("=" * 55)
    print(f"Expected:  {EXPECTED}")
    print(f"Predicted: {answer.variable}")
    print(f"Label:     {answer.label}")
    print(f"Reason:    {answer.explanation}")

    if answer.variable == EXPECTED:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")


if __name__ == "__main__":
    main()
