
from src.retrieval.retrieval import retrieve
from src.retrieval.reranker import rerank


def reciprocal_rank_fusion(query: str, top_k: int = 5, rrf_k: int = 60):
    """Combine Chroma and CrossEncoder rankings using RRF."""

    # Stage 1: Retrieve 20 candidates from Chroma
    chroma_results = retrieve(
        query,
        retrieve_function_top_k=20,
    )

    # Chroma returns nested lists because it supports batch queries
    chroma_ids = chroma_results["ids"][0]

    # Stage 2: Get the CrossEncoder ranking of those candidates
    reranked_results = rerank(
        query,
        rerank_function_top_k=20,
    )

    reranked_ids = [
        candidate["variable"]
        for candidate in reranked_results
    ]

    # Stage 3: Calculate combined RRF scores
    scores = {}

    for rank, variable in enumerate(chroma_ids, start=1):
        scores[variable] = scores.get(variable, 0) + (
            1 / (rrf_k + rank)
        )

    for rank, variable in enumerate(reranked_ids, start=1):
        scores[variable] = scores.get(variable, 0) + (
            1 / (rrf_k + rank)
        )

    # Stage 4: Sort by combined score
    fused_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return fused_results[:top_k]


if __name__ == "__main__":

    queries = [
        "Where was the service performed?",
        "What is the place of service code for a claim line item?",
        "What type of setting was the medical service performed in?",
    ]

    expected = "LINE_PLACE_OF_SRVC_CD"

    for query in queries:

        print(f"\nQUESTION: {query}")

        results = reciprocal_rank_fusion(
            query,
            top_k=20,
        )

        for rank, (variable, score) in enumerate(results, start=1):

            marker = " <-- EXPECTED" if variable == expected else ""

            print(
                f"{rank:2}. {variable:35} "
                f"RRF score: {score:.6f}{marker}"
            )

        print("-" * 65)
