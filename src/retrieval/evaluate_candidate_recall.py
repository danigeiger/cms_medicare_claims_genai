
from src.retrieval.retrieval import retrieve_exact
from src.retrieval.reranker import rerank_candidates
from src.retrieval.evaluate_rrf import calculate_rrf
from src.retrieval.evaluate_retrieval import TEST_QUERIES


def evaluate_candidate_recall():
    crossencoder_hits = 0
    rrf_hits = 0

    print("\nCROSSENCODER VS. RRF — RECALL@5")
    print("=" * 65)

    for index, test in enumerate(TEST_QUERIES, start=1):
        query = test["query"]
        expected = test["expected"]

        # Stage 1: Exact embedding retrieval
        retrieval_results = retrieve_exact(
            query,
            retrieve_function_top_k=20,
        )
        retrieval_ids = retrieval_results["ids"][0]

        # Stage 2: CrossEncoder reranking
        reranked = rerank_candidates(
            query,
            retrieval_results,
            top_k=20,
        )
        crossencoder_ids = [
            candidate["variable"]
            for candidate in reranked
        ]

        # Stage 3: Reciprocal Rank Fusion
        rrf_ids = calculate_rrf(
            retrieval_ids,
            crossencoder_ids,
        )

        crossencoder_correct = expected in crossencoder_ids[:5]
        rrf_correct = expected in rrf_ids[:5]

        crossencoder_hits += crossencoder_correct
        rrf_hits += rrf_correct

        print(f"\n{index}. {query}")
        print(f"   Expected: {expected}")
        print(
            f"   CrossEncoder: "
            f"{'HIT' if crossencoder_correct else 'MISS'}"
        )
        print(
            f"   RRF:          "
            f"{'HIT' if rrf_correct else 'MISS'}"
        )

        if crossencoder_correct != rrf_correct:
            print("   *** METHODS DISAGREE ***")

    total = len(TEST_QUERIES)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(
        f"CrossEncoder Recall@5: "
        f"{crossencoder_hits}/{total} "
        f"({crossencoder_hits / total:.1%})"
    )
    print(
        f"RRF Recall@5:          "
        f"{rrf_hits}/{total} "
        f"({rrf_hits / total:.1%})"
    )


if __name__ == "__main__":
    evaluate_candidate_recall()
