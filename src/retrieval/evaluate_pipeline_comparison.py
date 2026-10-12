
from src.retrieval.evaluate_retrieval import TEST_QUERIES
from src.retrieval.retrieval import retrieve, retrieve_exact
from src.retrieval.reranker import rerank_candidates
from src.retrieval.evaluate_rrf import calculate_rrf


def evaluate():
    pipelines = {
        "Chroma + RRF": retrieve,
        "Exact + RRF": retrieve_exact,
    }

    scores = {
        name: {1: 0, 3: 0, 5: 0}
        for name in pipelines
    }

    total = len(TEST_QUERIES)

    for index, test_case in enumerate(TEST_QUERIES, start=1):
        query = test_case["query"]
        expected = test_case["expected"]

        print(f"\n[{index}/{total}] {query}")
        print(f"Expected: {expected}")

        for name, retrieval_function in pipelines.items():

            # Stage 1: Retrieve 20 candidates
            retrieved = retrieval_function(
                query,
                retrieve_function_top_k=20,
            )

            retrieved_ids = retrieved["ids"][0]

            # Stage 2: CrossEncoder reranking
            reranked = rerank_candidates(
                query,
                retrieved,
                top_k=20,
            )

            reranked_ids = [
                candidate["variable"]
                for candidate in reranked
            ]

            # Stage 3: Reciprocal Rank Fusion
            fused_ids = calculate_rrf(
                retrieved_ids,
                reranked_ids,
            )

            rank = (
                fused_ids.index(expected) + 1
                if expected in fused_ids
                else None
            )

            print(f"  {name:15} Rank: {rank}")

            for k in [1, 3, 5]:
                if expected in fused_ids[:k]:
                    scores[name][k] += 1

    print("\n" + "=" * 55)
    print("FULL RETRIEVAL PIPELINE COMPARISON")
    print("=" * 55)

    for name in pipelines:
        print(f"\n{name}")

        for k in [1, 3, 5]:
            correct = scores[name][k]
            accuracy = correct / total * 100

            print(
                f"  Top-{k}: {correct}/{total} "
                f"({accuracy:.1f}%)"
            )


if __name__ == "__main__":
    evaluate()

