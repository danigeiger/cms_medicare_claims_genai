
from src.retrieval.evaluate_retrieval import TEST_QUERIES
from src.retrieval.retrieval import retrieve
from src.retrieval.reranker import rerank


def calculate_rrf(chroma_ids, reranked_ids, rrf_k=60):
    """Combine two ranked lists using Reciprocal Rank Fusion."""

    scores = {}

    for rank, variable in enumerate(chroma_ids, start=1):
        scores[variable] = scores.get(variable, 0) + (
            1 / (rrf_k + rank)
        )

    for rank, variable in enumerate(reranked_ids, start=1):
        scores[variable] = scores.get(variable, 0) + (
            1 / (rrf_k + rank)
        )

    return [
        variable
        for variable, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def evaluate_rrf():
    """Compare Chroma, CrossEncoder, and RRF retrieval accuracy."""

    strategies = ["Chroma", "CrossEncoder", "RRF"]

    results = {
        strategy: {1: 0, 3: 0, 5: 0}
        for strategy in strategies
    }

    total = len(TEST_QUERIES)

    for index, test_case in enumerate(TEST_QUERIES, start=1):

        query = test_case["query"]
        expected = test_case["expected"]

        # Retrieve 20 candidates from Chroma
        chroma_results = retrieve(
            query,
            retrieve_function_top_k=20,
        )

        chroma_ids = chroma_results["ids"][0]

        # Rerank 20 candidates with CrossEncoder
        reranked_results = rerank(
            query,
            rerank_function_top_k=20,
        )

        reranked_ids = [
            candidate["variable"]
            for candidate in reranked_results
        ]

        # Combine both rankings
        rrf_ids = calculate_rrf(
            chroma_ids,
            reranked_ids,
        )

        rankings = {
            "Chroma": chroma_ids,
            "CrossEncoder": reranked_ids,
            "RRF": rrf_ids,
        }

        print(f"\n[{index}/{total}] {query}")
        print(f"Expected: {expected}")

        for strategy, ranked_ids in rankings.items():

            rank = (
                ranked_ids.index(expected) + 1
                if expected in ranked_ids
                else None
            )

            print(f"  {strategy:12} Rank: {rank}")

            for k in [1, 3, 5]:
                if expected in ranked_ids[:k]:
                    results[strategy][k] += 1

    print("\n" + "=" * 55)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 55)

    print(f"Total questions: {total}\n")

    for strategy in strategies:

        print(strategy)

        for k in [1, 3, 5]:

            correct = results[strategy][k]
            accuracy = (correct / total) * 100

            print(
                f"  Top-{k}: {correct}/{total} "
                f"({accuracy:.1f}%)"
            )

        print("-" * 35)


if __name__ == "__main__":
    evaluate_rrf()
