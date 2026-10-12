
from src.retrieval.evaluate_retrieval import TEST_QUERIES
from src.retrieval.retrieval import retrieve, retrieve_exact


def evaluate():
    """Compare Chroma HNSW retrieval with exact NumPy retrieval."""

    methods = {
        "Chroma": retrieve,
        "Exact NumPy": retrieve_exact,
    }

    scores = {
        method: {1: 0, 3: 0, 5: 0}
        for method in methods
    }

    total = len(TEST_QUERIES)
    disagreements = []

    for index, test_case in enumerate(TEST_QUERIES, start=1):
        query = test_case["query"]
        expected = test_case["expected"]

        rankings = {}

        for name, retrieval_function in methods.items():
            results = retrieval_function(
                query,
                retrieve_function_top_k=20,
            )

            ids = results["ids"][0]
            rankings[name] = ids

            for k in [1, 3, 5]:
                if expected in ids[:k]:
                    scores[name][k] += 1

        chroma_ids = rankings["Chroma"]
        exact_ids = rankings["Exact NumPy"]

        if chroma_ids != exact_ids:
            disagreements.append({
                "query": query,
                "expected": expected,
                "chroma_rank": (
                    chroma_ids.index(expected) + 1
                    if expected in chroma_ids else None
                ),
                "exact_rank": (
                    exact_ids.index(expected) + 1
                    if expected in exact_ids else None
                ),
            })

        print(f"\n[{index}/{total}] {query}")
        print(f"Expected: {expected}")

        for name, ids in rankings.items():
            rank = ids.index(expected) + 1 if expected in ids else None
            print(f"  {name:12} Rank: {rank}")

    print("\n" + "=" * 55)
    print("CHROMA VS EXACT NUMPY EVALUATION")
    print("=" * 55)

    for name in methods:
        print(f"\n{name}")

        for k in [1, 3, 5]:
            correct = scores[name][k]
            accuracy = (correct / total) * 100

            print(f"  Top-{k}: {correct}/{total} ({accuracy:.1f}%)")

    print(f"\nQuestions with ranking differences: {len(disagreements)}")

    for item in disagreements:
        print(f"\nQuestion: {item['query']}")
        print(f"Expected: {item['expected']}")
        print(f"Chroma rank: {item['chroma_rank']}")
        print(f"Exact rank: {item['exact_rank']}")


if __name__ == "__main__":
    evaluate()

