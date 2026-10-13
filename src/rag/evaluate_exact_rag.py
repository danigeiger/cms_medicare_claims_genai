
from src.rag.rag import generate_cms_answer
from src.retrieval.evaluate_retrieval import TEST_QUERIES


def evaluate_exact_rag():
    """Evaluate end-to-end RAG accuracy with exact NumPy retrieval."""

    total = len(TEST_QUERIES)
    correct = 0
    results = []

    print("\n=== EXACT RETRIEVAL RAG EVALUATION ===\n")

    for index, test_case in enumerate(TEST_QUERIES, start=1):
        query = test_case["query"]
        expected = test_case["expected"]

        print(f"[{index}/{total}] {query}", flush=True)

        try:
            answer = generate_cms_answer(query)
            predicted = answer.variable
            is_correct = predicted == expected

            if is_correct:
                correct += 1

            results.append({
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "error": None,
            })

            print(f"  Expected:  {expected}")
            print(f"  Predicted: {predicted}")
            print(f"  Correct:   {is_correct}\n", flush=True)

        except Exception as exc:
            results.append({
                "query": query,
                "expected": expected,
                "predicted": None,
                "correct": False,
                "error": str(exc),
            })

            print(f"  ERROR: {exc}\n", flush=True)

    completed = sum(
        result["error"] is None
        for result in results
    )

    print("=" * 55)
    print("FINAL EXACT RETRIEVAL RAG EVALUATION")
    print("=" * 55)
    print(f"Total questions: {total}")
    print(f"Successful API responses: {completed}")
    print(f"Correct answers: {correct}")

    if completed:
        print(
            f"Accuracy on completed questions: "
            f"{correct / completed:.1%}"
        )

    print(
        f"Overall success rate: "
        f"{correct / total:.1%}"
    )

    print("\nIncorrect answers or errors:")

    for result in results:
        if not result["correct"]:
            print(f"\nQuestion: {result['query']}")
            print(f"Expected: {result['expected']}")
            print(f"Predicted: {result['predicted']}")
            if result["error"]:
                print(f"Error: {result['error']}")


if __name__ == "__main__":
    evaluate_exact_rag()
