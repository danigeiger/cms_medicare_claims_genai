
from src.rag.rag import generate_cms_answer
from src.retrieval.evaluate_retrieval import TEST_QUERIES


def evaluate_rag():
    """Evaluate generated CMS answers against known variables."""

    correct_predictions = 0
    total_questions = len(TEST_QUERIES)

    for test_case in TEST_QUERIES:
        query = test_case["query"]
        expected_variable = test_case["expected"]

        generated_answer = generate_cms_answer(query)

        is_correct = expected_variable in generated_answer

        if is_correct:
            correct_predictions += 1

        print(f"\nQuestion: {query}")
        print(f"Expected: {expected_variable}")
        print(f"Correct: {is_correct}")
        print(f"Generated answer:\n{generated_answer}")
        print("-" * 60)

    accuracy = correct_predictions / total_questions

    print("\n=== RAG Evaluation Results ===")
    print(f"Correct: {correct_predictions}/{total_questions}")
    print(f"Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    evaluate_rag()
