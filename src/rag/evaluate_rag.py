
from src.rag.rag import generate_cms_answer
from src.retrieval.evaluate_retrieval import TEST_QUERIES


def evaluate_rag():
    """Evaluate structured CMS answers against expected variables."""

    correct_predictions = 0
    total_questions = len(TEST_QUERIES)

    for test_case in TEST_QUERIES:
        query = test_case["query"]
        expected_variable = test_case["expected"]

        # Generate structured answer
        cms_answer = generate_cms_answer(query)

        # Extract the predicted CMS variable
        predicted_variable = cms_answer.variable

        # Compare the exact variable identifiers
        is_correct = predicted_variable == expected_variable

        if is_correct:
            correct_predictions += 1

        print(f"\nQuestion: {query}")
        print(f"Expected: {expected_variable}")
        print(f"Predicted: {predicted_variable}")
        print(f"Correct: {is_correct}")
        print(f"Explanation: {cms_answer.explanation}")
        print("-" * 60)

    accuracy = correct_predictions / total_questions

    print("\n=== Structured RAG Evaluation Results ===")
    print(f"Correct: {correct_predictions}/{total_questions}")
    print(f"Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    evaluate_rag()
