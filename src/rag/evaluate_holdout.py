
import argparse
import json
from pathlib import Path

from src.rag.rag import get_cms_context, generate_cms_answer


HOLDOUT_PATH = Path("tests/evaluation_holdout.json")
RESULTS_DIR = Path("evaluation_results")


def load_questions():
    with HOLDOUT_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate(mode):
    questions = load_questions()
    results = []

    recall_hits = 0
    correct_answers = 0
    api_errors = 0

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected = item["expected_variable"]

        print(f"\n[{index}/{len(questions)}] {question}", flush=True)

        result = {
            "id": item["id"],
            "question": question,
            "expected_variable": expected,
        }

        try:
            candidates = get_cms_context(question)

            retrieved_variables = [
                candidate["variable"]
                for candidate in candidates
            ]

            recall_hit = expected in retrieved_variables
            recall_hits += int(recall_hit)

            result["retrieved_variables"] = retrieved_variables
            result["recall_at_5"] = recall_hit

            print(f"Expected: {expected}")
            print(f"Recall@5: {'PASS' if recall_hit else 'FAIL'}")

            if mode == "full":
                try:
                    answer = generate_cms_answer(question)
                    predicted = answer.variable
                    correct = predicted == expected

                    correct_answers += int(correct)

                    result["predicted_variable"] = predicted
                    result["correct"] = correct
                    result["explanation"] = answer.explanation

                    print(f"Predicted: {predicted}")
                    print(
                        f"Answer: {'CORRECT' if correct else 'INCORRECT'}"
                    )

                except Exception as error:
                    api_errors += 1
                    result["api_error"] = str(error)
                    print(f"API error: {error}")

        except Exception as error:
            result["retrieval_error"] = str(error)
            print(f"Retrieval error: {error}")

        results.append(result)

    total = len(questions)

    summary = {
        "mode": mode,
        "total_questions": total,
        "recall_at_5": recall_hits / total,
        "retrieval_hits": recall_hits,
    }

    if mode == "full":
        summary["correct_answers"] = correct_answers
        summary["api_errors"] = api_errors
        summary["end_to_end_accuracy"] = correct_answers / total

    RESULTS_DIR.mkdir(exist_ok=True)

    output_path = RESULTS_DIR / f"holdout_{mode}_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "results": results},
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("HOLDOUT EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Total questions: {total}")
    print(f"Retrieval hits: {recall_hits}")
    print(f"Recall@5: {recall_hits / total:.1%}")

    if mode == "full":
        print(f"Correct answers: {correct_answers}")
        print(f"API errors: {api_errors}")
        print(f"End-to-end accuracy: {correct_answers / total:.1%}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["retrieval", "full"],
        default="retrieval",
    )
    args = parser.parse_args()

    evaluate(args.mode)
