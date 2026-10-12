
import json
import subprocess
import sys

RUNS = 10

TEST_CODE = """
import json

from src.retrieval.evaluate_retrieval import TEST_QUERIES
from src.retrieval.retrieval import retrieve, retrieve_exact

differences = []

for test_case in TEST_QUERIES:
    query = test_case["query"]
    expected = test_case["expected"]

    chroma_ids = retrieve(
        query, retrieve_function_top_k=20
    )["ids"][0]

    exact_ids = retrieve_exact(
        query, retrieve_function_top_k=20
    )["ids"][0]

    if chroma_ids != exact_ids:
        differences.append({
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
            "chroma_top5": chroma_ids[:5],
            "exact_top5": exact_ids[:5],
        })

print("RESULT_JSON:" + json.dumps(differences))
"""


def main():
    total_disagreements = 0
    failed_runs = 0

    print(f"Testing {RUNS} independent Python processes...\n")

    for run in range(1, RUNS + 1):
        completed = subprocess.run(
            [sys.executable, "-c", TEST_CODE],
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            failed_runs += 1
            print(f"Run {run}: ERROR")
            print(completed.stderr[-1500:])
            continue

        result_line = next(
            (
                line for line in completed.stdout.splitlines()
                if line.startswith("RESULT_JSON:")
            ),
            None,
        )

        if result_line is None:
            failed_runs += 1
            print(f"Run {run}: No results returned")
            print(completed.stdout[-1000:])
            continue

        differences = json.loads(
            result_line.removeprefix("RESULT_JSON:")
        )

        total_disagreements += len(differences)

        print(
            f"Run {run}/{RUNS}: "
            f"{len(differences)} ranking differences"
        )

        for item in differences:
            print(f"  Question: {item['query']}")
            print(f"  Expected: {item['expected']}")
            print(f"  Chroma rank: {item['chroma_rank']}")
            print(f"  Exact rank: {item['exact_rank']}")
            print(f"  Chroma top 5: {item['chroma_top5']}")
            print(f"  Exact top 5: {item['exact_top5']}")

    successful_runs = RUNS - failed_runs

    print("\n" + "=" * 55)
    print("RELIABILITY TEST SUMMARY")
    print("=" * 55)
    print(f"Successful runs: {successful_runs}/{RUNS}")
    print(f"Failed runs: {failed_runs}")
    print(f"Questions per run: 20")
    print(f"Total comparisons: {successful_runs * 20}")
    print(f"Ranking disagreements: {total_disagreements}")


if __name__ == "__main__":
    main()
