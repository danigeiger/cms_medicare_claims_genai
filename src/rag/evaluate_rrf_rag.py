
from dotenv import load_dotenv
from openai import OpenAI

from src.rag.rag import CMSAnswer, format_cms_context
from src.retrieval.evaluate_retrieval import TEST_QUERIES
from src.retrieval.retrieval import retrieve
from src.retrieval.reranker import rerank
from src.retrieval.evaluate_rrf import calculate_rrf


load_dotenv()

client = OpenAI()

INSTRUCTIONS = (
    "You are an assistant specializing in CMS Medicare "
    "claims data dictionaries. "
    "Answer the user's question using only the supplied "
    "CMS codebook context. "
    "Select the single most relevant CMS variable. "
    "Return its exact variable identifier, label, "
    "and a concise explanation of why it matches. "
    "Do not invent CMS variables or definitions. "
    "If the context does not contain sufficient information "
    "to identify the correct variable, return null for "
    "variable and label, and explain why."
)


def generate_answer(query, candidates):
    """Generate a structured CMS answer from supplied candidates."""

    context = format_cms_context(candidates)

    response = client.responses.parse(
        model="gpt-4.1-mini",
        instructions=INSTRUCTIONS,
        input=(
            f"CMS CODEBOOK CONTEXT:\n"
            f"{context}\n\n"
            f"USER QUESTION:\n"
            f"{query}"
        ),
        text_format=CMSAnswer,
    )

    if response.output_parsed is None:
        raise ValueError("No structured answer returned.")

    return response.output_parsed


def evaluate():
    scores = {
        "CrossEncoder": 0,
        "RRF": 0,
    }

    total = len(TEST_QUERIES)

    for index, test_case in enumerate(TEST_QUERIES, start=1):
        query = test_case["query"]
        expected = test_case["expected"]

        # Retrieve candidates and their original Chroma ranks
        chroma_results = retrieve(
            query,
            retrieve_function_top_k=20,
        )

        chroma_ids = chroma_results["ids"][0]

        # Rerank the same query with CrossEncoder
        reranked = rerank(
            query,
            rerank_function_top_k=20,
        )

        reranked_ids = [
            candidate["variable"]
            for candidate in reranked
        ]

        # Build RRF ranking
        rrf_ids = calculate_rrf(
            chroma_ids,
            reranked_ids,
        )

        # Look up complete candidate dictionaries
        candidate_lookup = {
            candidate["variable"]: candidate
            for candidate in reranked
        }

        crossencoder_candidates = reranked[:5]

        rrf_candidates = [
            candidate_lookup[variable]
            for variable in rrf_ids[:5]
        ]

        # Run both pipelines through the same LLM
        crossencoder_answer = generate_answer(
            query,
            crossencoder_candidates,
        )

        rrf_answer = generate_answer(
            query,
            rrf_candidates,
        )

        crossencoder_correct = (
            crossencoder_answer.variable == expected
        )

        rrf_correct = (
            rrf_answer.variable == expected
        )

        scores["CrossEncoder"] += int(crossencoder_correct)
        scores["RRF"] += int(rrf_correct)

        print(f"\n[{index}/{total}] {query}")
        print(f"Expected: {expected}")

        print(
            f"CrossEncoder: {crossencoder_answer.variable} "
            f"{'PASS' if crossencoder_correct else 'FAIL'}"
        )

        print(
            f"RRF:          {rrf_answer.variable} "
            f"{'PASS' if rrf_correct else 'FAIL'}"
        )

    print("\n" + "=" * 50)
    print("FINAL END-TO-END RAG EVALUATION")
    print("=" * 50)

    for strategy, correct in scores.items():
        accuracy = 100 * correct / total

        print(
            f"{strategy}: {correct}/{total} "
            f"({accuracy:.1f}%)"
        )


if __name__ == "__main__":
    evaluate()
