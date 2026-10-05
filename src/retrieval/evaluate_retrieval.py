from src.retrieval.retrieval import retrieve
from src.retrieval.reranker import rerank


TEST_QUERIES = [
    {
        "query": "What was the patient's diagnosis when they were admitted?",
        "expected": "ADMTG_DGNS_CD",
    },
    {
        "query": "What is the specialty of the doctor attending the patient?",
        "expected": "AT_PHYSN_SPCLTY_CD",
    },
    {
        "query": "What is the NPI of the doctor responsible for the patient?",
        "expected": "AT_PHYSN_NPI",
    },
    {
        "query": "How much did Medicare pay for the claim?",
        "expected": "CLM_PMT_AMT",
    },
    {
        "query": "When was the patient admitted?",
        "expected": "CLM_ADMSN_DT",
    },
    {
        "query": "When was the patient discharged?",
        "expected": "NCH_BENE_DSCHRG_DT",
    },
    {
        "query": "What is the patient's date of birth?",
        "expected": "DOB_DT",
    },
    {
        "query": "What is the patient's race?",
        "expected": "BENE_RACE_CD",
    },
    {
        "query": "What state does the beneficiary live in?",
        "expected": "BENE_STATE_CD",
    },
    {
        "query": "What ZIP code does the beneficiary live in?",
        "expected": "BENE_MLG_CNTCT_ZIP_CD",
    },
    {
        "query": "What is the principal diagnosis for the claim?",
        "expected": "PRNCPAL_DGNS_CD",
    },
    {
        "query": "What procedure was performed?",
        "expected": "ICD_PRCDR_CD25",
    },
    {
        "query": "What HCPCS code was billed?",
        "expected": "HCPCS_CD",
    },
    {
        "query": "What is the National Drug Code for this line?",
        "expected": "LINE_NDC_CD",
    },
    {
        "query": "Where was the service performed?",
        "expected": "LINE_PLACE_OF_SRVC_CD",
    },
    {
        "query": "How much coinsurance does the beneficiary owe for this line?",
        "expected": "LINE_COINSRNC_AMT",
    },
    {
        "query": "What is the NPI of the physician who rendered the service?",
        "expected": "RNDRNG_PHYSN_NPI",
    },
    {
        "query": "What is the specialty of the physician who rendered the service?",
        "expected": "RNDRNG_PHYSN_SPCLTY_CD",
    },
    {
        "query": "What is the NPI of the physician who referred the patient?",
        "expected": "RFR_PHYSN_NPI",
    },
    {
        "query": "What specialty does the doctor performing the operation have?",
        "expected": "OP_PHYSN_SPCLTY_CD",
    },
]


def evaluate_retrieval():
    """Evaluate Chroma retrieval against known CMS variables."""

    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0

    print("\n=== CMS Chroma Retrieval Evaluation ===\n")

    for test in TEST_QUERIES:
        query = test["query"]
        expected = test["expected"]

        results = retrieve(
            query,
            retrieve_function_top_k=5,
        )

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved_variables = [
            metadata["variable"]
            for metadata in metadatas
        ]

        if expected in retrieved_variables:
            rank = retrieved_variables.index(expected) + 1
        else:
            rank = None

        if rank == 1:
            top_1_correct += 1

        if rank is not None and rank <= 3:
            top_3_correct += 1

        if rank is not None and rank <= 5:
            top_5_correct += 1

        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(
            f"Found at rank: "
            f"{rank if rank is not None else 'Not in top 5'}"
        )
        print("Top 5:")

        for i, metadata in enumerate(
            metadatas,
            start=1,
        ):
            distance = distances[i - 1]

            print(
                f"  {i}. {metadata['variable']} "
                f"(distance: {distance:.3f})"
            )

        print()

    total = len(TEST_QUERIES)

    print("=== Evaluation Results ===\n")
    print(f"Test queries: {total}")
    print(
        f"Top-1 accuracy: "
        f"{top_1_correct / total:.1%}"
    )
    print(
        f"Top-3 accuracy: "
        f"{top_3_correct / total:.1%}"
    )
    print(
        f"Top-5 accuracy: "
        f"{top_5_correct / total:.1%}"
    )

def evaluate_reranker():
    """Evaluate retrieval + CrossEncoder reranking against known CMS variables."""

    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0

    print("\n=== CMS Reranker Evaluation ===\n")

    for test in TEST_QUERIES:
        query = test["query"]
        expected = test["expected"]

        reranked_candidates = rerank(
            query,
            rerank_function_top_k=5,
        )

        reranked_variables = [
            candidate["variable"]
            for candidate in reranked_candidates
        ]

        if expected in reranked_variables:
            rank = reranked_variables.index(expected) + 1
        else:
            rank = None

        if rank == 1:
            top_1_correct += 1

        if rank is not None and rank <= 3:
            top_3_correct += 1

        if rank is not None and rank <= 5:
            top_5_correct += 1

        print(f"Query: {query}")
        print(f"Expected: {expected}")
        print(
            f"Found at rank: "
            f"{rank if rank is not None else 'Not in top 5'}"
        )
        print("Reranked top 5:")

        for i, candidate in enumerate(
            reranked_candidates,
            start=1,
        ):
            print(
                f"  {i}. {candidate['variable']} "
                f"(reranker score: "
                f"{candidate['reranker_score']:.3f})"
            )

        print()

    total = len(TEST_QUERIES)

    print("=== Reranker Evaluation Results ===\n")
    print(f"Test queries: {total}")
    print(
        f"Top-1 accuracy: "
        f"{top_1_correct / total:.1%}"
    )
    print(
        f"Top-3 accuracy: "
        f"{top_3_correct / total:.1%}"
    )
    print(
        f"Top-5 accuracy: "
        f"{top_5_correct / total:.1%}"
    )


if __name__ == "__main__":
    evaluate_retrieval()
    evaluate_reranker()