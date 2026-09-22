from sentence_transformers import SentenceTransformer

from src.retrieval.semantic_search import (
    KNOWLEDGE_BASE_PATH,
    MODEL_NAME,
    load_knowledge_base,
    semantic_search,
)


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


def evaluate_retrieval(
    test_queries: list[dict],
    variables: list[dict],
    model: SentenceTransformer,
) -> None:
    """Evaluate semantic retrieval using known query-variable pairs."""

    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0

    print("\n=== CMS Retrieval Evaluation ===\n")

    for test in test_queries:
        query = test["query"]
        expected = test["expected"]

        results = semantic_search(
            query=query,
            variables=variables,
            model=model,
            top_k=5,
        )

        retrieved_variables = [
            variable["variable"]
            for variable, score in results
        ]

        print(f"Query: {query}")
        print(f"Expected: {expected}")

        if expected in retrieved_variables:
            rank = retrieved_variables.index(expected) + 1
            print(f"Found at rank: {rank}")
        else:
            rank = None
            print("Found at rank: Not in top 5")

        print("Top 5:")
        for i, (variable, score) in enumerate(results, start=1):
            print(
                f"  {i}. {variable['variable']} "
                f"({score:.3f})"
            )

        print()

        if rank == 1:
            top_1_correct += 1

        if rank is not None and rank <= 3:
            top_3_correct += 1

        if rank is not None and rank <= 5:
            top_5_correct += 1

    total = len(test_queries)

    top_1_accuracy = top_1_correct / total
    top_3_accuracy = top_3_correct / total
    top_5_accuracy = top_5_correct / total

    print("=== Evaluation Results ===\n")
    print(f"Test queries: {total}")
    print(f"Top-1 accuracy: {top_1_accuracy:.1%}")
    print(f"Top-3 accuracy: {top_3_accuracy:.1%}")
    print(f"Top-5 accuracy: {top_5_accuracy:.1%}")


if __name__ == "__main__":
    variables = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    model = SentenceTransformer(MODEL_NAME)

    evaluate_retrieval(
        test_queries=TEST_QUERIES,
        variables=variables,
        model=model,
    )