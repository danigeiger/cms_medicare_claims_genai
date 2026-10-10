from sentence_transformers import CrossEncoder

from src.retrieval.retrieval import retrieve


reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(
    query: str,
    rerank_function_top_k: int = 5,
) -> list[dict]:
    """Retrieve CMS candidates and rerank them with a cross-encoder."""

    chroma_results = retrieve(
        query,
        retrieve_function_top_k=20,
    )

    retrieved_cms_documents = chroma_results["documents"][0]
    retrieved_cms_metadata = chroma_results["metadatas"][0]

    query_document_pairs = [
        [query, cms_document]
        for cms_document in retrieved_cms_documents
    ]

    reranker_scores = reranker_model.predict(
        query_document_pairs
    )

    reranked_cms_candidates = [
        {
            "variable": metadata["variable"],
            "label": metadata["label"],
            "document": cms_document,
            "reranker_score": float(reranker_score),
        }
        for metadata, cms_document, reranker_score in zip(
            retrieved_cms_metadata,
            retrieved_cms_documents,
            reranker_scores,
        )
    ]

    reranked_cms_candidates.sort(
        key=lambda candidate: candidate["reranker_score"],
        reverse=True,
    )

    return reranked_cms_candidates[:rerank_function_top_k]


if __name__ == "__main__":
    query = "What is the specialty of the attending physician?"

    print("\n=== CMS Reranker Test ===\n")
    print(f"Query: {query}\n")

    reranked_cms_candidates = rerank(
        query,
        rerank_function_top_k=5,
    )

    print("Reranked CMS variable matches:\n")

    for rank, candidate in enumerate(
        reranked_cms_candidates,
        start=1,
    ):
        print(f"{rank}. {candidate['variable']}")
        print(f"   Label: {candidate['label']}")
        print(
            f"   Reranker score: "
            f"{candidate['reranker_score']:.3f}"
        )
        print()