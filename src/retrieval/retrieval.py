from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CMS_CODEBOOK_VECTORS_PATH = (PROJECT_ROOT / "cms_codebook_vectors")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_collection():
    """Connect to the persistent Chroma collection."""
    client = chromadb.PersistentClient(
        path=str(CMS_CODEBOOK_VECTORS_PATH)
    )

    collection = client.get_collection(
        name= "cms_codebook"
    )

    return collection


def create_query_embedding(query: str):
    """Convert a user's question into a normalized embedding vector."""
    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    return embedding


def retrieve(query: str, retrieve_function_top_k: int = 5) -> dict:
    """Retrieve the most relevant CMS variables for a user query."""

    query_embedding = create_query_embedding(query)

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=retrieve_function_top_k,
    )

    return results


if __name__ == "__main__":
    query = "What is the specialty of the attending physician?"

    print("\n=== CMS Chroma Retrieval Test ===\n")
    print(f"Query: {query}\n")

    results = retrieve(
        query,
        retrieve_function_top_k=5,
    )

    print("Top CMS variable matches:\n")

    for rank, metadata in enumerate(
        results["metadatas"][0],
        start=1,
    ):
        print(f"{rank}. {metadata['variable']}")
        print(f"   Label: {metadata['label']}")
        print(
            f"   Distance: "
            f"{results['distances'][0][rank - 1]:.3f}"
        )
        print()