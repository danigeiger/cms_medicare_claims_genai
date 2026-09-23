from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cms_codebook"

def get_collection():
    """Connect to the persistent Chroma collection."""
    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection

def create_query_embedding(query: str, model: SentenceTransformer):
    """Convert a user's question into a normalized embedding vector."""
    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    return embedding


def retrieve(query: str, top_k: int = 5) -> dict:
    """Retrieve the most relevant CMS variables for a user query."""

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = create_query_embedding(
        query,
        model,
    )

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    return results


if __name__ == "__main__":
    query = "What is the specialty of the attending physician?"

    print("\n=== CMS Chroma Retrieval Test ===\n")
    print(f"Query: {query}\n")

    results = retrieve(
        query,
        top_k=5,
    )

    print("Top CMS variable matches:\n")

    for rank, metadata in enumerate(
        results["metadatas"][0],
        start=1,
    ):
        print(f"{rank}. {metadata['variable']}")
        print(f"   Label: {metadata['label']}")
        print(f"   Distance: {results['distances'][0][rank - 1]:.3f}")
        print()