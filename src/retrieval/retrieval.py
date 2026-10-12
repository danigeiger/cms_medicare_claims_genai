
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CMS_CODEBOOK_VECTORS_PATH = PROJECT_ROOT / "cms_codebook_vectors"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_collection():
    """Connect to the persistent Chroma collection."""
    client = chromadb.PersistentClient(
        path=str(CMS_CODEBOOK_VECTORS_PATH)
    )

    collection = client.get_collection(
        name="cms_codebook"
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
    """Retrieve CMS variables using Chroma's indexed search."""

    query_embedding = create_query_embedding(query)
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=retrieve_function_top_k,
    )

    return results


def retrieve_exact(query: str, retrieve_function_top_k: int = 5) -> dict:
    """Retrieve CMS variables using exact NumPy L2 distance."""

    query_embedding = np.asarray(
        create_query_embedding(query),
        dtype=np.float32,
    )

    collection = get_collection()

    # Load every stored embedding and its associated information
    data = collection.get(
        include=["embeddings", "documents", "metadatas"]
    )

    embeddings = np.asarray(
        data["embeddings"],
        dtype=np.float32,
    )

    # Compute squared Euclidean distance to every stored vector
    distances = np.sum(
        (embeddings - query_embedding) ** 2,
        axis=1,
    )

    # Sort from closest to farthest
    top_k = min(retrieve_function_top_k, len(distances))
    top_indices = np.argsort(distances)[:top_k]

    # Return the same structure as collection.query()
    results = {
        "ids": [[data["ids"][i] for i in top_indices]],
        "documents": [[data["documents"][i] for i in top_indices]],
        "metadatas": [[data["metadatas"][i] for i in top_indices]],
        "distances": [[float(distances[i]) for i in top_indices]],
    }

    return results


if __name__ == "__main__":
    query = "What is the specialty of the attending physician?"

    print("\n=== CMS Retrieval Comparison ===\n")
    print(f"Query: {query}\n")

    for method_name, retrieval_function in [
        ("Chroma", retrieve),
        ("Exact NumPy", retrieve_exact),
    ]:
        results = retrieval_function(
            query,
            retrieve_function_top_k=5,
        )

        print(f"\n{method_name} results:\n")

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
