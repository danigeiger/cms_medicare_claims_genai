import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT
    / "knowledge_base"
    / "processed"
    / "cms_codebook_variables.json"
)

VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cms_codebook"

def load_knowledge_base(json_path: Path) -> list[dict]:
    """Load the structured CMS codebook records from JSON."""
    with json_path.open("r", encoding="utf-8") as file:
        variables = json.load(file)

    return variables


def create_embedding_text(variable: dict) -> str:
    """Create the text that will be converted into an embedding."""
    return (
        f"Label: {variable['label']}\n"
        f"Label: {variable['label']}\n"
        f"Variable: {variable['variable']}\n"
        f"Description: {variable['description']}"
    )


if __name__ == "__main__":
    variables = load_knowledge_base(KNOWLEDGE_BASE_PATH)

    print("\n=== Building CMS Vector Store ===\n")
    print(f"CMS variables loaded: {len(variables)}")

    documents = [
        create_embedding_text(variable)
        for variable in variables
    ]

    print(f"Embedding documents created: {len(documents)}")

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
    )

    print(f"Embeddings created: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")


print("\nCreating Chroma vector store...")

client = chromadb.PersistentClient(
    path=str(VECTOR_STORE_PATH)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


ids = [
    variable["variable"]
    for variable in variables
]

metadatas = [
    {
        "variable": variable["variable"],
        "label": variable["label"],
    }
    for variable in variables
]


collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
)

print(f"Records stored in Chroma: {collection.count()}")
print(f"Vector store saved to: {VECTOR_STORE_PATH}")
