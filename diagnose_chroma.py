
from pathlib import Path
import chromadb

from src.retrieval.retrieval import (
    get_collection,
    create_query_embedding,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DB_PATH = PROJECT_ROOT / "chroma_diagnostic_db"

# Original database: existing index, ef_search=100
original = get_collection()

# Separate persistent diagnostic database
client = chromadb.PersistentClient(path=str(TEST_DB_PATH))

# Copy the original data, including its embeddings
data = original.get(
    include=["embeddings", "documents", "metadatas"]
)

# New index with ef_search=100
test_100 = client.get_or_create_collection(
    name="cms_codebook_diagnostic_100",
    configuration={
        "hnsw": {
            "space": "l2",
            "ef_search": 100,
        }
    },
)

# Previously created index with ef_search=300
test_300 = client.get_collection(
    name="cms_codebook_diagnostic"
)

# Populate only the new collection
if test_100.count() == 0:
    test_100.add(
        ids=data["ids"],
        embeddings=data["embeddings"].tolist(),
        documents=data["documents"],
        metadatas=data["metadatas"],
    )

query = create_query_embedding(
    "What is the patient's race?"
).tolist()

# Compare all three collections
collections = [
    ("Original (100)", original),
    ("Rebuilt (100)", test_100),
    ("Rebuilt (300)", test_300),
]

for name, collection in collections:
    results = collection.query(
        query_embeddings=[query],
        n_results=20,
    )

    ids = results["ids"][0]

    rank = (
        ids.index("BENE_RACE_CD") + 1
        if "BENE_RACE_CD" in ids
        else "MISSING"
    )

    print(f"{name}: BENE_RACE_CD rank = {rank}")
    print("Top 3:", ids[:3])
    print()

