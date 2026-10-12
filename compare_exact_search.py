
import numpy as np

from src.retrieval.retrieval import (
    get_collection,
    create_query_embedding,
)

QUESTION = "What is the patient's race?"
EXPECTED = "BENE_RACE_CD"
TOP_K = 20


def rank_of(variable, ids):
    return (
        ids.index(variable) + 1
        if variable in ids
        else "MISSING"
    )


# Open the original collection
collection = get_collection()

# Generate the query embedding once
query = np.asarray(
    create_query_embedding(QUESTION),
    dtype=np.float32,
)

# TEST 1: Query Chroma BEFORE loading stored embeddings
before = collection.query(
    query_embeddings=[query.tolist()],
    n_results=TOP_K,
)

before_ids = before["ids"][0]

# TEST 2: Load embeddings and calculate exact distances
data = collection.get(include=["embeddings"])

ids = data["ids"]
embeddings = np.asarray(
    data["embeddings"],
    dtype=np.float32,
)

distances = np.sum(
    (embeddings - query) ** 2,
    axis=1,
)

exact_indices = np.argsort(distances)
exact_ids = [ids[i] for i in exact_indices[:TOP_K]]

# TEST 3: Query Chroma AFTER loading stored embeddings
after = collection.query(
    query_embeddings=[query.tolist()],
    n_results=TOP_K,
)

after_ids = after["ids"][0]

# Display results
print("\nQUESTION:", QUESTION)
print("Collection size:", len(ids))

print("\n1. CHROMA BEFORE LOADING EMBEDDINGS")
print("Rank:", rank_of(EXPECTED, before_ids))
print("Top 5:", before_ids[:5])

print("\n2. EXACT NUMPY SEARCH")
print("Rank:", rank_of(EXPECTED, exact_ids))
print("Top 5:", exact_ids[:5])

print("\n3. CHROMA AFTER LOADING EMBEDDINGS")
print("Rank:", rank_of(EXPECTED, after_ids))
print("Top 5:", after_ids[:5])

print("\nCOMPARISON")
print("Before matches exact:", before_ids == exact_ids)
print("After matches exact:", after_ids == exact_ids)
print("Before equals after:", before_ids == after_ids)

if before_ids != after_ids:
    print("RESULT: CHROMA RESULTS CHANGED")
elif before_ids != exact_ids:
    print("RESULT: CHROMA DISAGREES WITH EXACT SEARCH")
else:
    print("RESULT: ALL THREE SEARCHES AGREE")

