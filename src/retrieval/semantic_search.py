import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT / "knowledge_base" / "processed" / "cms_codebook_variables.json"
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_knowledge_base(json_path: Path) -> list[dict]:
    """Load the structured CMS codebook records."""
    with json_path.open("r", encoding="utf-8") as file:
        variables = json.load(file)
    return variables


def create_embedding_text(variable: dict) -> str:
    """Create searchable text for a CMS variable."""
    return (
        f"Label: {variable['label']}\n"
        f"Label: {variable['label']}\n"
        f"Variable: {variable['variable']}\n"
        f"Description: {variable['description']}"
    )


def semantic_search(
    query: str,
    variables: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[tuple[dict, float]]:
    """Find the CMS variables most semantically similar to a query."""

    documents = [
        create_embedding_text(variable)
        for variable in variables
    ]

    document_embeddings = model.encode(
        documents,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    scores = util.cos_sim(
        query_embedding,
        document_embeddings,
    )[0]

    top_results = scores.topk(k=top_k)
    results = []

    for score, index in zip(
        top_results.values,
        top_results.indices,
    ):
        variable = variables[index.item()]
        results.append((variable, score.item()))

    return results


if __name__ == "__main__":
    variables = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    model = SentenceTransformer(MODEL_NAME)

    query = "What is the specialty of the attending physician?"

    results = semantic_search(
        query=query,
        variables=variables,
        model=model,
    )

    print(f"\nQuery: {query}")
    print("\nTop CMS variable matches:\n")

    for rank, (variable, score) in enumerate(results, start=1):
        print(f"{rank}. {variable['variable']}")
        print(f"   Label: {variable['label']}")
        print(f"   Similarity: {score:.3f}")
        print(f"   Description: {variable['description']}\n")