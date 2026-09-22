import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_PATH = (PROJECT_ROOT/"knowledge_base"/"processed"/"cms_codebook_variables.json")



# Load the CMS knowledge base

def load_knowledge_base(json_path: Path) -> list[dict]:
    """Load the structured CMS codebook records."""

    with json_path.open("r", encoding="utf-8") as file:
        variables = json.load(file)

    return variables



# Build text for embedding

def create_embedding_text(variable: dict) -> str:
    """
    Create the text representation of a CMS variable
    that will be sent to the embedding model.
    """

    return (
        f"Variable: {variable['variable']}\n"
        f"Label: {variable['label']}\n"
        f"Description: {variable['description']}"
    )



# Main

if __name__ == "__main__":

    print("\n=== CMS Embedding Test ===\n")

    # Load our 302 structured CMS records
    variables = load_knowledge_base(KNOWLEDGE_BASE_PATH)

    print(f"CMS variables loaded: {len(variables)}")

    # We'll test only ONE variable first.
    example = variables[0]

    embedding_text = create_embedding_text(example)

    print("\nText being embedded:\n")
    print(embedding_text)

    # Load a small Hugging Face embedding model.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Convert the CMS text into an embedding vector.
    embedding = model.encode(embedding_text)

    print(f"\nEmbedding dimensions: "f"{len(embedding)}")

    print("\nFirst 10 numbers in the embedding:\n")

    print(embedding[:10])