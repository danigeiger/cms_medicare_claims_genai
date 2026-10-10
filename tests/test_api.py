
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.rag.rag import CMSAnswer


# Create a test client for our FastAPI application
client = TestClient(app)


def test_health_check():
    """Verify that the API health endpoint works."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_cms_question():
    """Verify that the API returns a structured CMS answer."""

    # Create a fake response from our RAG pipeline
    mock_cms_answer = CMSAnswer(
        variable="BENE_RACE_CD",
        label="Beneficiary Race Code",
        explanation="Identifies the beneficiary's race code.",
    )

    # Replace the real RAG function during this test
    with patch(
        "src.api.main.generate_cms_answer",
        return_value=mock_cms_answer,
    ) as mock_generate_answer:

        response = client.post(
            "/ask",
            json={
                "query": "What is the patient's race?"
            },
        )

    # Verify that the API responded successfully
    assert response.status_code == 200

    # Verify the returned JSON
    response_data = response.json()

    assert response_data["variable"] == "BENE_RACE_CD"
    assert response_data["label"] == "Beneficiary Race Code"
    assert response_data["explanation"] == (
        "Identifies the beneficiary's race code."
    )

    # Verify that FastAPI passed the question to our RAG function
    mock_generate_answer.assert_called_once_with(
        "What is the patient's race?"
    )


def test_ask_missing_query():
    """Verify that FastAPI rejects a request without a query."""

    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422
