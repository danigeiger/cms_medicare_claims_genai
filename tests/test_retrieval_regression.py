
"""
Regression tests for the CMS Medicare Claims GenAI project.

Tests:
1. Exact retrieval identifies the beneficiary race variable.
2. CrossEncoder reranking reuses existing retrieval candidates.
3. Reciprocal Rank Fusion combines existing rankings.
4. RAG evaluation retrieves candidates only once.
5. Main RAG pipeline uses exact retrieval, CrossEncoder, and RRF.

Run:
    python -m pytest tests/test_retrieval_regression.py -v
"""

from unittest.mock import MagicMock

from src.retrieval.retrieval import retrieve_exact
from src.retrieval import reranker
from src.retrieval.evaluate_rrf import calculate_rrf


def test_beneficiary_race_retrieval():
    """
    Verify that exact retrieval finds the correct CMS variable.
    """

    results = retrieve_exact(
        "What is the patient's race?",
        retrieve_function_top_k=20,
    )

    assert results["ids"][0][0] == "BENE_RACE_CD"


def test_reranker_uses_existing_candidates(monkeypatch):
    """
    Verify that reranking uses supplied candidates rather than
    performing another retrieval.
    """

    fake_results = {
        "ids": [["BENE_RACE_CD", "DOB_DT"]],
        "documents": [[
            (
                "Label: Beneficiary Race Code\n"
                "Variable: BENE_RACE_CD\n"
                "Description: Race of the beneficiary."
            ),
            (
                "Label: Date of Birth\n"
                "Variable: DOB_DT\n"
                "Description: Beneficiary date of birth."
            ),
        ]],
        "metadatas": [[
            {
                "variable": "BENE_RACE_CD",
                "label": "Beneficiary Race Code",
            },
            {
                "variable": "DOB_DT",
                "label": "Date of Birth",
            },
        ]],
    }

    class FakeCrossEncoder:
        def predict(self, pairs, **kwargs):
            assert len(pairs) == 2
            return [0.9, 0.2]

    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        lambda: FakeCrossEncoder(),
        raising=False,
    )

    # Prevent accidental retrieval calls during reranking.
    def forbidden_retrieval(*args, **kwargs):
        raise AssertionError(
            "Reranker must not perform another retrieval."
        )

    monkeypatch.setattr(
        reranker,
        "retrieve",
        forbidden_retrieval,
        raising=False,
    )

    candidates = reranker.rerank_candidates(
        "What is the patient's race?",
        fake_results,
        top_k=2,
    )

    assert len(candidates) == 2
    assert candidates[0]["variable"] == "BENE_RACE_CD"
    assert candidates[1]["variable"] == "DOB_DT"


def test_rrf_combines_existing_rankings():
    """
    Verify that RRF combines the two supplied rankings.
    """

    retrieval_ids = [
        "BENE_RACE_CD",
        "DOB_DT",
        "CLM_PMT_AMT",
    ]

    reranked_ids = [
        "DOB_DT",
        "BENE_RACE_CD",
        "CLM_PMT_AMT",
    ]

    combined = calculate_rrf(
        retrieval_ids,
        reranked_ids,
    )

    assert set(combined) == set(retrieval_ids)
    assert combined[-1] == "CLM_PMT_AMT"
    assert set(combined[:2]) == {
        "BENE_RACE_CD",
        "DOB_DT",
    }


def test_rag_evaluation_retrieves_once(monkeypatch):
    """
    Verify that a RAG evaluation reuses retrieval results
    instead of running retrieval repeatedly.
    """

    from src.rag import rag

    calls = {"context": 0, "api": 0}

    fake_candidates = [{
        "variable": "BENE_RACE_CD",
        "label": "Beneficiary Race Code",
        "document": (
            "Label: Beneficiary Race Code\n"
            "Variable: BENE_RACE_CD\n"
            "Description: Race of the beneficiary."
        ),
        "reranker_score": 0.9,
    }]

    def fake_get_cms_context(query):
        calls["context"] += 1
        assert query == "What is the patient's race?"
        return fake_candidates

    class FakeResponses:
        def parse(self, **kwargs):
            calls["api"] += 1

            answer = rag.CMSAnswer(
                variable="BENE_RACE_CD",
                label="Beneficiary Race Code",
                explanation="Identifies the beneficiary's race.",
            )

            return MagicMock(output_parsed=answer)

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = FakeResponses()

    monkeypatch.setattr(
        rag,
        "get_cms_context",
        fake_get_cms_context,
    )

    monkeypatch.setattr(
        rag,
        "OpenAI",
        FakeOpenAI,
    )

    answer = rag.generate_cms_answer(
        "What is the patient's race?"
    )

    assert answer.variable == "BENE_RACE_CD"
    assert calls == {
        "context": 1,
        "api": 1,
    }


def test_main_rag_uses_exact_retrieval(monkeypatch):
    """
    Verify that the main RAG pipeline:
    - Retrieves candidates exactly once.
    - Reranks the existing candidates.
    - Combines rankings with RRF.
    - Returns candidates in RRF order.
    """

    from src.rag import rag

    calls = {
        "exact": 0,
        "rerank": 0,
        "rrf": 0,
    }

    fake_results = {
        "ids": [["BENE_RACE_CD", "DOB_DT"]],
        "documents": [[
            "Label: Beneficiary Race Code",
            "Label: Date of Birth",
        ]],
        "metadatas": [[
            {
                "variable": "BENE_RACE_CD",
                "label": "Beneficiary Race Code",
            },
            {
                "variable": "DOB_DT",
                "label": "Date of Birth",
            },
        ]],
    }

    fake_candidates = [
        {
            "variable": "DOB_DT",
            "label": "Date of Birth",
            "document": "Label: Date of Birth",
            "reranker_score": 0.9,
        },
        {
            "variable": "BENE_RACE_CD",
            "label": "Beneficiary Race Code",
            "document": "Label: Beneficiary Race Code",
            "reranker_score": 0.8,
        },
    ]

    def fake_retrieve_exact(
        query,
        retrieve_function_top_k=20,
    ):
        calls["exact"] += 1

        assert query == "What is the patient's race?"
        assert retrieve_function_top_k == 20

        return fake_results

    def fake_rerank_candidates(
        query,
        results,
        top_k=20,
    ):
        calls["rerank"] += 1

        assert query == "What is the patient's race?"
        assert results is fake_results
        assert top_k == 20

        return fake_candidates

    def fake_calculate_rrf(
        retrieval_ids,
        reranked_ids,
    ):
        calls["rrf"] += 1

        assert retrieval_ids == [
            "BENE_RACE_CD",
            "DOB_DT",
        ]

        assert reranked_ids == [
            "DOB_DT",
            "BENE_RACE_CD",
        ]

        return [
            "BENE_RACE_CD",
            "DOB_DT",
        ]

    monkeypatch.setattr(
        rag,
        "retrieve_exact",
        fake_retrieve_exact,
    )

    monkeypatch.setattr(
        rag,
        "rerank_candidates",
        fake_rerank_candidates,
    )

    monkeypatch.setattr(
        rag,
        "calculate_rrf",
        fake_calculate_rrf,
    )

    candidates = rag.get_cms_context(
        "What is the patient's race?"
    )

    assert [
        candidate["variable"]
        for candidate in candidates
    ] == [
        "BENE_RACE_CD",
        "DOB_DT",
    ]

    assert calls == {
        "exact": 1,
        "rerank": 1,
        "rrf": 1,
    }
