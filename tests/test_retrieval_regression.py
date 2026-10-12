
import pytest

from src.retrieval.retrieval import retrieve_exact
from src.retrieval.evaluate_rrf import calculate_rrf
from src.retrieval import reranker


def test_beneficiary_race_retrieval():
    """Exact retrieval should rank BENE_RACE_CD first."""

    results = retrieve_exact(
        "What is the patient's race?",
        retrieve_function_top_k=20,
    )

    ids = results["ids"][0]

    assert ids, "Exact retrieval returned no candidates"

    assert ids[0] == "BENE_RACE_CD", (
        f"Expected BENE_RACE_CD at rank 1, "
        f"but received {ids[0]}"
    )

def test_reranker_uses_existing_candidates(monkeypatch):
    """Reranking must not trigger another Chroma retrieval."""

    chroma_results = {
        "documents": [[
            "Label: Beneficiary Race Code",
            "Label: Rendering Physician NPI",
        ]],
        "metadatas": [[
            {
                "variable": "BENE_RACE_CD",
                "label": "Beneficiary Race Code",
            },
            {
                "variable": "RNDRNG_PHYSN_NPI",
                "label": "Rendering Physician NPI",
            },
        ]],
    }

    def fail_if_retrieved(*args, **kwargs):
        pytest.fail("Unexpected duplicate Chroma retrieval")

    def fake_predict(pairs):
        assert len(pairs) == 2
        return [0.9, 0.2]

    monkeypatch.setattr(
        reranker,
        "retrieve",
        fail_if_retrieved,
    )

    monkeypatch.setattr(
        reranker.reranker_model,
        "predict",
        fake_predict,
    )

    candidates = reranker.rerank_candidates(
        "What is the patient's race?",
        chroma_results,
        top_k=2,
    )

    assert len(candidates) == 2
    assert candidates[0]["variable"] == "BENE_RACE_CD"
    assert candidates[1]["variable"] == "RNDRNG_PHYSN_NPI"


def test_rrf_combines_existing_rankings():
    """RRF should fuse the supplied rankings correctly."""

    chroma_ids = [
        "BENE_RACE_CD",
        "RNDRNG_PHYSN_NPI",
        "OP_PHYSN_NPI",
    ]

    reranked_ids = [
        "RNDRNG_PHYSN_NPI",
        "BENE_RACE_CD",
        "OP_PHYSN_NPI",
    ]

    fused_ids = calculate_rrf(
        chroma_ids,
        reranked_ids,
    )

    assert set(fused_ids) == set(chroma_ids)

    # Both variables have equal RRF scores.
    assert set(fused_ids[:2]) == {
        "BENE_RACE_CD",
        "RNDRNG_PHYSN_NPI",
    }

    assert fused_ids[2] == "OP_PHYSN_NPI"



def test_rag_evaluation_retrieves_once(monkeypatch):
    """Both RAG strategies must share one Chroma retrieval."""

    from src.rag import evaluate_rrf_rag

    calls = {"retrieve": 0, "generate": 0}

    fake_results = {
        "ids": [["BENE_RACE_CD"]],
        "documents": [["Label: Beneficiary Race Code"]],
        "metadatas": [[{
            "variable": "BENE_RACE_CD",
            "label": "Beneficiary Race Code",
        }]],
    }

    fake_candidates = [{
        "variable": "BENE_RACE_CD",
        "label": "Beneficiary Race Code",
        "document": "Label: Beneficiary Race Code",
        "reranker_score": 0.9,
    }]

    def fake_retrieve(query, retrieve_function_top_k=20):
        calls["retrieve"] += 1
        return fake_results

    def fake_rerank_candidates(query, results, top_k=20):
        assert results is fake_results
        return fake_candidates

    def fake_generate_answer(query, candidates):
        calls["generate"] += 1
        assert candidates == fake_candidates
        return type("Answer", (), {
            "variable": "BENE_RACE_CD"
        })()

    monkeypatch.setattr(
        evaluate_rrf_rag,
        "TEST_QUERIES",
        [{
            "query": "What is the patient's race?",
            "expected": "BENE_RACE_CD",
        }],
    )

    monkeypatch.setattr(
        evaluate_rrf_rag,
        "retrieve",
        fake_retrieve,
    )

    monkeypatch.setattr(
        evaluate_rrf_rag,
        "rerank_candidates",
        fake_rerank_candidates,
    )

    monkeypatch.setattr(
        evaluate_rrf_rag,
        "generate_answer",
        fake_generate_answer,
    )

    evaluate_rrf_rag.evaluate()

    assert calls["retrieve"] == 1
    assert calls["generate"] == 2

def test_main_rag_uses_exact_retrieval(monkeypatch):
    """Main RAG must use exact retrieval and reuse its candidates."""

    from src.rag import rag

    calls = {"exact": 0, "rerank": 0}

    fake_results = {
        "ids": [["BENE_RACE_CD"]],
        "documents": [["Label: Beneficiary Race Code"]],
        "metadatas": [[{
            "variable": "BENE_RACE_CD",
            "label": "Beneficiary Race Code",
        }]],
    }

    fake_candidates = [{
        "variable": "BENE_RACE_CD",
        "label": "Beneficiary Race Code",
        "document": "Label: Beneficiary Race Code",
        "reranker_score": 0.9,
    }]

    def fake_retrieve_exact(query, retrieve_function_top_k=20):
        calls["exact"] += 1
        assert query == "What is the patient's race?"
        assert retrieve_function_top_k == 20
        return fake_results

    def fake_rerank_candidates(query, results, top_k=5):
        calls["rerank"] += 1
        assert results is fake_results
        assert top_k == 5
        return fake_candidates

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

    candidates = rag.get_cms_context(
        "What is the patient's race?"
    )

    assert calls["exact"] == 1
    assert calls["rerank"] == 1
    assert candidates == fake_candidates