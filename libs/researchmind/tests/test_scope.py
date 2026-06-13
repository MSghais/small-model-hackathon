from researchmind.scope import (
    rag_scope_warning,
    resolve_retrieve_scope,
    retrieval_query,
)


def test_resolve_retrieve_scope_doc_ids():
    assert resolve_retrieve_scope("sess-1", ["d1", "d2"]) == (None, ["d1", "d2"])


def test_resolve_retrieve_scope_session():
    assert resolve_retrieve_scope("sess-1", None) == ("sess-1", None)
    assert resolve_retrieve_scope("sess-1", []) == ("sess-1", None)


def test_resolve_retrieve_scope_corpus():
    assert resolve_retrieve_scope(None, None) == (None, None)
    assert resolve_retrieve_scope("", None) == (None, None)


def test_retrieval_query_combines_topic():
    assert retrieval_query("How does it work?", topic="Photosynthesis") == (
        "Photosynthesis: How does it work?"
    )


def test_retrieval_query_skips_duplicate_topic():
    assert retrieval_query("Explain photosynthesis", topic="Photosynthesis") == (
        "Explain photosynthesis"
    )


def test_rag_scope_warning_messages():
    assert "selected documents" in rag_scope_warning(session_id="s", doc_ids=["d"])
    assert "this session" in rag_scope_warning(session_id="s", doc_ids=None)
    assert "corpus" in rag_scope_warning(session_id=None, doc_ids=None)
