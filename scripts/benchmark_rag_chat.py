#!/usr/bin/env python3
"""Benchmark ResearchMind RAG retrieval and optional full chat latency."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from researchmind.config import get_config
from researchmind.embeddings import embed_texts
from researchmind.ingest import IngestPipeline
from researchmind.retrieve import retrieve


def _load_sessions() -> list[tuple[str, str]]:
    store = IngestPipeline().store
    return [(s.id, s.topic or "Untitled") for s in store.list_sessions()]


def benchmark_retrieve(
    question: str,
    *,
    session_id: str,
    runs: int,
) -> dict[str, object]:
    cfg = get_config()
    store = IngestPipeline().store
    chunks_in_scope = store.get_chunks_with_embeddings(session_id=session_id or None)
    timings: list[float] = []
    retrieved = 0
    for _ in range(runs):
        started = time.perf_counter()
        chunks = retrieve(question, store, config=cfg, session_id=session_id or None)
        timings.append((time.perf_counter() - started) * 1000)
        retrieved = len(chunks)

    warm = timings[1:] if len(timings) > 1 else timings

    embed_started = time.perf_counter()
    embed_texts(["warmup query"], model_name=cfg.embed_model)
    embed_warm_ms = (time.perf_counter() - embed_started) * 1000

    return {
        "question": question,
        "session_id": session_id,
        "chunks_in_scope": len(chunks_in_scope),
        "retrieved_chunks": retrieved,
        "top_k": cfg.top_k,
        "max_context_chunks": cfg.max_context_chunks,
        "embed_model": cfg.embed_model,
        "embedder_warm_ms": round(embed_warm_ms, 1),
        "retrieve_ms_cold": round(timings[0], 1) if timings else 0.0,
        "retrieve_ms_mean": round(statistics.mean(warm), 1),
        "retrieve_ms_stdev": round(statistics.stdev(warm), 1) if len(warm) > 1 else 0.0,
        "retrieve_ms_min": round(min(warm), 1),
        "retrieve_ms_max": round(max(warm), 1),
    }


def benchmark_chat(
    question: str,
    *,
    session_id: str,
    model_key: str | None,
) -> dict[str, object]:
    from agent.runner import AgentRunner
    from gradio_space.model_loading import ensure_model_loaded, get_active_model_key
    from inference.factory import get_backend

    key = model_key or get_active_model_key()
    load_err = ensure_model_loaded(key)
    if load_err:
        return {"error": load_err, "model": key}

    backend = get_backend(key)
    runner = AgentRunner()
    started = time.perf_counter()
    result = runner.run_researchmind_chat(
        question=question,
        session_id=session_id,
        model_key=key,
        backend=backend,
        doc_ids=None,
    )
    total_ms = (time.perf_counter() - started) * 1000
    trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
    steps = [
        {
            "name": step.get("name"),
            "label": step.get("label"),
            "duration_ms": step.get("duration_ms"),
        }
        for step in trace.get("steps", [])
        if step.get("type") == "step"
    ]
    return {
        "model": key,
        "question": question,
        "session_id": session_id,
        "total_ms": round(total_ms, 1),
        "citations": len(result.citations),
        "answer_preview": result.answer[:240],
        "steps": steps,
        "trace_path": result.trace_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ResearchMind RAG chat")
    parser.add_argument(
        "--question",
        default="how we can finetune model",
        help="Question to benchmark",
    )
    parser.add_argument("--session-id", default="", help="Research session id")
    parser.add_argument("--runs", type=int, default=5, help="Retrieve benchmark repetitions")
    parser.add_argument(
        "--full-chat",
        action="store_true",
        help="Also run one full RAG chat (loads local LLM)",
    )
    parser.add_argument("--model-key", default="", help="Override ACTIVE_MODEL preset")
    args = parser.parse_args()

    sessions = _load_sessions()
    session_id = args.session_id.strip()
    if not session_id:
        session_id = sessions[0][0] if sessions else ""

    if not session_id:
        print("No indexed session found. Ingest sources first.")
        return 1

    retrieve_report = benchmark_retrieve(
        args.question,
        session_id=session_id,
        runs=max(1, args.runs),
    )
    print(json.dumps({"retrieve": retrieve_report}, indent=2))

    if args.full_chat:
        chat_report = benchmark_chat(
            args.question,
            session_id=session_id,
            model_key=args.model_key or None,
        )
        print(json.dumps({"chat": chat_report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
