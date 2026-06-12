"""
benchmarks/tau_bench.py
────────────────────────
τ-bench: Tool-Agent-User interaction benchmark.

What it tests: Multi-turn agentic loops where the model must call
retail/airline domain tools to satisfy a simulated user, and the
database state must exactly match ground truth at episode end.

Dataset: ShishirPatil/tau-bench on HF Hub, or local JSONL.

Scoring: Pass@1 task success (database state matches ground truth).

Note: Full τ-bench uses a live GPT-4o user simulator. This adapter
      uses a lightweight rule-based user simulator by default, which
      is deterministic, free, and reproducible for offline/local evals.
      Set cfg["use_llm_user"] = True to enable the LLM user (costs tokens).
"""

from __future__ import annotations
import json
import re
import copy
from typing import Any

from slm_evals.benchmarks.base import BaseBenchmark


AGENT_SYSTEM = """\
You are a helpful customer service agent.
You have access to the following tools. To call a tool, output ONLY:
TOOL_CALL: {"name": "<tool>", "args": {<key>: <value>}}

After receiving the tool result, continue the conversation.
When the task is complete, output: DONE
"""


class TauBenchmark(BaseBenchmark):
    """
    τ-bench multi-turn tool-agent-user benchmark.

    Config keys (benchmark_overrides.tau_bench):
        data_path   – local JSONL
        domain      – "retail" | "airline" | "both" (default: "retail")
        max_turns   – max dialogue turns per episode (default: 15)
        use_llm_user – use an LLM to simulate user (default: False)
    """

    name = "tau_bench"

    def load_dataset(self) -> list[dict]:
        data_path = self.cfg.get("data_path")
        if data_path:
            return self._load_local(data_path)
        return self._load_from_hub()

    def _load_local(self, path: str) -> list[dict]:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _load_from_hub(self) -> list[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        domain = self.cfg.get("domain", "retail")
        config = "retail" if domain in ("retail", "both") else "airline"
        ds = load_dataset("ShishirPatil/tau-bench", config, split="test", trust_remote_code=True)
        return list(ds)

    # ── Prompt is used only for the first turn ────────────────────────────────

    def build_prompt(self, sample: dict) -> str:
        tools_block = json.dumps(sample.get("tools", []), indent=2)
        context     = sample.get("context", "")
        user_goal   = sample.get("instruction", sample.get("user_goal", ""))
        return (
            f"{AGENT_SYSTEM}\n"
            f"Available tools:\n{tools_block}\n\n"
            f"Context:\n{context}\n\n"
            f"[User]: {user_goal}\n"
            f"[Agent]:"
        )

    # ── Multi-turn simulation ─────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Override run() to support multi-turn episodes."""
        dataset = self.load_dataset()
        if self.max_samples:
            dataset = dataset[: self.max_samples]

        import time
        samples_out = []
        errors = 0

        for sample in dataset:
            t0 = time.perf_counter()
            try:
                result = self._run_episode(sample)
            except Exception as exc:
                errors += 1
                result = {"passed": False, "score": 0.0, "note": f"ERROR: {exc}"}

            latency = round(time.perf_counter() - t0, 3)
            samples_out.append({"id": sample.get("id", ""), "latency_s": latency, **result})

        passed = sum(1 for s in samples_out if s["passed"])
        total  = len(samples_out)

        return {
            "benchmark":    self.name,
            "passed":       passed,
            "total":        total,
            "score":        (passed / total) if total else 0.0,
            "error_count":  errors,
            "avg_latency_s": round(
                sum(s["latency_s"] for s in samples_out) / total, 3
            ) if total else 0.0,
            "samples":      samples_out,
        }

    def _run_episode(self, sample: dict) -> dict:
        """Simulate one full τ-bench episode."""
        max_turns   = self.cfg.get("max_turns", 15)
        tools       = {t["name"]: t for t in sample.get("tools", [])}
        db_state    = copy.deepcopy(sample.get("initial_state", {}))
        ground_truth_state = sample.get("ground_truth_state", {})
        conversation: list[str] = []

        # First user message
        user_msg = sample.get("instruction", sample.get("user_goal", ""))
        conversation.append(f"[User]: {user_msg}")

        for _turn in range(max_turns):
            # Build prompt from conversation history
            tools_block = json.dumps(list(tools.values()), indent=2)
            history     = "\n".join(conversation)
            prompt = (
                f"{AGENT_SYSTEM}\n"
                f"Available tools:\n{tools_block}\n\n"
                f"{history}\n[Agent]:"
            )

            agent_reply = self.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            conversation.append(f"[Agent]: {agent_reply}")

            # ── Check for termination ─────────────────────────────────────────
            if "DONE" in agent_reply.upper():
                break

            # ── Check for tool call ───────────────────────────────────────────
            tool_result = self._try_execute_tool(agent_reply, tools, db_state)
            if tool_result is not None:
                conversation.append(f"[Tool]: {json.dumps(tool_result)}")
                # Simulated user follow-up
                conversation.append(self._simulated_user_reply(sample, db_state))

        # ── Score: does final db state match ground truth? ────────────────────
        state_match = self._compare_states(db_state, ground_truth_state)
        return {
            "passed":     state_match >= 1.0,
            "score":      round(state_match, 3),
            "note":       f"state_match={state_match:.2f}  turns={len(conversation)}",
            "prediction": "\n".join(conversation[-4:]),  # last 4 lines for logging
        }

    @staticmethod
    def _try_execute_tool(
        agent_reply: str,
        tools: dict,
        db_state: dict,
    ) -> dict | None:
        """
        Extract a TOOL_CALL from the agent reply and execute it
        against the in-memory db_state (simple rule-based simulation).
        """
        match = re.search(r"TOOL_CALL:\s*(\{.*?\})", agent_reply, re.DOTALL)
        if not match:
            return None

        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError:
            return {"error": "Malformed tool call JSON"}

        tool_name = call.get("name", "")
        args      = call.get("args", {})

        # ── Simple state-mutation rules ───────────────────────────────────────
        # Extend this section with real tool logic or mock implementations.
        if tool_name not in tools:
            return {"error": f"Unknown tool: {tool_name}"}

        # Generic: apply key-value updates from args to db_state
        for key, value in args.items():
            if key in db_state:
                db_state[key] = value

        return {"status": "ok", "tool": tool_name, "args": args}

    @staticmethod
    def _simulated_user_reply(sample: dict, db_state: dict) -> str:
        """Minimal rule-based user simulator (replace with LLM if cfg allows)."""
        # Real τ-bench uses GPT-4o here; we return a generic confirmation.
        return "[User]: Thank you, please continue."

    @staticmethod
    def _compare_states(actual: dict, expected: dict) -> float:
        """0–1 fraction of expected keys that match in actual state."""
        if not expected:
            return 1.0
        hits = sum(
            1 for k, v in expected.items()
            if str(actual.get(k, "")).strip() == str(v).strip()
        )
        return hits / len(expected)

    # ── Not used (multi-turn overrides run()) ─────────────────────────────────
    def evaluate_sample(self, sample, prediction):
        pass
