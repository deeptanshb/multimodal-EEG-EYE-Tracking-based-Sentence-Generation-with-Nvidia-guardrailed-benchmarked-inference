# ── benchmark/nim_benchmark.py ────────────────────────────────────────────────
# Inference benchmark harness for the EEG agent pipeline
# Measures: TTFT, total latency, tokens/s, guardrail pass-rate per agent
# Works with both self-hosted NIM and api.nvidia.com endpoints
#
# Usage (standalone):
#   python nim_benchmark.py --endpoint http://localhost:8000/v1 --runs 10
#
# Or import and call run_agent_benchmark() from the notebook.

import time
import asyncio
import json
import argparse
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional
from openai import AsyncOpenAI


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class CallMetrics:
    agent:           str
    run_id:          int
    ttft_ms:         float        # time-to-first-token (ms)
    total_ms:        float        # wall-clock latency (ms)
    input_tokens:    int
    output_tokens:   int
    tokens_per_sec:  float
    guardrail_pass:  bool         # True if no rail fired
    guardrail_fired: str          # which rail fired, or ""
    error:           Optional[str] = None


@dataclass
class AgentBenchmarkReport:
    endpoint:        str
    model:           str
    agent_reports:   dict = field(default_factory=dict)  # agent -> list[CallMetrics]
    summary:         dict = field(default_factory=dict)

    def add(self, metrics: CallMetrics):
        self.agent_reports.setdefault(metrics.agent, []).append(metrics)

    def compute_summary(self):
        """Compute per-agent and overall summary statistics."""
        for agent, calls in self.agent_reports.items():
            successful = [c for c in calls if c.error is None]
            if not successful:
                continue
            self.summary[agent] = {
                "n_runs":            len(calls),
                "n_success":         len(successful),
                "guardrail_pass_pct": round(100 * sum(c.guardrail_pass for c in calls) / len(calls), 1),
                "ttft_ms_mean":      round(statistics.mean(c.ttft_ms for c in successful), 1),
                "ttft_ms_p95":       round(sorted(c.ttft_ms for c in successful)[int(0.95 * len(successful)) - 1], 1),
                "latency_ms_mean":   round(statistics.mean(c.total_ms for c in successful), 1),
                "latency_ms_p95":    round(sorted(c.total_ms for c in successful)[int(0.95 * len(successful)) - 1], 1),
                "tokens_per_sec_mean": round(statistics.mean(c.tokens_per_sec for c in successful), 1),
                "output_tokens_mean":  round(statistics.mean(c.output_tokens for c in successful), 1),
            }

        # Overall pipeline latency (sum of 3 agents, sequential)
        all_calls = [c for calls in self.agent_reports.values() for c in calls if c.error is None]
        if all_calls:
            # Group by run_id to get per-pipeline total
            from collections import defaultdict
            by_run = defaultdict(list)
            for c in all_calls:
                by_run[c.run_id].append(c.total_ms)

            pipeline_totals = [sum(v) for v in by_run.values()]
            self.summary["__pipeline__"] = {
                "total_pipeline_ms_mean": round(statistics.mean(pipeline_totals), 1),
                "total_pipeline_ms_p95":  round(sorted(pipeline_totals)[int(0.95 * len(pipeline_totals)) - 1], 1),
                "overall_guardrail_pass_pct": round(
                    100 * sum(c.guardrail_pass for c in all_calls) / len(all_calls), 1
                ),
            }

    def print_report(self):
        self.compute_summary()
        print("\n" + "=" * 68)
        print(f"  INFERENCE BENCHMARK REPORT")
        print(f"  Endpoint : {self.endpoint}")
        print(f"  Model    : {self.model}")
        print("=" * 68)

        for agent, stats in self.summary.items():
            if agent == "__pipeline__":
                continue
            print(f"\n── {agent.upper()} ─────────────────────────────────────")
            print(f"  Runs: {stats['n_runs']}  Success: {stats['n_success']}  "
                  f"Guardrail pass: {stats['guardrail_pass_pct']}%")
            print(f"  TTFT    mean={stats['ttft_ms_mean']}ms  p95={stats['ttft_ms_p95']}ms")
            print(f"  Latency mean={stats['latency_ms_mean']}ms  p95={stats['latency_ms_p95']}ms")
            print(f"  Tokens/s mean={stats['tokens_per_sec_mean']}  "
                  f"Output tokens mean={stats['output_tokens_mean']}")

        if "__pipeline__" in self.summary:
            p = self.summary["__pipeline__"]
            print(f"\n── FULL PIPELINE (3 agents sequential: scientist/critic/qml_synthesiser) ─")
            print(f"     NOTE: qml_synthesiser covers both clean QML and noisy QML analysis")
            print(f"  Total latency mean={p['total_pipeline_ms_mean']}ms  "
                  f"p95={p['total_pipeline_ms_p95']}ms")
            print(f"  Overall guardrail pass: {p['overall_guardrail_pass_pct']}%")
        print("=" * 68)

    def to_dict(self):
        self.compute_summary()
        return {
            "endpoint": self.endpoint,
            "model":    self.model,
            "summary":  self.summary,
            "calls":    {
                agent: [asdict(c) for c in calls]
                for agent, calls in self.agent_reports.items()
            },
        }


# ── Core benchmark runner ──────────────────────────────────────────────────────

class NIMBenchmark:
    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint = endpoint.rstrip("/")
        self.model    = model
        self.client   = AsyncOpenAI(base_url=endpoint, api_key=api_key)
        self.report   = AgentBenchmarkReport(endpoint=endpoint, model=model)

    async def _call_with_metrics(
        self,
        agent: str,
        run_id: int,
        system: str,
        user: str,
        max_tokens: int = 1800,
        guardrail_check_fn=None,
    ) -> CallMetrics:
        """
        Make one streaming NIM call and record:
        - TTFT (time to first token)
        - Total latency
        - Token counts + throughput
        - Guardrail pass/fail
        """
        t_start = time.perf_counter()
        ttft_ms = None
        output_tokens = 0
        full_text = []
        error = None
        guardrail_pass = True
        guardrail_fired = ""

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000

                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_text.append(delta)
                    output_tokens += 1  # approximate; use usage if available

            # If usage available in last chunk, use it
            if hasattr(stream, "usage") and stream.usage:
                output_tokens = stream.usage.completion_tokens

        except Exception as ex:
            error = str(ex)
            guardrail_pass = False
            guardrail_fired = f"error: {ex}"

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000
        ttft_ms = ttft_ms or total_ms

        response_text = "".join(full_text)
        tokens_per_sec = (output_tokens / (total_ms / 1000)) if total_ms > 0 else 0.0

        # Run Python-side guardrail checks
        if error is None and guardrail_check_fn is not None:
            pass_flag, fired = await guardrail_check_fn(response_text, agent)
            guardrail_pass  = pass_flag
            guardrail_fired = fired

        # Approximate input tokens: ~0.75 tokens per word
        input_tokens = int(len((system + user).split()) * 0.75)

        return CallMetrics(
            agent=agent,
            run_id=run_id,
            ttft_ms=round(ttft_ms, 1),
            total_ms=round(total_ms, 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_sec=round(tokens_per_sec, 1),
            guardrail_pass=guardrail_pass,
            guardrail_fired=guardrail_fired,
            error=error,
        )

    async def run_pipeline_once(
        self,
        run_id: int,
        scientist_system: str,
        critic_system: str,
        qml_system: str,
        user_payload: str,
        guardrail_check_fn=None,
    ) -> dict:
        """Run all 3 agents sequentially (matches real pipeline flow) and record metrics."""

        sci_metrics = await self._call_with_metrics(
            "scientist", run_id, scientist_system,
            user_payload, guardrail_check_fn=guardrail_check_fn
        )
        self.report.add(sci_metrics)
        sci_response = ""  # would decode from stream if needed

        critic_user = f"SCIENTIST ANALYSIS:\n{sci_response}\n\nKEY NUMBERS:\n{user_payload[:500]}"
        crit_metrics = await self._call_with_metrics(
            "critic", run_id, critic_system,
            critic_user, guardrail_check_fn=guardrail_check_fn
        )
        self.report.add(crit_metrics)

        qml_user = f"SCIENTIST: [prior]\nCRITIC: [prior]\n\n{user_payload[:300]}"
        qml_metrics = await self._call_with_metrics(
            "qml_synthesiser", run_id, qml_system,
            qml_user, guardrail_check_fn=guardrail_check_fn
        )
        self.report.add(qml_metrics)

        return {
            "scientist":      sci_metrics,
            "critic":         crit_metrics,
            "qml_synthesiser": qml_metrics,
        }

    async def run_benchmark(
        self,
        scientist_system: str,
        critic_system: str,
        qml_system: str,
        user_payload: str,
        n_runs: int = 5,
        guardrail_check_fn=None,
    ) -> AgentBenchmarkReport:
        """Run the full 3-agent pipeline n_runs times and return the report."""
        print(f"\n  Benchmarking {n_runs} pipeline runs on {self.endpoint}...")
        for run_id in range(1, n_runs + 1):
            print(f"  Run {run_id}/{n_runs}...", end=" ", flush=True)
            await self.run_pipeline_once(
                run_id, scientist_system, critic_system, qml_system,
                user_payload, guardrail_check_fn
            )
            print("done")

        self.report.print_report()
        return self.report


# ── Guardrail check function (Python-side, mirrors actions.py) ─────────────────

async def default_guardrail_check(response: str, agent: str) -> tuple[bool, str]:
    """
    Fast Python-side check (no LLM call) that mirrors the Colang output rails.
    Returns (pass: bool, fired_rail: str)
    """
    import re
    text = response.lower()

    # 1. Metric bounds check
    # Require explicit separator + decimal-or-2digit value to avoid
    # matching the digit suffix in "BLEU-1" as the metric value (= 1.0%)
    pattern = r'(bleu[-_]?[14]?|rouge[-_]?[1l]?|bertscore)[\s:=\(of]+(\d{1,3}\.\d+|\d{2,3}(?!\.))'
    RANGES  = {
        "bleu-1": (20.0, 55.0),
        "bleu-4": (1.0,  15.0),
        "bleu_4": (1.0,  15.0),
        "bleu":   (20.0, 55.0),
        "rouge":  (22.0, 58.0),
        "bertscore": (78.0, 96.5),
    }
    for metric_raw, value_str in re.findall(pattern, text):
        try:
            value = float(value_str)
        except ValueError:
            continue
        if "bert" in metric_raw:
            key = "bertscore"
        elif "rouge" in metric_raw:
            key = "rouge"
        elif "4" in metric_raw:
            key = "bleu-4"
        else:
            key = "bleu-1" if "1" in metric_raw else "bleu"
        lo, hi = RANGES[key]
        if value < lo or value > hi:
            return False, f"metric_out_of_range:{metric_raw}={value}%"

    # 2. Domain relevance
    domain_terms = ["eeg","zuco","bleu","rouge","bertscore","attention","encoder",
                    "transformer","gpt","lora","htp","qml","quantum","neuroscience",
                    "region","temporal","condition","val","baseline","spectral",
                    "decoding","moco","stage","checkpoint",
                    # noisy QML vocabulary
                    "noisy","noise","depolarizing","phase damping","decoherence",
                    "hardware","monte carlo","circuit","fidelity","gate error",
                    "noiseless","clean qml","noisy qml","mc average"]
    found = sum(1 for t in domain_terms if t in text)
    if found < 3:
        return False, f"low_domain_relevance:found={found}"

    # 3. Agent-specific scope
    if agent == "qml_synthesiser":
        qml_terms = ["quantum","qml","vqc","qubit","pennylane","circuit","classical","residual",
                     "noisy","depolarizing","phase damping","monte carlo","noise","hardware"]
        if not any(t in text for t in qml_terms):
            return False, "qml_scope:no_quantum_terms"

    return True, ""


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG agent pipeline NIM benchmark")
    parser.add_argument("--endpoint", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--api-key",  default="", help="NVIDIA API key")
    parser.add_argument("--model",    default="meta/llama-3.1-70b-instruct")
    parser.add_argument("--runs",     type=int, default=3)
    parser.add_argument("--output",   default="benchmark_report.json")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print("ERROR: set --api-key or NVIDIA_API_KEY env var"); exit(1)

    # Minimal test payload
    test_payload = json.dumps({
        "live_metrics": {
            "n": 2032,
            "v9_tf_bleu1_pct": 31.02,
            "v9_tf_rouge1_pct": 36.07,
            "qml_tf_bleu1_pct": 31.00,
            "delta_v9_vs_v8_bleu1": 0.62,
            "delta_qml_vs_v9_bleu1": -0.02,
            "delta_qml_vs_v8_bleu1": 0.60,
            # noisy QML keys (required by check_noisy_qml_keys guardrail)
            "noisy_qml_tf_bleu1_pct": 31.00,
            "noisy_qml_tf_rouge1_pct": 36.05,
            "delta_noisy_vs_clean_bleu1": 0.00,
            "delta_noisy_vs_v8_bleu1": 0.60,
        },
        "attention_analysis": {
            "v9_qml_noisy_hybrid": {
                "cross_region_fusion": {
                    "values": {}, "dominant": "Left_Temporal"
                }
            }
        }
    }, indent=2)

    # QML_SYSTEM prompt now covers both clean and noisy QML variants
    from nat_agents_guardrailed import SCIENTIST_SYSTEM, CRITIC_SYSTEM, QML_SYSTEM

    bm = NIMBenchmark(endpoint=args.endpoint, api_key=api_key, model=args.model)

    async def main():
        report = await bm.run_benchmark(
            SCIENTIST_SYSTEM, CRITIC_SYSTEM, QML_SYSTEM,
            test_payload, n_runs=args.runs,
            guardrail_check_fn=default_guardrail_check,
        )
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n✅ Benchmark saved → {args.output}")

    asyncio.run(main())
