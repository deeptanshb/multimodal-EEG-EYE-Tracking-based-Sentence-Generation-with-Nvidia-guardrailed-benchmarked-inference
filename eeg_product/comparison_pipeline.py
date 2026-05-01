# ── comparison_pipeline.py ───────────────────────────────────────────────────
# 4-agent comparison pipeline for external researchers.
# Runs when someone submits their EEG model via EEGModelSubmission.
#
# Agents:
#   1. Scientist    — analyses submitted model on its own merits
#   2. Comparator   — head-to-head vs V9+QML with per-metric verdict
#   3. Critic       — challenges both the submission AND the comparison
#   4. Synthesiser  — plain-language summary with "what to do next"
#
# Usage:
#   from comparison_pipeline import run_comparison_pipeline
#   from eeg_submission_schema import EEGModelSubmission
#
#   submission = EEGModelSubmission(model_name="MyModel", tf_bleu1_pct=32.1, ...)
#   results = await run_comparison_pipeline(submission)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Optional

# Path setup
sys.path.insert(0, str(Path(__file__).parent))
from eeg_submission_schema import (
    EEGModelSubmission,
    V5_BASELINE, V8_BASELINE,
    load_v9_qml_baseline,
    load_v9_qml_noisy_baseline,
    V9_QML_NOISY_BASELINE,
)

try:
    from nat_agents_guardrailed import (
        call_nim_guardrailed, _load_rails,
        NVIDIA_API_KEY, NIM_BASE_URL, NIM_MODEL,
    )
    _NAG_AVAILABLE = True
except ImportError:
    _NAG_AVAILABLE = False
    print("⚠  nat_agents_guardrailed not found. Make sure eeg_product/ is in your path.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON AGENT SYSTEM PROMPTS
# These are separate from the original 3 agents — they accept a submitted model
# as the subject and compare it explicitly against V9+QML.
# ─────────────────────────────────────────────────────────────────────────────

def build_scientist_prompt(submission: EEGModelSubmission) -> str:
    return f"""
[ROLE: scientist]
You are a neuroscience and NLP researcher reviewing a submitted EEG-to-text model on ZuCo.

The submitted model is: {submission.model_name}
Architecture: {submission.architecture_desc}

The known baseline chain for ZuCo EEG-to-text decoding is:
  V5  → Conv1D + Bi-GRU + single mean-pool EEG vector, prefix-tuned DistilGPT2
        TF BLEU-1=29.24%  ROUGE-1=33.92%  per-condition NR/TSR/SR=30.70/32.78/26.49
  V8  → 6 parallel GRU-Transformer RegionEncoders, MoCo Stage0, LoRA rank=8 GPT2, SR adapter
        TF BLEU-1=30.40%  ROUGE-1=35.78%  BERTScore=85.46%  per-condition NR/TSR/SR=30.90/32.93/27.20
  V9+QML clean  → V8 + HierarchicalTemporalPooling + QuantumFusionProjector (4-qubit noiseless VQC)
                  LoRA rank=4 alpha=8.0 block=[11]; best val loss=4.1733
  V9+QML noisy  → same + DepolarizingChannel(p=0.01) + PhaseDamping(γ=0.02) + 16-pass MC inference
                  Hardware-realistic noise simulation; best val loss=4.1729
                  Values for both variants provided in submitted_data below.

Your task:
1. Analyse the submitted model's architecture — what does it add or change vs the known chain?
2. Evaluate its TF metrics in the context of the ZuCo benchmark
3. Assess per-condition performance — does it handle SR (speed reading) better or worse?
4. Comment on the TF/FG ratio if provided — what does it say about EEG conditioning strength?
5. Note any missing metrics and what they would tell us
6. Give 3 specific strengths and 2 specific weaknesses of this submission

Stay strictly within EEG-to-text neuroscience and NLP evaluation. Do not discuss unrelated topics.
""".strip()


def build_comparator_prompt(submission: EEGModelSubmission) -> str:
    return f"""
[ROLE: comparator]
You are a benchmark specialist producing a head-to-head comparison report.

Your job: compare {submission.model_name} vs V9+QML clean AND V9+QML noisy on ZuCo.
V9+QML clean  = noiseless statevector simulation (lightning.qubit)
V9+QML noisy  = hardware-realistic noise simulation (DepolarizingChannel + PhaseDamping + MC average)
Both are valid reference points. The noisy model is the hardware-deployable variant.

Comparison format — for EACH metric, give exactly:
  [METRIC NAME]
  Submitted: X%    V9+QML: Y%    Delta: ±Z pp    Verdict: BETTER / WORSE / EQUIVALENT / N/A

After the per-metric table:
  OVERALL VERDICT vs CLEAN:  BEATS / MATCHES / BELOW / INSUFFICIENT DATA
  OVERALL VERDICT vs NOISY:  BEATS / MATCHES / BELOW / INSUFFICIENT DATA
  STRONGEST IMPROVEMENT: (one sentence)
  BIGGEST GAP: (one sentence)
  NOISE ROBUSTNESS NOTE: (one sentence — does the submitted model have any noise-awareness?)
  FAIRNESS NOTE: (one sentence on whether the comparison is apple-to-apple — same split? same eval?)

Use the data provided in submitted_comparison_data exactly. Do not invent numbers.
Equivalence threshold: |delta| < 0.5pp for BLEU-1, < 0.3pp for ROUGE.

Stay within EEG/NLP metric evaluation only.
""".strip()


def build_critic_prompt(submission: EEGModelSubmission) -> str:
    return f"""
[ROLE: critic]
You are a senior reviewer at NeurIPS / IEEE TNSRE.

You are reviewing a comparison between a submitted model ({submission.model_name}) and V9+QML.

Review format:
  [ISSUE-N] <short label>
  Problem: one sentence
  Fix: one sentence

Focus on:
  1. Eval protocol — is the submitted model using the same val split (TEST_SIZE=0.15 seed=42)?
     If unknown, flag it. A different split makes comparison invalid.
  2. Metric completeness — is BERTScore provided? Is FG BLEU provided?
     TF BLEU alone is insufficient for a full EEG decoding evaluation.
  3. Architecture comparison fairness — does the submitted model have more parameters?
     A larger model beating V9+QML is less impressive than a smaller one.
  4. Per-condition analysis — is SR performance better? SR is the hardest condition and
     V8/V9 both struggle there. Improvement on SR is the most meaningful delta.
  5. Statistical significance — at ZuCo scale (~2032 val samples), what delta is significant?
     Rule of thumb: < 0.5pp BLEU-1 is not meaningful.
  6. Noise robustness — V9+QML noisy (val loss 4.1729 vs clean 4.1733) showed that
     hardware-realistic noise barely degrades performance. Does the submitted model
     have any noise-awareness or is it purely classical?

End with:
  "Correctly identified contributions:" (bulleted)
  "Verdict: ACCEPT / CONDITIONAL ACCEPT / REVISE"
  "Confidence: X/10 — one sentence."

Stay within EEG decoding and model evaluation.
""".strip()


def build_synthesiser_prompt(submission: EEGModelSubmission) -> str:
    return f"""
[ROLE: qml_synthesiser]
You are a science communicator explaining research results to a fellow EEG researcher
who is not an expert in the specific V5→V8→V9→QML architecture chain.

You have a scientist's analysis, a head-to-head comparison, and a critic's review of
{submission.model_name} vs V9+QML.

Write 4 paragraphs, no headers, no bullets, max 400 words:
  Para 1: What the submitted model does and what it was trying to improve
  Para 2: How it compares to V9+QML clean AND noisy — where it wins, where it loses,
          what the clean vs noisy gap (0.0004 val loss, ~0pp BLEU) means for hardware deployment
  Para 3: What the critic's concerns mean practically — should a researcher trust this comparison?
  Para 4: What the submitting researcher should do next to strengthen their model or their evaluation

End with exactly one sentence: "The single most important next step for {submission.model_name} is: ___"

Stay strictly within EEG-to-text research on ZuCo.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPARISON PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def run_comparison_pipeline(
    submission: EEGModelSubmission,
    results_json_path: str = "nat_v9_qml_results.json",
    verbose: bool = True,
) -> dict:
    """
    Run the full 4-agent comparison pipeline for an external researcher's submission.

    Parameters
    ----------
    submission : EEGModelSubmission
        The researcher's model + metrics (fill in the dataclass fields)
    results_json_path : str
        Path to nat_v9_qml_results.json produced by the product notebook
    verbose : bool
        Print progress

    Returns
    -------
    dict with keys:
        scientist, comparator, critic, synthesiser  — agent outputs (str)
        comparison_data                             — full structured data used
        benchmark_records                           — per-call timing
        guardrail_audit                             — per-call rail status
        pipeline_summary                            — latency + pass-rate
    """

    # 1. Validate submission
    warnings = submission.validate()
    if warnings:
        print("⚠  Submission warnings:")
        for w in warnings:
            print(f"   - {w}")

    # 2. Load V9+QML baselines (clean + noisy)
    v9qml       = load_v9_qml_baseline(results_json_path)
    v9qml_noisy = load_v9_qml_noisy_baseline(results_json_path)

    # 3. Print quick summary table
    if verbose:
        submission.print_summary(v9qml)

    # 4. Build comparison data dict (what agents will read)
    comparison_data = submission.to_agent_stats_slice(v9qml)

    # 5. Load guardrails
    rails = _load_rails() if _NAG_AVAILABLE else None
    benchmark_records = []

    if verbose:
        print("=" * 64)
        print(f"  EEG Model Comparison Pipeline")
        print(f"  Submitted model : {submission.model_name}")
        print(f"  Reference       : V9+QML clean + noisy (V5/V8 also in context)")
        print(f"  Endpoint        : {NIM_BASE_URL if _NAG_AVAILABLE else 'N/A'}")
        print(f"  Rails           : {'NeMo Guardrails' if rails else 'Python-side'}")
        print("=" * 64)

    # ── [1/4] Scientist ───────────────────────────────────────────────────────
    if verbose: print("\n[1/4] Scientist agent...")
    sci_prompt = build_scientist_prompt(submission)
    sci_user   = f"""
SUBMITTED MODEL DATA:
{json.dumps(comparison_data['submitted_metrics'], indent=2)}

COMPARISON vs V9+QML:
{json.dumps(comparison_data['deltas_vs_v9qml'], indent=2)}

COMPARISON vs V8:
{json.dumps(comparison_data['deltas_vs_v8'], indent=2)}

V9+QML REFERENCE:
  CLEAN  BLEU-1={v9qml.get('tf_bleu1_pct')}%  ROUGE-1={v9qml.get('tf_rouge1_pct')}%
         BERTScore={v9qml.get('bertscore_f1')}%  FG={v9qml.get('fg_bleu1_pct')}%  TF/FG={v9qml.get('tf_fg_ratio')}×
         Per-condition: {json.dumps(v9qml.get('per_condition', {}))}
  NOISY  BLEU-1={v9qml_noisy.get('tf_bleu1_pct',"?")}%  ROUGE-1={v9qml_noisy.get('tf_rouge1_pct',"?")}%
         noise: DepolarizingChannel(p=0.01) + PhaseDamping(γ=0.02) + 16-pass MC inference
         val_loss=4.1729 (clean=4.1733)  hardware-deployable variant

Write your analysis of the submitted model.
""".strip()

    sci_out, sci_timing = await call_nim_guardrailed(
        sci_prompt, sci_user, "scientist",
        rails=rails, benchmark_record=benchmark_records
    )
    if verbose:
        print(f"  ✓  latency={sci_timing['total_ms']}ms  "
              f"guard={'✅' if sci_timing['guardrail_pass'] else '⛔'}")

    # ── [2/4] Comparator ─────────────────────────────────────────────────────
    if verbose: print("[2/4] Comparator agent...")
    cmp_prompt = build_comparator_prompt(submission)
    cmp_user   = f"""
submitted_comparison_data:
{json.dumps(comparison_data, indent=2)}

SCIENTIST ANALYSIS (context):
{sci_out[:800]}

Produce the head-to-head metric comparison table.
""".strip()

    cmp_out, cmp_timing = await call_nim_guardrailed(
        cmp_prompt, cmp_user, "comparator",
        rails=rails, benchmark_record=benchmark_records
    )
    if verbose:
        print(f"  ✓  latency={cmp_timing['total_ms']}ms  "
              f"guard={'✅' if cmp_timing['guardrail_pass'] else '⛔'}")

    # ── [3/4] Critic ──────────────────────────────────────────────────────────
    if verbose: print("[3/4] Critic agent...")
    crit_prompt = build_critic_prompt(submission)
    crit_user   = f"""
SCIENTIST ANALYSIS:
{sci_out[:600]}

COMPARATOR OUTPUT:
{cmp_out[:800]}

SUBMITTED METRICS:
{json.dumps(comparison_data['submitted_metrics'], indent=2)}

SUBMITTED MODEL INFO:
  name: {submission.model_name}
  val_split: {submission.val_split}
  n_val_samples: {submission.n_val_samples}
  notes: {submission.notes}

Review the submission and comparison.
""".strip()

    crit_out, crit_timing = await call_nim_guardrailed(
        crit_prompt, crit_user, "critic",
        rails=rails, benchmark_record=benchmark_records
    )
    if verbose:
        print(f"  ✓  latency={crit_timing['total_ms']}ms  "
              f"guard={'✅' if crit_timing['guardrail_pass'] else '⛔'}")

    # ── [4/4] Synthesiser ─────────────────────────────────────────────────────
    if verbose: print("[4/4] Synthesiser agent...")
    syn_prompt = build_synthesiser_prompt(submission)
    syn_user   = f"""
SCIENTIST: {sci_out[:600]}
COMPARATOR: {cmp_out[:600]}
CRITIC: {crit_out[:500]}

METRICS HEADLINE:
  {submission.model_name}: BLEU-1={submission.tf_bleu1_pct}%  ROUGE-1={submission.tf_rouge1_pct}%
  V9+QML clean:            BLEU-1={v9qml.get('tf_bleu1_pct')}%  ROUGE-1={v9qml.get('tf_rouge1_pct')}%
  V9+QML noisy:            BLEU-1={v9qml_noisy.get('tf_bleu1_pct',"?")}%  (hardware-sim; Δ vs clean ~0pp)
  Delta BLEU-1 vs clean: {comparison_data['deltas_vs_v9qml']['bleu1']:+}pp
  Delta ROUGE-1 vs clean: {comparison_data['deltas_vs_v9qml']['rouge1']:+}pp

Write your synthesis.
""".strip()

    syn_out, syn_timing = await call_nim_guardrailed(
        syn_prompt, syn_user, "qml_synthesiser",
        rails=rails, benchmark_record=benchmark_records
    )
    if verbose:
        print(f"  ✓  latency={syn_timing['total_ms']}ms  "
              f"guard={'✅' if syn_timing['guardrail_pass'] else '⛔'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_ms  = sum(r["total_ms"] for r in benchmark_records)
    pass_rate = 100 * sum(r["guardrail_pass"] for r in benchmark_records) / max(len(benchmark_records), 1)

    if verbose:
        print(f"\n── Comparison complete ─────────────────────────────────────")
        print(f"  Model compared : {submission.model_name}")
        print(f"  Total latency  : {total_ms:.0f}ms")
        print(f"  Pass rate      : {pass_rate:.0f}%")

    guardrail_audit = [
        {
            "agent":           r["agent"],
            "guardrail_pass":  r["guardrail_pass"],
            "guardrail_fired": r["guardrail_fired"],
            "total_ms":        r["total_ms"],
        }
        for r in benchmark_records
    ]

    return {
        "submitted_model":    submission.model_name,
        "scientist":          sci_out,
        "comparator":         cmp_out,
        "critic":             crit_out,
        "synthesiser":        syn_out,
        "comparison_data":    comparison_data,
        "benchmark_records":  benchmark_records,
        "guardrail_audit":    guardrail_audit,
        "pipeline_summary": {
            "total_pipeline_ms":       round(total_ms, 1),
            "guardrail_pass_rate_pct": round(pass_rate, 1),
            "endpoint":                NIM_BASE_URL if _NAG_AVAILABLE else "N/A",
            "model":                   NIM_MODEL if _NAG_AVAILABLE else "N/A",
            "rails_active":            rails is not None,
            "n_agents":                4,
        },
    }


def save_comparison_report(results: dict, output_path: Optional[str] = None) -> str:
    """Save comparison results to JSON. Returns the output path."""
    model_slug = results["submitted_model"].replace(" ", "_").lower()
    path = output_path or f"comparison_{model_slug}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Comparison report saved → {path}")
    return path
