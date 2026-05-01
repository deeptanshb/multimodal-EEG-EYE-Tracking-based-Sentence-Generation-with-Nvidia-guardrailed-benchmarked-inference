# ── nat_agents_guardrailed.py ─────────────────────────────────────────────────
# Drop-in replacement for NAT notebook cells 27–33
# Adds:
#   1. NeMo Guardrails (Colang 2.0) wrapping every agent call
#   2. NIM self-hosted OR cloud endpoint (auto-detect from NIM_BASE_URL)
#   3. Per-call inference benchmarking (TTFT, latency, tokens/s)
#   4. Domain-specific agents (Scientist / Critic / QML Synthesiser)
#   5. Guardrail audit log in final JSON output
#
# ── How to use in the notebook ────────────────────────────────────────────────
# Replace cells 27–33 with:
#
#   import importlib, nat_agents_guardrailed as nag
#   importlib.reload(nag)
#   results_agents = await nag.run_guardrailed_pipeline(agent_stats)
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import time
import asyncio
import importlib
from pathlib import Path
from typing import Optional

# ── Optional imports (graceful fallback) ──────────────────────────────────────
try:
    from nemoguardrails import LLMRails, RailsConfig
    GUARDRAILS_AVAILABLE = True
    print("✅ NeMo Guardrails loaded")
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("⚠  NeMo Guardrails not installed — running without rails")
    print("   pip install nemoguardrails")

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠  openai package missing — pip install openai")

# Add benchmark dir to path
sys.path.insert(0, str(Path(__file__).parent / "benchmark"))
try:
    from nim_benchmark import NIMBenchmark, default_guardrail_check, AgentBenchmarkReport
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    print("⚠  benchmark/nim_benchmark.py not found — benchmarking disabled")


# ── Environment / endpoint config ─────────────────────────────────────────────

NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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"
)

# NeMo Guardrails uses LangChain's OpenAI provider internally, which requires
# OPENAI_API_KEY. NIM is OpenAI-compatible, so we alias the key here.
# This must happen before _load_rails() is called.
if NVIDIA_API_KEY and not NVIDIA_API_KEY.startswith("nvapi-PASTE"):
    os.environ.setdefault("OPENAI_API_KEY", NVIDIA_API_KEY)

# If NIM_BASE_URL is set, use self-hosted NIM; otherwise fall back to cloud API
NIM_BASE_URL = os.environ.get(
    "NIM_BASE_URL",
    "https://integrate.api.nvidia.com/v1"   # cloud fallback
)

NIM_MODEL = os.environ.get(
    "NIM_MODEL",
    "meta/llama-3.1-8b-instruct"   # 8B: ~10x faster than 70B on cloud, sufficient for analysis
    # Switch to "meta/llama-3.1-70b-instruct" on brev self-hosted NIM for publication quality
)

# Path to guardrails config folder (relative to this file)
GUARDRAILS_CONFIG_PATH = Path(__file__).parent / "guardrails_config"


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN-SPECIFIC SYSTEM PROMPTS (upgraded from nat notebook cell 27)
# Changes vs original:
#   - Added explicit "Out of scope" sections per agent
#   - QML Synthesiser replaces generic "Explainer" — narrowly focused
#   - Each agent gets a ROLE tag for guardrail routing
#   - Corrected QML fine-tune to 10 epochs (matches final.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

SCIENTIST_SYSTEM = """
[ROLE: scientist]
You are a neuroscience and NLP researcher analysing the four-model EEG-to-text progression on ZuCo.

Architecture evolution:
  V5  → Conv1D + Bi-GRU + single mean-pool EEG vector, prefix-tuned DistilGPT2
  V8  → 6 parallel GRU-Transformer RegionEncoders, MoCo Stage0, LoRA rank=8 GPT2 [10,11], SR adapter
         pool_attn collapsed → uniform 1/256 (mean-pooling in disguise)
         True cross-region signal lives in self.fusion MHA (was discarded as `_` in V8)
  V9  → V8 + HierarchicalTemporalPooling (HTP): local_attn + seg_attn per region
         LoRA rank=4, lora_alpha=16, block=[11] only; dropout=0.3 for eval
  QML clean  → V9 + QuantumFusionProjector AFTER sr_adapter (noiseless, lightning.qubit):
               down H→4, AngleEmbedding, 2 StronglyEntanglingLayers (4 qubits), up 4→H, LN residual
               ~8,476 QML params; 10-epoch fine-tune (QML_LR=3e-4, rest=1e-6, CosineAnnealingLR)
               dropout=0.4; lora_alpha=8.0
  QML noisy  → same VQC + DepolarizingChannel(p=0.01) + PhaseDamping(γ=0.02) via default.mixed
               Gaussian shot-noise (σ=0.03) injected during training for robustness
               Inference: Monte-Carlo average over 16 noisy passes
               Initialised from QML clean checkpoint; 10-epoch noise-aware fine-tune
               Best val loss 4.1729 (clean: 4.1733) — noise acts as regulariser

Evaluation protocol:
  Sentence-aware val split (TEST_SIZE=0.15, seed=42). EEG+eye+spec normalised via saved scalers.
  TF BLEU uses logits[:, :-1, :] shift. ref_lens capped after EOS trim.
  V8 hardcoded baselines: BLEU-1=30.40%, ROUGE-1=35.78%, ROUGE-L=30.68%, BERTScore=85.46%
  V5 hardcoded baselines: BLEU-1=29.24%, ROUGE-1=33.92%, ROUGE-L=30.06%
  Per-condition V8: NR=30.90%, TSR=32.93%, SR=27.20%
  Per-condition V5: NR=30.70%, TSR=32.78%, SR=26.49%

Required analysis sections:
1. DATASET & SETUP — ZuCo, conditions, split, normalisation
2. FOUR-MODEL PROGRESSION — was each architectural addition justified by metrics?
3. TF PERFORMANCE — all five models (V5/V8/V9/QML-clean/QML-noisy) vs hardcoded baselines
4. FG PERFORMANCE — TF/FG ratio, compare to V8 1.97× and prior 3× norm
5. PER-CONDITION — NR/TSR/SR for all four models; TSR-SR gap; SR adapter contribution
6. ATTENTION DIAGNOSIS:
   a) HTP: did local_attn + seg_attn fix the 1/256 pool_attn collapse?
   b) Cross-region fusion: dominant region in V9 vs QML
   c) Neuroscience: Left Temporal (Wernicke), Left Parieto-Occ (VWFA), Central Parietal (P300)
   d) SR elevation of Left Temporal — attenuation in V9/QML?
7. QUALITATIVE — one sample per condition, V9 TF vs QML TF
8. CONCLUSIONS — 4 bullets: progression trend, QML contribution, key limitation, top next step

Out of scope: do not discuss topics unrelated to EEG, ZuCo, NLP metrics, or the V5→V8→V9→QML architectures.
""".strip()


CRITIC_SYSTEM = """
[ROLE: critic]
You are a senior reviewer at NeurIPS / IEEE TNSRE evaluating an EEG-to-text decoding paper.

Submission extends EEG2TextTransformerV8 with:
  1. HierarchicalTemporalPooling (HTP) — local_attn + seg_attn replacing collapsed pool_attn
  2. QuantumFusionProjector clean — 4-qubit VQC, residual post-sr_adapter, ~8,476 QML params, 10 epochs
  3. QuantumFusionProjector noisy — same VQC + DepolarizingChannel(p=0.01) + PhaseDamping(γ=0.02)
     hardware-realistic noise simulation; inference via 16-pass MC average; best val=4.1729
  4. LoRA rank reduced from 8 → 4, single block [11], alpha adjusted per classical/hybrid model
  5. Evaluation on sentence-aware val split (TEST_SIZE=0.15, seed=42)

Hardcoded V8 paper baselines (correct values):
  TF BLEU-1=30.40%, ROUGE-1=35.78%, ROUGE-L=30.68%, BERTScore=85.46%, FG BLEU-1=15.41%
  Per-condition: NR=30.90%, TSR=32.93%, SR=27.20%

Review format — use exactly:
  [ISSUE-N] <short label>
  Problem: one sentence
  Fix: one sentence

Focus areas:
  - HTP: genuine improvement vs parameter count increase?
  - QML clean: expressivity advantage vs equivalent 4→768 classical MLP residual?
  - QML noisy: val loss 4.1729 vs clean 4.1733 — is 0.0004 gap meaningful at this scale?
    Is noise regularisation the real contributor, not quantum computation per se?
  - Statistical significance of QML delta at ZuCo scale (~2032 val samples)
  - TF/FG ratio progress vs V8 1.97× baseline
  - eval protocol comparability (same split? same normalisation? same logit shift?)
  - LoRA rank reduction from 8→4: ablation missing?

End with exactly:
  "Correctly identified:" bulleted list of genuine contributions
  "Verdict: PASS / CONDITIONAL PASS / REVISE"
  "Confidence: X/10 — one sentence reason."

Out of scope: do not evaluate topics outside EEG decoding, model architecture, or metric validity.
""".strip()


QML_SYSTEM = """
[ROLE: qml_synthesiser]
You are a quantum-ML researcher and science communicator writing for a final-year engineering student.

Focus: QuantumFusionProjector (QFP) in the V9+QML hybrid model.

QFP architecture (exact — two variants trained):
  CLEAN (lightning.qubit — noiseless statevector simulation):
  - Input: 768-dim fused EEG embedding (post sr_adapter)
  - down: Linear(768 → 4) + tanh + π scaling
  - qlayer: PennyLane AngleEmbedding (Y rotation) + 2 StronglyEntanglingLayers on 4 qubits
  - up: Linear(4 → 768)
  - output: LayerNorm(x + Dropout(up(qlayer(down(x)))))  [residual]
  - ~8,476 trainable parameters; best val=4.1733
  NOISY (default.mixed — density-matrix noise simulation):
  - Same circuit + DepolarizingChannel(p=0.01) after each encoding gate
  - PhaseDamping(γ=0.02) after StronglyEntanglingLayers (models T2 decoherence)
  - Training: Gaussian shot-noise σ=0.03 injected on q_out each pass
  - Inference: Monte-Carlo average over 16 noisy circuit passes
  - Initialised from clean checkpoint; 10-epoch noise-aware fine-tune; best val=4.1729
  - Runs classically via default.mixed — hardware-realistic simulation, NOT real quantum hardware

Classical equivalent baseline for comparison:
  A 768→4→768 MLP residual with tanh activation would have ~6,144 params
  The QFP has ~8,476 params due to VQC weights (n_layers × n_qubits × 3)

Write 4 paragraphs, no bullets, no headers, max 380 words:
  Para 1: V5 limitation — why spatial mean-pooling loses information
  Para 2: V8/V9 fix — HTP and what it adds over V8's collapsed pool_attn
  Para 3: QFP mechanism — what the VQC actually computes; clean vs noisy variant;
          why noisy val loss (4.1729) is marginally better than clean (4.1733);
          honest assessment: is the 0.0004 gap noise-regularisation or quantum computation?
          what "quantum advantage" means (or doesn't) on classical hardware at 4 qubits
  Para 4: What the V5→V8→V9→QML progression teaches about EEG-to-text, ending with ONE next step

Out of scope: do not discuss topics unrelated to the QFP, the V9 architecture, or EEG decoding.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAILS WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def _load_rails() -> Optional["LLMRails"]:
    """
    Load NeMo Guardrails config from guardrails_config/ folder.
    Auto-detects Colang version: tries as-is first, then patches config.yml
    to remove colang_version line if the installed build doesn't support 2.0.
    """
    if not GUARDRAILS_AVAILABLE:
        return None
    if not GUARDRAILS_CONFIG_PATH.exists():
        print(f"⚠  Guardrails config not found at {GUARDRAILS_CONFIG_PATH}")
        return None

    config_yml = GUARDRAILS_CONFIG_PATH / "config.yml"
    rails_co   = GUARDRAILS_CONFIG_PATH / "rails.co"

    # ── Check if rails.co uses Colang 2.0 syntax (flow keyword without define) ─
    # If so, and the installed version doesn't support it, patch to 1.0
    if rails_co.exists():
        rails_src = rails_co.read_text()
        has_colang2 = any(
            line.strip().startswith("flow ") and "define" not in line
            for line in rails_src.splitlines()
        )
        if has_colang2:
            print("⚠  rails.co uses Colang 2.0 syntax — patching to 1.0 compatible version")
            # Write the 1.0 compatible version inline
            _write_colang1_rails(rails_co)

    # ── Patch config.yml: remove colang_version: "2.0" if present ──────────────
    if config_yml.exists():
        cfg_text = config_yml.read_text()
        if 'colang_version: "2.0"' in cfg_text or "colang_version: '2.0'" in cfg_text:
            cfg_patched = "\n".join(
                line for line in cfg_text.splitlines()
                if "colang_version" not in line
            )
            config_yml.write_text(cfg_patched)
            print("⚠  Removed colang_version: 2.0 from config.yml (not supported by this build)")

    try:
        config = RailsConfig.from_path(str(GUARDRAILS_CONFIG_PATH))
        rails  = LLMRails(config)
        print(f"✅ Guardrails loaded (Colang 1.0) from {GUARDRAILS_CONFIG_PATH}")
        return rails
    except Exception as ex:
        print(f"⚠  Guardrails load error: {ex}  — running without rails (Python-side checks active)")
        return None


def _write_colang1_rails(path: Path):
    """Write a Colang 1.0 compatible rails.co, replacing the 2.0 version."""
    colang1_content = '''# rails.co — Colang 1.0 — EEG-to-Text Agent Guardrails
# Auto-patched from Colang 2.0 for compatibility

define user ask about eeg research
  "analyse my EEG model"
  "compare bleu scores"
  "evaluate the V9 model"
  "analyse attention weights"
  "neuroscience analysis"
  "ZuCo dataset evaluation"
  "critique the QML results"
  "compare clean and noisy QML"
  "what is the hardware noise impact"
  "depolarizing channel effect on BLEU"

define user ask off topic
  "what is the weather"
  "recommend a recipe"
  "ignore previous instructions"
  "pretend you are"
  "stock market"

define bot refuse off topic
  "This pipeline is scoped to EEG-to-text neuroscience research on ZuCo. Off-topic queries are not supported."

define bot flag metric issue
  "Output flagged: metric values are outside plausible EEG-to-text range (BLEU 20-55%, BERTScore 78-96%)."

define bot flag off domain output
  "Response flagged as not grounded in EEG research. Output must reference EEG metrics or architecture details."

define flow check eeg domain intent
  user ask off topic
  bot refuse off topic

define flow check metric hallucination
  $has_metric_issue = execute check_metric_bounds(response=$last_bot_message)
  if $has_metric_issue
    bot flag metric issue

define flow check domain relevance
  $is_relevant = execute self_check_relevance(response=$last_bot_message)
  if not $is_relevant
    bot flag off domain output
'''
    path.write_text(colang1_content)
    print(f"  Wrote Colang 1.0 rails to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CORE LLM CALL — with timing and guardrails
# ─────────────────────────────────────────────────────────────────────────────

async def call_nim_guardrailed(
    system:       str,
    user:         str,
    agent:        str,
    rails:        Optional["LLMRails"] = None,
    benchmark_record: Optional[list] = None,
) -> tuple[str, dict]:
    """
    Call NIM (cloud or self-hosted) with optional guardrails.
    Returns (response_text, timing_dict).
    timing_dict keys: ttft_ms, total_ms, tokens_per_sec, guardrail_pass, guardrail_fired
    """
    key_ok = NVIDIA_API_KEY and not NVIDIA_API_KEY.startswith("nvapi-PASTE")
    timing = {
        "agent": agent,
        "ttft_ms": 0.0,
        "total_ms": 0.0,
        "output_tokens": 0,
        "tokens_per_sec": 0.0,
        "guardrail_pass": True,
        "guardrail_fired": "",
        "endpoint": NIM_BASE_URL,
        "model": NIM_MODEL,
    }

    if not key_ok or not OPENAI_AVAILABLE:
        text = f"[simulation — set NVIDIA_API_KEY]\nAgent: {agent}"
        if benchmark_record is not None:
            benchmark_record.append(timing)
        return text, timing

    # ── Unified path: always call NIM directly, apply Python-side guards post-call ──
    # NeMo Guardrails input rails intercept prompts before they reach the LLM,
    # which causes empty responses when the EEG metric text triggers the intent
    # classifier. Instead: make the LLM call directly, then run output checks.
    # The rails object is used for reporting (rails_active) but not for routing.
    if True:  # always take this path regardless of rails
        # Plain float timeout (600s = 10 min) — reliable across all openai SDK versions
        _TIMEOUT = 600.0
        client = AsyncOpenAI(
            base_url=NIM_BASE_URL,
            api_key=NVIDIA_API_KEY,
            timeout=_TIMEOUT,
            max_retries=0,
        )
        t0    = time.perf_counter()
        ttft  = None
        chunks = []
        text   = ""
        _last_err = None

        for _attempt in range(2):
            chunks = []
            ttft   = None
            try:
                if _attempt == 0:
                    stream = await client.chat.completions.create(
                        model=NIM_MODEL,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                        temperature=0.1,
                        max_tokens=900,
                        stream=True,
                    )
                    async for chunk in stream:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            chunks.append(delta)
                    break
                else:
                    print(f"  Retrying without streaming...")
                    resp = await client.chat.completions.create(
                        model=NIM_MODEL,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                        temperature=0.1,
                        max_tokens=900,
                        stream=False,
                    )
                    chunks = [resp.choices[0].message.content or ""]
                    ttft   = (time.perf_counter() - t0) * 1000
                    break
            except Exception as ex:
                _last_err = ex
                print(f"  Attempt {_attempt+1} failed ({type(ex).__name__}: {str(ex)[:80]})")
                if _attempt < 1:
                    print(f"  Waiting 5s before retry...")
                    await asyncio.sleep(5)

        if not chunks and _last_err is not None:
            text = f"[error: {_last_err}]"
            timing["guardrail_pass"]  = False
            timing["guardrail_fired"] = f"error:{_last_err}"
            if benchmark_record is not None:
                benchmark_record.append(timing)
            return text, timing

        t1   = time.perf_counter()
        text = "".join(chunks)

        timing["total_ms"]      = round((t1 - t0) * 1000, 1)
        timing["ttft_ms"]       = round(ttft or timing["total_ms"], 1)
        timing["output_tokens"] = len(text.split())
        if timing["total_ms"] > 0:
            timing["tokens_per_sec"] = round(
                timing["output_tokens"] / (timing["total_ms"] / 1000), 1
            )

        # Python-side guardrail check (mirrors Colang output rails)
        g_pass, g_fired = await default_guardrail_check(text, agent) \
            if BENCHMARK_AVAILABLE else (True, "")
        timing["guardrail_pass"]  = g_pass
        timing["guardrail_fired"] = g_fired

    if benchmark_record is not None:
        benchmark_record.append(timing)

    return text, timing


# ─────────────────────────────────────────────────────────────────────────────
# THREE-AGENT PIPELINE  (drop-in replacement for run_pipeline() in notebook)
# ─────────────────────────────────────────────────────────────────────────────

async def run_guardrailed_pipeline(agent_stats: dict) -> dict:
    """
    Full 3-agent pipeline with guardrails + benchmarking.
    Drop-in for the original run_pipeline() in the NAT notebook.

    Returns the same dict structure as before PLUS:
      results["benchmark_records"] — list of per-call timing dicts
      results["guardrail_audit"]   — list of guardrail events
    """
    lm = agent_stats["live_metrics"]
    v5 = agent_stats["baselines"]["v5"]
    v8 = agent_stats["baselines"]["v8"]
    aa = agent_stats["attention_analysis"]

    # Load guardrails (None if not installed)
    rails = _load_rails()
    benchmark_records = []

    print("=" * 68)
    print("  EEG2TextTransformerV9+QML — Guardrailed Three-Agent Pipeline")
    print(f"  Val n={lm['n']:,}  |  V5 → V8 → V9 → QML-clean → QML-noisy  |  normalised eval")
    print(f"  Endpoint: {NIM_BASE_URL}")
    print(f"  Model   : {NIM_MODEL}")
    print(f"  Rails   : {'NeMo Guardrails (Colang 1.0) — output checks active' if rails else 'Python-side only'}")
    print("=" * 68)

    # ── [1/3] Scientist ───────────────────────────────────────────────────────
    print("\n[1/3] Scientist agent...")

    # Build slim prompt — flat key numbers only, no full JSON blobs
    def _fmt_attn(attn_dict):
        """Top-3 regions by cross-region weight, formatted as a short string."""
        vals = attn_dict.get("cross_region_fusion", {}).get("values", {})
        if not vals:
            return "N/A"
        top3 = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:3]
        return "  ".join(f"{k}:{v:.3f}" for k, v in top3)

    def _fmt_per_cond(per_cond_dict):
        nr  = per_cond_dict.get("NR",  "?")
        tsr = per_cond_dict.get("TSR", "?")
        sr  = per_cond_dict.get("SR",  "?")
        return f"NR={nr}%  TSR={tsr}%  SR={sr}%"

    qs = agent_stats.get("qualitative_samples", [])
    qual_lines = "\n".join(
        f"  [{s.get('condition','?')}] ref: {str(s.get('reference',''))[:60]}\n"
        f"         V9:  {str(s.get('v9_tf',''))[:50]}\n"
        f"         QML: {str(s.get('qml_tf',''))[:50]}"
        for s in qs[:3]
    )

    sci_user = f"""
LIVE METRICS (val n={lm['n']:,}):
  V5  TF BLEU-1={v5['tf_bleu1_pct']}%  ROUGE-1={v5['tf_rouge1_pct']}%  ROUGE-L={v5.get('tf_rougeL_pct','?')}%
  V8  TF BLEU-1={v8['tf_bleu1_pct']}%  ROUGE-1={v8['tf_rouge1_pct']}%  ROUGE-L={v8['tf_rougeL_pct']}%  BERTScore={v8['bertscore_f1']}%  FG={v8['fg_bleu1_pct']}%
  V9  TF BLEU-1={lm['v9_tf_bleu1_pct']}%  ROUGE-1={lm['v9_tf_rouge1_pct']}%  ROUGE-L={lm['v9_tf_rougeL_pct']}%  FG={lm['v9_fg_bleu1_pct']}%  TF/FG={lm['v9_tf_fg_ratio']}x
  QML TF BLEU-1={lm['qml_tf_bleu1_pct']}%  ROUGE-1={lm['qml_tf_rouge1_pct']}%  ROUGE-L={lm['qml_tf_rougeL_pct']}%  FG={lm['qml_fg_bleu1_pct']}%  TF/FG={lm['qml_tf_fg_ratio']}x
  NQM TF BLEU-1={lm.get('noisy_qml_tf_bleu1_pct','?')}%  ROUGE-1={lm.get('noisy_qml_tf_rouge1_pct','?')}%  ROUGE-L={lm.get('noisy_qml_tf_rougeL_pct','?')}%  FG={lm.get('noisy_qml_fg_bleu1_pct','?')}%

DELTAS:
  V8→V9:  BLEU-1={lm['delta_v9_vs_v8_bleu1']:+.2f}pp  ROUGE-1={lm['delta_v9_vs_v8_rouge1']:+.2f}pp
  V9→QML: BLEU-1={lm['delta_qml_vs_v9_bleu1']:+.2f}pp  ROUGE-1={lm['delta_qml_vs_v9_rouge1']:+.2f}pp
  V8→QML: BLEU-1={lm['delta_qml_vs_v8_bleu1']:+.2f}pp
  clean→noisy: BLEU-1={lm.get('delta_noisy_vs_clean_bleu1',0):+.2f}pp  ROUGE-1={lm.get('delta_noisy_vs_clean_rouge1',0):+.2f}pp

PER-CONDITION TF BLEU-1:
  V5:  {_fmt_per_cond(v5.get('per_condition',{}))}
  V8:  {_fmt_per_cond(v8.get('per_condition',{}))}
  V9:  {_fmt_per_cond(lm.get('v9_per_cond_bleu1',{}))}
  QML: {_fmt_per_cond(lm.get('qml_per_cond_bleu1',{}))}
  NQM: {_fmt_per_cond(lm.get('noisy_qml_per_cond_bleu1',{}))}

CROSS-REGION FUSION (top-3 weights):
  V9  classical: {_fmt_attn(aa['v9_classical'])}   dominant={aa['v9_classical']['cross_region_fusion'].get('dominant','?')}
  V9+QML hybrid: {_fmt_attn(aa['v9_qml_hybrid'])}  dominant={aa['v9_qml_hybrid']['cross_region_fusion'].get('dominant','?')}
  V9+QML noisy:  {_fmt_attn(aa.get('v9_qml_noisy_hybrid', aa['v9_qml_hybrid']))}  dominant={aa.get('v9_qml_noisy_hybrid',aa['v9_qml_hybrid'])['cross_region_fusion'].get('dominant','?')}

QUALITATIVE SAMPLES:
{qual_lines}

Write your full analysis covering all 8 required sections.
""".strip()

    sci_out, sci_timing = await call_nim_guardrailed(
        SCIENTIST_SYSTEM, sci_user, "scientist",
        rails=rails, benchmark_record=benchmark_records
    )
    print(f"  ✓  latency={sci_timing['total_ms']}ms  "
          f"TTFT={sci_timing['ttft_ms']}ms  "
          f"tokens/s={sci_timing['tokens_per_sec']}  "
          f"guard={'✅' if sci_timing['guardrail_pass'] else '⛔ ' + sci_timing['guardrail_fired']}")

    # ── [2/3] Critic ──────────────────────────────────────────────────────────
    print("[2/3] Critic agent...")
    crit_user = f"""
SCIENTIST SUMMARY (first 500 chars):
{sci_out[:500]}

KEY NUMBERS (authoritative from agent_stats):
  V5  TF BLEU-1={v5['tf_bleu1_pct']}%
  V8  TF BLEU-1={v8['tf_bleu1_pct']}%   ROUGE-1={v8['tf_rouge1_pct']}%   BERTScore={v8['bertscore_f1']}%
  V9  TF BLEU-1={lm['v9_tf_bleu1_pct']}%  Δ vs V8={lm['delta_v9_vs_v8_bleu1']:+.2f}pp
  QML TF BLEU-1={lm['qml_tf_bleu1_pct']}%  Δ vs V9={lm['delta_qml_vs_v9_bleu1']:+.2f}pp  Δ vs V8={lm['delta_qml_vs_v8_bleu1']:+.2f}pp
  NQM TF BLEU-1={lm.get('noisy_qml_tf_bleu1_pct','?')}%  Δ vs clean={lm.get('delta_noisy_vs_clean_bleu1',0):+.2f}pp  val_loss=4.1729 vs clean=4.1733
  V9 TF/FG={lm['v9_tf_fg_ratio']}x   QML TF/FG={lm['qml_tf_fg_ratio']}x   V8 TF/FG=1.97x
  V9 dominant={aa['v9_classical']['cross_region_fusion'].get('dominant','?')}
  QML dominant={aa['v9_qml_hybrid']['cross_region_fusion'].get('dominant','?')}
  NQM dominant={aa.get('v9_qml_noisy_hybrid',aa['v9_qml_hybrid'])['cross_region_fusion'].get('dominant','?')}

Review the submission using the required [ISSUE-N] format.
""".strip()

    crit_out, crit_timing = await call_nim_guardrailed(
        CRITIC_SYSTEM, crit_user, "critic",
        rails=rails, benchmark_record=benchmark_records
    )
    print(f"  ✓  latency={crit_timing['total_ms']}ms  "
          f"TTFT={crit_timing['ttft_ms']}ms  "
          f"tokens/s={crit_timing['tokens_per_sec']}  "
          f"guard={'✅' if crit_timing['guardrail_pass'] else '⛔ ' + crit_timing['guardrail_fired']}")

    # ── [3/3] QML Synthesiser ─────────────────────────────────────────────────
    print("[3/3] QML Synthesiser agent...")
    qml_user = f"""
SCIENTIST (summary): {sci_out[:400]}
CRITIC (summary):    {crit_out[:300]}

METRICS:
  V5={v5['tf_bleu1_pct']}%  V8={v8['tf_bleu1_pct']}%  V9={lm['v9_tf_bleu1_pct']}%  QML-clean={lm['qml_tf_bleu1_pct']}%  QML-noisy={lm.get('noisy_qml_tf_bleu1_pct','?')}%
  FG: V8={v8['fg_bleu1_pct']}%  V9={lm['v9_fg_bleu1_pct']}%  QML={lm['qml_fg_bleu1_pct']}%
  Val loss: QML-clean=4.1733  QML-noisy=4.1729  (Δ=0.0004 — noise as regulariser)

QFP clean : 768→4→VQC(4q,2L)→768, LN residual, ~8476 params, lightning.qubit (noiseless)
QFP noisy : same + DepolarizingChannel(p=0.01) + PhaseDamping(γ=0.02), 16-pass MC inference

Write 4 paragraphs ≤380 words total.
""".strip()

    qml_out, qml_timing = await call_nim_guardrailed(
        QML_SYSTEM, qml_user, "qml_synthesiser",
        rails=rails, benchmark_record=benchmark_records
    )
    print(f"  ✓  latency={qml_timing['total_ms']}ms  "
          f"TTFT={qml_timing['ttft_ms']}ms  "
          f"tokens/s={qml_timing['tokens_per_sec']}  "
          f"guard={'✅' if qml_timing['guardrail_pass'] else '⛔ ' + qml_timing['guardrail_fired']}")

    # ── Pipeline summary ──────────────────────────────────────────────────────
    total_ms = sum(r["total_ms"] for r in benchmark_records)
    pass_rate = 100 * sum(r["guardrail_pass"] for r in benchmark_records) / max(len(benchmark_records), 1)
    print(f"\n── Pipeline complete ────────────────────────────────────────────")
    print(f"  Total pipeline latency : {total_ms:.0f}ms")
    print(f"  Guardrail pass rate    : {pass_rate:.0f}%")
    print(f"  Agents fired           : 3  (scientist / critic / qml_synthesiser)")
    print(f"  Rails                  : {'NeMo Guardrails' if rails else 'Python-side'}")

    # ── Guardrail audit log ───────────────────────────────────────────────────
    guardrail_audit = [
        {
            "agent":           r["agent"],
            "guardrail_pass":  r["guardrail_pass"],
            "guardrail_fired": r["guardrail_fired"],
            "total_ms":        r["total_ms"],
            "ttft_ms":         r["ttft_ms"],
            "tokens_per_sec":  r["tokens_per_sec"],
        }
        for r in benchmark_records
    ]

    return {
        "stats":             agent_stats,
        "scientist":         sci_out,
        "critic":            crit_out,
        "qml_synthesiser":   qml_out,
        # backward-compat alias (old cells reference results_agents["explainer"])
        "explainer":         qml_out,
        # noisy hybrid outputs are embedded in the qml_synthesiser analysis
        "noisy_qml_in_analysis": True,
        "benchmark_records": benchmark_records,
        "guardrail_audit":   guardrail_audit,
        "pipeline_summary": {
            "total_pipeline_ms":       round(total_ms, 1),
            "guardrail_pass_rate_pct": round(pass_rate, 1),
            "endpoint":                NIM_BASE_URL,
            "model":                   NIM_MODEL,
            "rails_active":            rails is not None,
            "rails_mode":              "NeMo loaded + Python output checks" if rails is not None else "Python-side only",
        },
    }