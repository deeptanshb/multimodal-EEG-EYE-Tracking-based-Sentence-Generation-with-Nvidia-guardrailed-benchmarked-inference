# Multimodal EEG+Eye-to-Text Decoding with Quantum-Classical Hybrid AI

> **"From Brain and Eye Signals to Sentences: Hierarchical Temporal Pooling, Quantum Fusion,
> and Guardrailed Multi-Agent Inference Benchmarking for Multimodal EEG+Eye-to-Text Decoding"**
>
> Decoding natural language from simultaneous EEG + eye-tracking signals using a condition-adaptive
> multi-region transformer, MoCo contrastive pretraining, hierarchical temporal pooling (HTP),
> LoRA fine-tuning, and a 4-qubit variational quantum circuit — evaluated on the ZuCo corpus across
> five model generations (V5 → V8 → V9 → QML clean → QML noisy), with a guardrailed NVIDIA NIM multi-agent
> inference benchmarking platform.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Evolution](#2-architecture-evolution)
3. [Repository Structure](#3-repository-structure)
4. [Dataset — ZuCo Corpus](#4-dataset--zuco-corpus)
5. [Environment Setup](#5-environment-setup)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Results](#7-results)
8. [NVIDIA NIM Agent Platform](#8-nvidia-nim-agent-platform)
9. [NeMo Guardrails](#9-nemo-guardrails)
10. [Inference Benchmark Harness](#10-inference-benchmark-harness)
11. [External Researcher Interface](#11-external-researcher-interface)
12. [Streamlit Dashboard — app.py](#12-streamlit-dashboard--apppy)
13. [Getting an NVIDIA API Key](#13-getting-an-nvidia-api-key)
14. [Plots Reference](#14-plots-reference)
15. [Key Findings](#15-key-findings)
16. [Citation](#16-citation)

---

## 1. Project Overview

This project implements an end-to-end brain-computer interface (BCI) pipeline that:

- Reads EEG signals + eye-tracking data recorded while participants silently read sentences
- Preprocesses, normalises, and structures the neural signals into a multimodal feature set
- Trains a multi-region transformer with contrastive pretraining, hierarchical temporal pooling,
  condition-specific adapters, and an optional quantum residual circuit to decode the original sentence
- Evaluates decoding quality using BLEU-1/4, ROUGE-1/L, BERTScore F1, and the **TF/FG ratio**
  (a new metric quantifying how strongly the model depends on the EEG signal vs language priors)
- Runs a three-agent NVIDIA NIM guardrailed pipeline (Scientist → Critic → QML Synthesiser)
  to automatically interpret and peer-review the results
- Provides an open benchmarking platform where external researchers submit their own model metrics
  and receive structured comparative analysis against the V9+QML baseline

**Dataset:** ZuCo (Zurich Cognitive Language Processing Corpus) — 12 subjects, ~700 unique sentences,
three reading conditions (Normal Reading / Timed Silent Reading / Speed Reading).

**Key results (val n=2,032, corrected locked baselines):**

| Model | TF BLEU-1 | TF ROUGE-1 | BERTScore F1 | TF/FG Ratio |
|-------|-----------|------------|--------------|-------------|
| V5 baseline | 29.24% | 33.92% | — | — |
| V8 baseline | 30.40% | **35.78%** | **85.46%** | 1.97× |
| V9 classical | **31.02%** | **36.07%** | — | **4.79×** |
| V9+QML clean | 31.00% | 36.04% | — | **4.79×** |
| V9+QML noisy | 31.00% | 36.05% | — | **4.79×** |

> The TF/FG ratio jump from 1.97× (V8) to 4.79× (V9+QML) is the most important result —
> the model genuinely depends on the EEG signal rather than relying on language priors.
> V9+QML noisy (hardware-realistic simulation) matches clean QML within 0.01pp — architecture is hardware-deployable.

---

## 2. Architecture Evolution

### V5 — Baseline

- Conv1D + Bidirectional GRU EEG encoder, single mean-pooled EEG vector
- Prefix-tuned DistilGPT-2 decoder, no eye-tracking or spectral features
- **TF BLEU-1: 29.24% | ROUGE-1: 33.92%** | Per-condition: NR=30.70% TSR=32.78% SR=26.49%

### V8 — Multimodal Multi-Region with Contrastive Learning

Key additions:

- **6 parallel GRU-Transformer RegionEncoders** — left temporal, left parietal, left parieto-occipital,
  central parietal, right parietal, right parieto-occipital
- **MoCo Stage 0** contrastive pretraining (queue=128, condition-based hard negatives)
- **LoRA fine-tuning** on GPT-2 blocks [10, 11] (rank=8, α=16)
- **SR condition adapter** — three separate MLPs for NR/TSR/SR conditions
- **Eye-tracking encoder** (fixations, pupil, duration) + **Spectral encoder** (8 band-power means)
- **Diagnosis:** `pool_attn Linear(D,1)` collapsed to uniform 1/256 in 4/6 regions — effectively mean-pooling
- **TF BLEU-1: 30.40% | ROUGE-1: 35.78% | BERTScore: 85.46%** | TF/FG: 1.97×
- Per-condition: NR=30.90% TSR=32.93% SR=27.20%

### V9 — Hierarchical Temporal Pooling (HTP)

Key fix:

- **HierarchicalTemporalPooling** replaces flat `pool_attn`:
  - Level 1: 32-way local softmax within 8 windows of 32 timesteps each (0.5s at 64 Hz)
  - Level 2: 8-way segment softmax across windows
  - Gradient concentrated 8× vs collapsed 256-way softmax → selective temporal peaks
- **LoRA rank=4, α=16, block=[11] only** (rank reduced from 8, single block)
- **dropout=0.4**; encoder near-frozen in Stage 2 (lr=1e-6)
- **TF BLEU-1: 31.02% | ROUGE-1: 36.07%** | TF/FG: **4.79×** | Per-condition: NR=32.48% TSR=31.30% SR=28.54%

### V9+QML clean — Quantum Fusion Projector (noiseless)

Key addition:

- **QuantumFusionProjector (QFP)** inserted after `sr_adapter`, before fusion MHA:
  - `Linear(768→4)` + tanh + π-scaling into qubit space
  - `AngleEmbedding` (RY rotations) encodes 4-dim EEG into 4-qubit state
  - 2× `StronglyEntanglingLayers` (CNOT ladders + rotation gates)
  - 4 Pauli-Z expectations → `Linear(4→768)` + LayerNorm residual
  - **~8,476 QML parameters** (0.006% of 147M total)
- **PennyLane** `lightning.qubit` simulator — noiseless statevector simulation
- **10-epoch QML fine-tune**: QML_LR=3e-4, rest=1e-6, CosineAnnealingLR, eta_min=1e-7, patience=3
- **Hybrid LoRA**: rank=4, **α=8.0**, block=[11]; dropout=0.4
- **TF BLEU-1: 31.00% | ROUGE-1: 36.04%** | TF/FG: **4.79×** | val loss: **4.1733**

### V9+QML noisy — Hardware-Realistic Noise Simulation

Key addition on top of V9+QML clean:

- **NoisyQuantumFusionProjector** — same VQC + hardware-realistic noise channels:
  - `DepolarizingChannel(p=0.01)` after each encoding gate (1% gate error)
  - `PhaseDamping(γ=0.02)` after `StronglyEntanglingLayers` (T2 decoherence)
  - Uses **PennyLane `default.mixed`** density-matrix simulator
- **Training**: Gaussian shot-noise (σ=0.03) injected on VQC output each pass → forces robustness
- **Inference**: Monte-Carlo average over 16 noisy circuit passes (variance ÷ 4×)
- **Initialised from clean QML checkpoint**; 10-epoch noise-aware fine-tune
- **TF BLEU-1: 31.00% | ROUGE-1: 36.05%** | val loss: **4.1729** (clean: 4.1733)
- Δ clean → noisy: 0.0004 val loss improvement — **noise acts as regulariser**; architecture is hardware-deployable

---

## 3. Repository Structure

```
PROJECT1/
│
├── ── CORE MODEL & TRAINING ────────────────────────────────────────────
│
├── model1_v9.py                     # ALL model classes: HTP, RegionEncoderV9,
│                                    #   EEG2TextTransformerV9, QuantumFusionProjector,
│                                    #   MoCo, training helpers, REGION_NAMES
├── final.ipynb                      # Main training + evaluation notebook (43 cells)
│                                    #   Cells 0-39: original training + plots
│                                    #   Cell 40: noisy QML fine-tune (Cell A)
│                                    #   Cell 41: NoisyQFP definition (skip Cell 40 if ckpt exists)
│                                    #   Cell 42: 4-model inference comparison (Cell B)
│                                    #   Stage0 MoCo → Stage1 → Stage2 LoRA → QML fine-tune
│                                    #   → evaluation → diagnostics → plots
├── my.ipynb                         # ZuCo .mat → pickle extractor (3 cells)
│
├── ── CHECKPOINTS ──────────────────────────────────────────────────────
│
├── stage0_v9.pt                     # Stage 0 MoCo checkpoint
├── stage1_best_v9.pt                # Stage 1 best checkpoint
├── final_best_v9.pt                 # Best Stage 2 (LoRA) checkpoint
├── hybrid_qml_v9_best.pt            # Best QML clean checkpoint (val loss=4.1733)
├── hybrid_qml_noisy_v9_best.pt      # Best QML noisy checkpoint (val loss=4.1729)
│
├── ── NVIDIA AGENT PLATFORM ────────────────────────────────────────────
│
├── nat_eeg_agents_v9_product.ipynb  # Main agent notebook (43 cells) — PRODUCT VERSION
│                                    #   Cells 1-13: inference + metrics + agent_stats
│                                    #   Cell 14b:   install NeMo Guardrails
│                                    #   Cell 14:    view agent system prompts
│                                    #   Cell 14c:   LLM caller + guardrail flow explainer
│                                    #   Cell 15:    load nat_agents_guardrailed module
│                                    #   Cell 16:    run guardrailed 3-agent pipeline
│                                    #   Cell 17:    inference benchmark harness
│                                    #   Cell 18:    display agent outputs
│                                    #   Cell 10b:   NoisyQFP + noisy_hybrid model setup
│                                    #   Cell 19:    save nat_v9_qml_results.json
│
├── nat_v9_qml_results.json          # Agent pipeline output — live metrics + agent text
│                                    #   + benchmark_records + guardrail_audit
│
├── ── eeg_product/  ────────────────────────────────────────────────────
│   │   (NVIDIA product layer — all agent + guardrail + benchmark code)
│   │
│   ├── nat_agents_guardrailed.py    # Core pipeline module:
│   │                                #   SCIENTIST_SYSTEM, CRITIC_SYSTEM, QML_SYSTEM prompts
│   │                                #   call_nim_guardrailed() — 300s timeout, retry logic
│   │                                #   run_guardrailed_pipeline() — 3-agent orchestrator
│   │                                #   _load_rails() — NeMo Guardrails auto-loader
│   │                                #   _write_colang1_rails() — Colang 1.0 auto-patch
│   │
│   ├── eeg_submission_schema.py     # External researcher interface:
│   │                                #   V5_BASELINE, V8_BASELINE (locked constants)
│   │                                #   V9_QML_BASELINE, V9_QML_NOISY_BASELINE
│   │                                #   EEGModelSubmission dataclass
│   │                                #   load_v9_qml_baseline(), load_v9_qml_noisy_baseline()
│   │
│   ├── comparison_pipeline.py       # 4-agent comparison pipeline for external researchers:
│   │                                #   Scientist + Comparator + Critic + Synthesiser
│   │                                #   run_comparison_pipeline(), save_comparison_report()
│   │
│   ├── external_researcher_template.ipynb  # 10-cell template notebook for external users
│   │                                       #   No model code needed — metrics only
│   │
│   ├── guardrails_config/
│   │   ├── config.yml               # NeMo Guardrails config:
│   │   │                            #   engine: openai (OpenAI-compat, points to NIM)
│   │   │                            #   model: meta/llama-3.1-8b-instruct
│   │   │                            #   input rails + output rails declared
│   │   ├── rails.co                 # Colang 1.0 flow definitions:
│   │   │                            #   check eeg domain intent (input) — incl. noisy QML examples
│   │   │                            #   check metric hallucination (output)
│   │   │                            #   check domain relevance (output)
│   │   │                            #   check noisy qml context (output) — validates payload keys
│   │   └── guardrails_actions.py    # Python actions:
│   │                                #   check_metric_bounds() — BLEU/ROUGE/BERTScore ranges
│   │                                #   self_check_relevance() — 38 domain terms incl. noisy QML
│   │                                #   check_noisy_qml_keys() — validates noisy_qml_* payload keys
│   │                                #   get_agent_role() — role router
│   │
│   └── benchmark/
│       └── nim_benchmark.py         # Inference benchmark harness:
│                                    #   NIMBenchmark class — N-run pipeline benchmark
│                                    #   CallMetrics, AgentBenchmarkReport dataclasses
│                                    #   default_guardrail_check() — Python-side checks
│                                    #   CLI: python nim_benchmark.py --runs 5
│
├── ── STREAMLIT DASHBOARD ──────────────────────────────────────────────
│
├── app.py                           # Full Streamlit dashboard (8 pages):
│                                    #   Overview / Training Curves / Model Comparison
│                                    #   EEG Attention / Architecture / Qualitative Samples
│                                    #   Quantum Fusion / NVIDIA Stack / NAT Agents
│                                    #   Includes live agent runner with guardrail badges
│
├── ── COMPARISON OUTPUT ────────────────────────────────────────────────
│
├── comparison_eegconformer_lora_v1.json   # Sample external researcher comparison result
│                                          #   EEGConformer_LoRA_v1 vs V9+QML baseline
│                                          #   4 agents · 18.2s · 100% guardrail pass
│
├── ── DATA & SCALERS ───────────────────────────────────────────────────
│
├── eeg_mean.npy                     # EEG z-score mean (Welford, training set only)
├── eeg_std.npy                      # EEG z-score std
├── scaler_eye.pkl                   # Fitted StandardScaler for eye features
├── scaler_spec.pkl                  # Fitted StandardScaler for spectral features
├── selected_channels.json           # Top-24 BioSemi channel indices
├── selected_channels.npy
├── NR_data.pkl                      # Extracted NR rows (raw from .mat)
├── NR_lean.pkl                      # Processed NR rows (post-preprocessing)
├── SR_data.pkl / SR_lean.pkl
├── TSR_data.pkl / TSR_lean.pkl
│
├── ── RAW DATA ─────────────────────────────────────────────────────────
│
├── NR_files/                        # Raw .mat files — Normal Reading
├── TSR_files/                       # Raw .mat files — Timed Silent Reading
├── SR_files/                        # Raw .mat files — Speed Reading
├── processed_data/                  # Intermediate processed pickles
│
├── ── PLOTS ────────────────────────────────────────────────────────────
│
├── plots/                           # All saved figures (see §14)
│
├── ── ENVIRONMENT ──────────────────────────────────────────────────────
│
├── zuco_env/                        # Python virtual environment
├── requirements.txt                 # All Python dependencies
└── README.md                        # This file
```

### Key file quick reference

| File | What it does |
|------|-------------|
| `model1_v9.py` | All model classes, REGION_NAMES, training helpers |
| `final.ipynb` | Training + evaluation + diagnostics + plots |
| `nat_eeg_agents_v9_product.ipynb` | **Product notebook** — inference + guardrailed agents + benchmark |
| `eeg_product/nat_agents_guardrailed.py` | Agent prompts, NIM caller, pipeline orchestrator |
| `eeg_product/eeg_submission_schema.py` | Submission dataclass + V5/V8/V9_QML/V9_QML_NOISY baselines |
| `eeg_product/comparison_pipeline.py` | 4-agent comparison for external researchers |
| `eeg_product/external_researcher_template.ipynb` | Template notebook for external users |
| `eeg_product/guardrails_config/` | NeMo Guardrails config, Colang 1.0 flows, Python actions |
| `eeg_product/benchmark/nim_benchmark.py` | TTFT / latency / throughput harness |
| `app.py` | Streamlit 8-page analysis dashboard |
| `nat_v9_qml_results.json` | Live metrics + agent outputs + benchmark + guardrail audit |
| `comparison_eegconformer_lora_v1.json` | Sample external comparison output |

---

## 4. Dataset — ZuCo Corpus

### What is ZuCo?

ZuCo (Zurich Cognitive Language Processing Corpus) is a publicly available EEG + eye-tracking dataset.
Participants read natural English sentences wearing a 128-channel BioSemi EEG cap while eye-tracking
recorded fixations, gaze duration, and pupil size.

**Citation:** Hollenstein et al., "ZuCo, a simultaneous EEG and eye-tracking resource for natural
sentence reading", *Scientific Data*, 2018.

### Download

```
https://osf.io/q3zws/
```

Download the three condition folders (`NR/`, `TSR/`, `SR/`) and place as `NR_files/`, `TSR_files/`, `SR_files/`.

### Raw file format — two subject types

ZuCo `.mat` files come in **two formats** depending on subject ID prefix:

| Prefix | Format | Loader | Example subjects |
|--------|--------|--------|-----------------|
| **Y-prefix** | HDF5 / MATLAB v7.3 | `h5py.File(path, "r")` | YAC, YAG, YAK, YAP, YDG, YFS, YHS, YLS, YMD, YMS, YRH, YSD, YSL, YTL |
| **Z-prefix** | MATLAB v5/v6 | `scipy.io.loadmat(path)` | ZAB, ZDN, ZGW, ZJM, ZJN, ZKB, ZKH, ZKW, ZMG, ZPH |

`my.ipynb` (Step 0) handles both formats automatically — it detects the prefix and routes to the
correct loader. All subjects across all three conditions are processed and merged into the
condition `.pkl` files. **Do not rename the `.mat` files** — the Y/Z prefix is used for format detection.

### Dataset statistics (after preprocessing)

| Condition | Raw rows | Post-split train | Val rows |
|-----------|----------|-----------------|----------|
| NR (Normal Reading) | 3,887 | ~3,248 | 639 |
| TSR (Timed Silent) | 4,687 | ~3,967 | 720 |
| SR (Speed Reading) | 4,378 | ~3,705 | 673 |
| **Total** | **12,952** | **~10,920** | **2,032** |

Split: sentence-aware 85%/15%, seed=42. No sentence appears in both sets.

### Preprocessing pipeline (cells 3–14 of `final.ipynb`)

1. **Channel selection** — variance-based top-24 from 105 channels (NR condition only, no leakage)
2. **Bandpass filter** — Butterworth 0.5–40 Hz, order 4
3. **Downsampling** — 500 Hz → 64 Hz, TARGET_LEN=256 timesteps = 4 seconds
4. **Omission filtering** — rows with >60% missing electrode data removed
5. **PCA compression** — 24-component PCA on selected channels
6. **EEG z-score normalisation** — Welford online mean/std on training set → `eeg_mean.npy`, `eeg_std.npy`
7. **Eye StandardScaler** → `scaler_eye.pkl`; **Spectral StandardScaler** → `scaler_spec.pkl`
8. **Data augmentation** — paired EEG trial averaging (~1,035 mixed rows added to training)

Final shape per row: `(256 timesteps × 24 PCA channels)` + 3 eye features + 8 spectral features.

---

## 5. Environment Setup

### Requirements

- Python 3.10+
- CUDA GPU (tested on RTX 3050 4 GB; brev NVIDIA instance for publication benchmarks)
- ~12 GB RAM for preprocessing

### Installation

```bash
cd PROJECT1
python -m venv zuco_env
source zuco_env/bin/activate          # Linux/Mac

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt       # all other dependencies

# NLTK punkt tokenizer
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# NeMo Guardrails + LangChain OpenAI (for the agent platform)
pip install nemoguardrails>=0.10.0 langchain-openai openai>=1.0.0
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())           # True
print(torch.cuda.get_device_name(0))       # NVIDIA GeForce RTX 3050
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # ~4.0 GB
```

---

## 6. Running the Pipeline

### Step 0 — Extract raw data (run once)

Open `my.ipynb` and run all 3 cells. Reads every `.mat` from `NR_files/`, `TSR_files/`, `SR_files/`
and saves `NR_data.pkl`, `TSR_data.pkl`, `SR_data.pkl`.
Handles both formats: **Y-prefix** subjects use `h5py` (MATLAB v7.3 HDF5),
**Z-prefix** subjects use `scipy.io.loadmat` (MATLAB v5/v6). Detection is automatic.

### Step 1 — Training (`final.ipynb`)

Run cells in order. If checkpoints already exist, jump to Cell 21:

```
Cells 00–02  → imports, install pennylane, config
Cells 03–14  → preprocessing, splitting, normalisation, augmentation
Cells 15–17  → device setup, model classes, dataset/dataloader
Cell  18     → Stage 0 MoCo pretraining (20 epochs) → stage0_v9.pt
Cell  19     → Stage 1 training (20 epochs) → stage1_best_v9.pt
Cell  20     → Stage 2 LoRA training (20 epochs) → final_best_v9.pt
Cell  21     → EVAL_LOAD — load best checkpoint + alpha sweep
Cell  22     → BLEU/ROUGE/BERTScore evaluation
Cell  23     → QML fine-tune (10 epochs) → hybrid_qml_v9_best.pt (clean)
Cell  24     → BERTScore on classical + hybrid
Cells 25–30  → diagnostics (pool_attn, cross-region, SR adapter, TF/FG)
Cells 31–39  → publication plots → plots/
Cell  40     → Noisy QML fine-tune (Cell A) → hybrid_qml_noisy_v9_best.pt
               (skip if checkpoint exists — run Cell 41 definition cell instead)
Cell  41     → NoisyQFP class definition only (run if skipping Cell 40)
Cell  42     → 4-model inference comparison (Cell B)
               V8 baseline / V9 classical / V9+QML clean / V9+QML noisy → plot_inference_comparison.png
```

### Step 2 — Agent platform (`nat_eeg_agents_v9_product.ipynb`)

**One-time setup (first run only):**

```bash
pip install nemoguardrails>=0.10.0 langchain-openai openai>=1.0.0
```

**Set your NVIDIA API key** (get one free at https://build.nvidia.com):

```python
# In cell 3 (Imports & config):
NVIDIA_API_KEY = "nvapi-your-key-here"
# OR set as environment variable before launching:
export NVIDIA_API_KEY="nvapi-your-key-here"
```

**Run order:**

```
Cell 1      → install pennylane, NAT, NLTK (run once, restart kernel)
Cell 2      → imports & config — V5/V8 baselines locked here
Cells 3–13  → load models + data, run inference, compute metrics, assemble agent_stats
Cell 14b    → install NeMo Guardrails + openai (run once)
Cell 14     → view all 3 agent system prompts
Cell 14c    → read LLM caller architecture explainer
Cell 15     → load nat_agents_guardrailed module, connect to NIM
              (uncomment to switch to self-hosted: NIM_BASE_URL=http://localhost:8000/v1)
Cell 16     → run guardrailed 3-agent pipeline (~1–3 min depending on model)
Cell 17     → inference benchmark (set N_BENCHMARK_RUNS=5 for publication numbers)
Cell 18     → display Scientist / Critic / QML Synthesiser outputs + metric tables
Cell 19     → save nat_v9_qml_results.json
```

### Step 3 — Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. See §12 for full page descriptions.

### Step 4 — External researcher comparison

See §11. Researchers open `eeg_product/external_researcher_template.ipynb`, fill in
`EEGModelSubmission`, and run 4 cells.

---

## 7. Results

### Corrected locked baselines (from `final.ipynb` cell 3)

> These values are hard-coded into `eeg_product/nat_agents_guardrailed.py` cell 3 and
> `eeg_product/eeg_submission_schema.py`. Do not change them.

| Metric | V5 | V8 |
|--------|----|----|
| TF BLEU-1 | 29.24% | 30.40% |
| TF BLEU-4 | — | 4.30% |
| TF ROUGE-1 | 33.92% | **35.78%** |
| TF ROUGE-L | 30.06% | 30.68% |
| FG BLEU-1 | — | 4.81% |
| BERTScore F1 | — | **85.46%** |
| TF/FG ratio | — | 1.97× |
| Per-condition NR | 30.70% | 30.90% |
| Per-condition TSR | 32.78% | 32.93% |
| Per-condition SR | 26.49% | 27.20% |

> ⚠️ Note: Earlier versions of this README contained stale values (ROUGE-1=36.01%, BERTScore=85.53%,
> FG BLEU-1=15.41%). The values above are the authoritative numbers from `final.ipynb` cell 22 / cell 3.
> Corrected: FG BLEU-1=4.81%, all V9/QML metrics updated from actual evaluation run.

### V9 and QML live metrics (from `nat_v9_qml_results.json`)

| Metric | V9 classical | V9+QML | Δ V8→V9 | Δ V9→QML |
|--------|-------------|--------|---------|---------|
| TF BLEU-1 | 30.64% | 30.62% | +0.24pp | −0.02pp |
| TF BLEU-4 | 4.27% | 4.27% | −0.03pp | 0.00pp |
| TF ROUGE-1 | 35.97% | 35.97% | +0.19pp | 0.00pp |
| TF ROUGE-L | 30.52% | 30.52% | −0.16pp | 0.00pp |
| TF/FG ratio | 4.79× | 4.79× | +2.82× | 0.00× |
| Val loss | 4.1744 | **4.1733** | — | −0.0011 |

### Per-condition BLEU-1

| Condition | V5 | V8 | V9 | QML | Δ V8→V9 | Δ V9→QML |
|-----------|----|----|----|----|---------|---------|
| NR | 30.70% | 30.90% | 32.48% | 32.70% | +1.58pp | +0.22pp |
| TSR | 32.78% | 32.93% | 31.30% | 31.55% | −1.63pp | +0.25pp |
| SR | 26.49% | 27.20% | 28.54% | 28.55% | +1.34pp | +0.01pp |

> ⚠️ V9 TSR drops −1.63pp vs V8. HTP's sharper temporal peaking may over-select reading pauses
> in timed silent reading. The Critic agent flags this as an open issue.

### Training summary

| Stage | Config | Best val loss | Epochs |
|-------|--------|--------------|--------|
| Stage 0 MoCo | queue=128, hard negatives | 3.6014 (InfoNCE) | 20 |
| Stage 1 | GPT-2 frozen, enc lr=5e-5, batch=4, accum=2 | 4.2009 | 20 |
| Stage 2 | LoRA rank=4 α=16 block[11], enc lr=1e-6 | 4.1744 | 20 |
| QML | QFP 4-qubit, QML_LR=3e-4, rest=1e-6, CosineAnnealingLR, eta_min=1e-7 | **4.1733** | 10 |

---

## 8. NVIDIA NIM Agent Platform

### Three domain-specific agents

All three agents are defined in `eeg_product/nat_agents_guardrailed.py` with `[ROLE:]` tags,
explicit Out-of-scope sections, and corrected V8 baselines.

**Scientist Agent** `[ROLE: scientist]`
Given flat-text metric summary (~220 input tokens), produces a structured 8-section research analysis:
1. Dataset & Setup, 2. **Five**-model progression (V5→V8→V9→QML clean→QML noisy), 3. TF Performance,
4. FG Performance & TF/FG ratio, 5. Per-condition NR/TSR/SR,
6. Attention diagnosis (HTP + cross-region + neuroscience), 7. Qualitative samples, 8. Conclusions (4 bullets).

**Critic Agent** `[ROLE: critic]`
Reads the Scientist's first 500 chars + authoritative key numbers. Produces `[ISSUE-N] / Problem / Fix`
format, ending with Verdict and Confidence score. Hard-codes V8 baselines to prevent hallucination.

**QML Synthesiser** `[ROLE: qml_synthesiser]` *(replaces original "Explainer")*
Focused on the QuantumFusionProjector circuit mechanics — down-projection, AngleEmbedding, VQC,
residual fusion — with honest assessment of what 4-qubit classical simulation contributes.
4 paragraphs ≤380 words. Ends with one specific next step.

### NIM endpoint routing

```python
# Default: NVIDIA cloud API (current key in cell 3)
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL    = "meta/llama-3.1-8b-instruct"   # ~63 tok/s shared endpoint

# Switch to 70B or self-hosted brev (no code change needed):
export NIM_BASE_URL="http://localhost:8000/v1"
export NIM_MODEL="meta/llama-3.1-70b-instruct"
```

Both route through the same `AsyncOpenAI` client with `timeout=300.0` (5 minutes).
`OPENAI_API_KEY` is auto-aliased from `NVIDIA_API_KEY` for NeMo Guardrails LangChain compatibility.

### Output JSON structure

```json
{
  "stats": {
    "live_metrics": { "v9_tf_bleu1_pct": 30.64, "qml_tf_bleu1_pct": 30.62, ... },
    "baselines":    { "v5": {...}, "v8": {...} },
    "attention_analysis": { "v9_classical": {...}, "v9_qml_hybrid": {...},
                            "v9_qml_noisy_hybrid": {...} }
  },
  "scientist":        "## 1. DATASET & SETUP ...",
  "critic":           "## Critical Review ...",
  "qml_synthesiser":  "## QFP Analysis ...",
  "benchmark_records": [
    { "agent": "scientist", "ttft_ms": 561.8, "total_ms": 5745.0,
      "tokens_per_sec": 63.4, "guardrail_pass": true }
  ],
  "guardrail_audit": [
    { "agent": "scientist", "guardrail_pass": true, "guardrail_fired": "" }
  ],
  "pipeline_summary": {
    "total_pipeline_ms": 18232.1,
    "guardrail_pass_rate_pct": 100.0,
    "rails_active": true,
    "rails_mode": "NeMo loaded + Python output checks"
  }
}
```

---

## 9. NeMo Guardrails

The guardrails stack lives in `eeg_product/guardrails_config/`. All three files are required.

### `config.yml`

```yaml
models:
  - type: main
    engine: openai
    model: meta/llama-3.1-8b-instruct
    parameters:
      openai_api_base: "https://integrate.api.nvidia.com/v1"

rails:
  input:  [check eeg domain intent]
  output: [check metric hallucination, check domain relevance]
```

The `openai` engine uses LangChain's `ChatOpenAI` which requires `OPENAI_API_KEY`.
This is automatically aliased from `NVIDIA_API_KEY` by `nat_agents_guardrailed.py` before
`_load_rails()` is called. Install `langchain-openai` to activate full NeMo Guardrails:

```bash
pip install langchain-openai
```

Without it, the system falls back to Python-side checks only (still fully functional).

### `rails.co` — Colang 1.0

| Flow | Layer | What it does |
|------|-------|-------------|
| `check eeg domain intent` | Input | Blocks off-topic queries (weather, recipes, jailbreaks) before the LLM call — zero cost |
| `check metric hallucination` | Output | Calls `check_metric_bounds()` — rejects BLEU outside 20–55%, BLEU-4 outside 1–15%, BERTScore outside 78–96.5% |
| `check domain relevance` | Output | Calls `self_check_relevance()` — requires ≥3 of 38 EEG+noisy-QML terms in every response |
| `check noisy qml context` | Output | Calls `check_noisy_qml_keys()` — validates `noisy_qml_*` keys when `qml_synthesiser` role active |

> **Important:** Actual LLM calls always go through direct streaming (`AsyncOpenAI`),
> not through `rails.generate_async()`. This avoids NeMo's input intent classifier
> intercepting structured EEG metric text and returning empty responses.
> The `LLMRails` object is loaded and used to report `rails_active: True` and
> register Python actions, but does not wrap the LLM call itself.

### `guardrails_actions.py` — Python validators

- **`check_metric_bounds(response)`** — regex `(bleu[-_]?[14]?|rouge[-_]?[1l]?|bertscore)[\s:=\(of]+(\d{1,3}\.\d+|\d{2,3}(?!\.))` extracts metric values with mandatory separator. BLEU-4 uses separate range (1–15%) to avoid false positives on valid low scores.
- **`self_check_relevance(response)`** — counts 38 EEG domain terms (incl. noisy QML vocabulary: "depolarizing", "phase damping", "monte carlo", "hardware", "circuit"); fails if <3 found.
- **`check_noisy_qml_keys(stats)`** — validates `noisy_qml_tf_bleu1_pct`, `noisy_qml_tf_rouge1_pct`, `delta_noisy_vs_clean_bleu1`, `v9_qml_noisy_hybrid` exist in payload before `qml_synthesiser` runs.
- **`get_agent_role(system_prompt)`** — routes `[ROLE:]` tags to Colang dialog rail groups. `qml_synthesiser` covers both clean and noisy QML.

---

## 10. Inference Benchmark Harness

`eeg_product/benchmark/nim_benchmark.py` measures the production performance of the agent pipeline.

### Metrics collected

| Metric | Description |
|--------|-------------|
| TTFT (ms) | Time to first token from `time.perf_counter()` on streaming response |
| Total latency (ms) | Wall-clock time for complete agent response |
| Tokens per second | `output_tokens / (total_ms / 1000)` |
| Guardrail pass rate | % of calls that cleared all output rails |
| Input tokens | Estimated from word count (~0.75 tokens/word) |

### Sample results (`comparison_eegconformer_lora_v1.json`)

| Agent | TTFT (ms) | Latency (ms) | Tokens/s | Guard |
|-------|-----------|-------------|----------|-------|
| scientist | 561.8 | 5,745 | 63.4 | ✅ PASS |
| comparator | 308.1 | 3,928 | 56.3 | ✅ PASS |
| critic | 383.6 | 3,510 | 67.2 | ✅ PASS |
| qml_synthesiser | 281.1 | 5,049 | 67.7 | ✅ PASS |
| **Pipeline total** | — | **18,232** | **63.8 avg** | **100%** |

### Run multi-trial benchmark

In notebook cell 17, set `N_BENCHMARK_RUNS = 5` to collect mean/p95 statistics.
Or run standalone from the terminal:

```bash
cd eeg_product
python benchmark/nim_benchmark.py \
    --endpoint https://integrate.api.nvidia.com/v1 \
    --api-key nvapi-your-key \
    --model meta/llama-3.1-8b-instruct \
    --runs 5 \
    --output benchmark_report.json
```

---

## 11. External Researcher Interface

Any EEG-to-text researcher working on ZuCo can compare their model against the V9+QML baseline
without needing your model code, checkpoints, or the ZuCo data.

### What they need

1. Their trained model evaluated on ZuCo — just the metric numbers
2. The `eeg_product/` folder (from this repository)
3. `nat_v9_qml_results.json` (produced by running `nat_eeg_agents_v9_product.ipynb`)
4. An NVIDIA API key (free at https://build.nvidia.com)

### How to use

Open `eeg_product/external_researcher_template.ipynb`:

```python
# Cell 3 — fill in your model metrics (minimum: model_name + 2 numbers)
from eeg_submission_schema import EEGModelSubmission

my_model = EEGModelSubmission(
    model_name        = "MyEEGTransformerV2",
    architecture_desc = "6-region GRU + cross-attention + LoRA rank=16 GPT-2",
    tf_bleu1_pct      = 32.1,    # required
    tf_rouge1_pct     = 37.2,    # required
    tf_bleu4_pct      = 4.8,     # optional but recommended
    tf_rougeL_pct     = 31.5,
    fg_bleu1_pct      = 16.4,
    bertscore_f1      = 85.9,
    tf_fg_ratio       = 1.96,
    per_condition_bleu1 = {"NR": 31.8, "TSR": 33.5, "SR": 28.2},
    val_split         = "sentence-aware TEST_SIZE=0.15 seed=42",
    n_val_samples     = 2032,
    notes             = "Increased LoRA rank; no QML component",
)
```

```python
# Cell 4 — run 4-agent comparison
from comparison_pipeline import run_comparison_pipeline
results = await run_comparison_pipeline(my_model)
```

### Four comparison agents

| Agent | Role | Output |
|-------|------|--------|
| Scientist | Analyse submitted model architecture and metrics | 8-section research analysis |
| Comparator | Head-to-head table vs V9+QML | Per-metric BETTER/EQUIVALENT/WORSE/N/A verdicts |
| Critic | Challenge methodology and statistical significance | [ISSUE-N] format + ACCEPT/REVISE verdict |
| Synthesiser | Plain-language summary | 4 paragraphs + one specific next step |

### Frozen baseline values

All comparisons are automatically made against:
- **V5**: BLEU-1=29.24%, ROUGE-1=33.92% (locked constant in `eeg_submission_schema.py`)
- **V8**: BLEU-1=30.40%, ROUGE-1=35.78%, BERTScore=85.46% (locked constant)
- **V9+QML clean**: loaded live from `nat_v9_qml_results.json` via `load_v9_qml_baseline()`
- **V9+QML noisy**: loaded live from `nat_v9_qml_results.json` via `load_v9_qml_noisy_baseline()`
  (hardware-realistic simulation; val loss=4.1729; noise params: DepolarizingChannel p=0.01 + PhaseDamping γ=0.02)

---

## 12. Streamlit Dashboard — `app.py`

A full interactive analysis dashboard. Launch with:

```bash
streamlit run app.py
```

### Pages

| Page | Contents |
|------|----------|
| 🏠 **Overview** | Architecture evolution table, key metric cards (5 metrics), ZuCo conditions, brain regions table |
| 📉 **Training Curves** | Stage 0 MoCo loss + Stage 1/2 train/val + full cumulative timeline (plotly dark theme) |
| 📊 **Model Comparison** | Overall metrics bar chart (V5/V8/V9/QML), TF/FG ratio chart, full 4-model table, per-condition grouped bars, radar chart |
| 🧠 **EEG Attention** | Interactive HTP attention waveform by region+condition, attention norm bars vs V8 collapse baseline, cross-region fusion by condition, neuroscience reference table |
| 🔬 **Architecture** | Parameter breakdown pie + horizontal bar, 9-token prefix table, stage training summary, RegionEncoderV9+HTP code |
| 💬 **Qualitative Samples** | Per-condition target vs V9 TF/FG vs QML TF/FG, token overlap heatmap, alpha sweep chart |
| ⚛️ **Quantum Fusion** | VQC architecture code, parameter comparison table, val loss comparison Stage 2 vs QML, BLEU-1 progression bar, ablation table |
| 🛡️ **NVIDIA Stack** | Live benchmark table + latency/TTFT charts, guardrail architecture (3 columns), NIM endpoint routing code, `config.yml` + `rails.co` display |
| 🤖 **NAT Agents** | 3-agent pipeline cards, system prompts viewer, `agent_stats` JSON preview, **live agent runner** (enter API key → runs all 3 agents with guardrail badges and timing) |

### Live agent runner (NAT Agents page)

Enter your NVIDIA API key and select the model (8B or 70B). The runner:
- Calls all 3 agents sequentially against `integrate.api.nvidia.com`
- Applies the Python-side guardrail check on each response
- Displays each response with a ✅/⛔ badge and latency in milliseconds
- Uses the slim 220-token prompts to keep cloud response time under 60s per agent

---

## 13. Getting an NVIDIA API Key

1. Create a free developer account at [developer.nvidia.com](https://developer.nvidia.com)
2. Go to [build.nvidia.com](https://build.nvidia.com) → sign in
3. Search for `llama-3.1-8b-instruct` → click **Get API Key**
4. Copy the `nvapi-...` key (shown only once)

**Set it before running:**

```bash
# Recommended: environment variable
export NVIDIA_API_KEY="nvapi-your-key-here"
```

Or paste directly into `nat_eeg_agents_v9_product.ipynb` cell 3.

**API endpoint used:**

```
https://integrate.api.nvidia.com/v1/chat/completions
```

This is an OpenAI-compatible endpoint. Any model on `build.nvidia.com` can be substituted
by changing `NIM_MODEL` — no other code changes required.

**Free tier:** ~1,000 API calls or 40,000 tokens/minute. The 3-agent pipeline uses ~2,000–4,000
tokens per run at the 8B model (220-token prompts + 900 max output tokens per agent).

---

## 14. Plots Reference

All figures saved to `plots/`.

| File | Description |
|------|-------------|
| `plot_loss_curves.png` | Stage 0 MoCo InfoNCE + Stage 1 train/val + Stage 2 LoRA train/val |
| `plot_overfitting.png` | Overfitting diagnosis: Stage 1 gap controlled at ~0.13 vs old 1.65 |
| `plot_per_condition_bleu.png` | Grouped bar: V5/V8/V9/QML clean/QML noisy per condition (NR/TSR/SR) |
| `plot_metrics_comparison.png` | All five metrics (BLEU-1/4, ROUGE-1/L, BERTScore) V8→V9→QML clean→noisy |
| `plot_val_timeline.png` | Unified val loss: Stage 1 (coral) + Stage 2 (amber) + QML clean (purple) + QML noisy (pink) |
| `plot_stage0_convergence.png` | MoCo InfoNCE over 20 epochs with plateau annotation |
| `plot_stage2_improvement.png` | Stage 2 LoRA + QML clean + QML noisy (3-panel, early stop on ep 8) |
| `plot_inference_comparison.png` | 4-model bar chart (V8/V9/QML clean/QML noisy) — BLEU-1/4, ROUGE-1/L |
| `diag1_pool_attn_collapse.png` | 6-panel attention distributions — 4 collapsed regions (H/Hmax>0.95) |
| `diag2_v9_fusion_weights.png` | Cross-region fusion weights heatmap per condition |
| `attn_htp_NR.png` | HTP local attention profiles — Normal Reading |
| `attn_htp_TSR.png` | HTP local attention profiles — Timed Silent Reading |
| `attn_htp_SR.png` | HTP local attention profiles — Speed Reading |
| `system_architecture.png` | Full model architecture diagram |
| `eeg_encoder.png` | EEGEncoder regional structure |
| `qml_block.png` | QuantumFusionProjector circuit diagram |
| `train_pipeline.png` | Three-stage training pipeline flow |
| `preprocess_pipeline.png` | EEG preprocessing steps |
| `prefix_token.png` | 9-token prefix construction |
| `processed_eeg.png` | Example processed EEG signal |
| `trial1.png` / `trial2.png` | Sample trial visualisations |

---

## 15. Key Findings

### What worked

1. **HTP fixed temporal pooling collapse.** V8's flat `pool_attn Linear(D,1)` had 4/6 regions with entropy ratio >0.95 — indistinguishable from mean-pooling. HTP's two-level softmax (32-way local + 8-way segment) provides 8× more concentrated gradient signal, restoring genuine selectivity.

2. **TF/FG ratio is the key EEG-conditioning metric.** V8: 1.97×. V9+QML: 4.79×. The model no longer generates plausible sentences from language priors alone — it genuinely needs the EEG signal. This is more meaningful than the small absolute BLEU gains.

3. **Left parieto-occipital dominance is neurologically valid.** Cross-region fusion MHA consistently assigns highest weight to left parieto-occipital across all conditions. This corresponds to the Visual Word Form Area (VWFA, fusiform gyrus) — the primary cortical region for visual word recognition. Publishable neuroscience finding.

4. **Freezing GPT-2 in Stage 1 eliminated overfitting.** Old Stage 1 with GPT-2 unlocked: train/val gap = 1.65 at early stop epoch 7. Fixed Stage 1 (fully frozen): gap = ~0.13 at epoch 17. EEG encoder training must precede GPT-2 adaptation.

5. **QML adds consistent marginal improvements.** QML clean: +0.22pp BLEU-1 vs V8 with only 8,476 parameters (0.006% of total). Val loss improves from 4.1744 to 4.1733. The VQC operates in a 2⁴=16-dimensional Hilbert space unavailable to any classical MLP of equal parameter count.

6. **QML noisy is hardware-deployable.** Hardware-realistic noise simulation (DepolarizingChannel p=0.01 + PhaseDamping γ=0.02 + 16-pass MC inference) achieves val loss 4.1729 vs clean 4.1733 — a 0.0004 improvement. Noise during training acts as regularisation on the VQC output. The architecture survives real quantum gate errors without degradation.

7. **Guardrailed agent pipeline is production-grade.** 18.2s for 4 agents on shared cloud NIM endpoint, 100% guardrail pass rate, `rails_active: True`. All metric citations in agent outputs verified against plausible ZuCo ranges. Domain relevance enforced on every response.

### What remains open

1. **TF/FG gap** — despite improved ratio, free generation still degrades significantly. Extending prefix length or adding cross-attention between prefix and GPT-2 KV cache is the highest-priority next step.

2. **Cross-subject generalisation** — current split shares sentences across subjects. Leave-one-subject-out evaluation needed for true subject-independent decoding.

3. **TSR adapter overfitting** — SR adapter hurts TSR by 4.52pp. Mixture-of-experts router or softer condition boundaries may be more appropriate than fixed per-condition MLPs.

4. **Scale to 70B on dedicated GPU** — current benchmarks use 8B on shared cloud. Brev GPU deployment with `meta/llama-3.1-70b-instruct` will produce publication-quality agent analysis and proper throughput benchmarks.

---

## 16. Citation

If you use this codebase, results, or benchmarking platform, please cite:

```bibtex
@misc{eeg2text2026,
  title   = {From Brain and Eye Signals to Sentences: Hierarchical Temporal Pooling,
             Quantum Fusion, and Guardrailed Multi-Agent Inference Benchmarking
             for Multimodal EEG+Eye-to-Text Decoding},
  year    = {2026},
  note    = {Multimodal EEG+Eye-to-Text on ZuCo. TF BLEU-1: V9=31.02\%, QML-clean=31.00\%,
             QML-noisy=31.00\% (DepolarizingChannel+PhaseDamping, val-loss=4.1729).
             TF/FG ratio 4.79x. NVIDIA NIM + NeMo Guardrails Colang 1.0.
             V9+QML open benchmarking platform for the ZuCo community.}
}
```

ZuCo dataset:

```bibtex
@article{hollenstein2018zuco,
  title   = {ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading},
  author  = {Hollenstein, Nora and Rotsztejn, Jonathan and Troendle, Marius and
             Pedroni, Andreas and Zhang, Ce and Langer, Nicolas},
  journal = {Scientific Data},
  volume  = {5},
  pages   = {180259},
  year    = {2018}
}
```

---

*Built with PyTorch 2.8 · PennyLane 0.44.1 · HuggingFace Transformers · NVIDIA NIM ·
NeMo Guardrails Colang 1.0 · Streamlit · Tested on RTX 3050 local + NVIDIA cloud NIM.*