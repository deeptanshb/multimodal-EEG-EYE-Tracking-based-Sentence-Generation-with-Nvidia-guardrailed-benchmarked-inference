# EEG2Text — Brain-to-Language Decoding with Quantum-Classical Hybrid AI

> Decoding natural language from EEG brain signals using a multimodal transformer architecture, hierarchical temporal pooling, LoRA fine-tuning, and a 4-qubit variational quantum circuit — evaluated on the ZuCo corpus across four model generations (V5 → V8 → V9 → QML hybrid).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Evolution](#2-architecture-evolution)
3. [Repository Structure](#3-repository-structure)
4. [Dataset — ZuCo Corpus](#4-dataset--zucocorpus)
5. [Environment Setup](#5-environment-setup)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Results](#7-results)
8. [NVIDIA NIM Agent Pipeline](#8-nvidia-nim-agent-pipeline)
9. [Getting an NVIDIA API Key](#9-getting-an-nvidia-api-key)
10. [Plots Reference](#10-plots-reference)
11. [Key Findings](#11-key-findings)
12. [Citation](#12-citation)

---

## 1. Project Overview

This project implements an end-to-end brain-computer interface (BCI) pipeline that:

- Reads EEG signals recorded while participants silently read sentences
- Preprocesses, filters, and compresses the neural signals into a structured feature set
- Trains a multimodal transformer to decode the original sentence from brain activity
- Evaluates decoding quality using BLEU, ROUGE, and BERTScore
- Runs a three-agent NVIDIA NIM analysis pipeline (Scientist → Critic → Explainer) to auto-interpret the results

**Dataset:** ZuCo (Zurich Cognitive Language Processing Corpus) — 12 subjects, ~700 unique sentences, three reading conditions.

**Key result:** V9+QML hybrid achieves **TF BLEU-1 = 30.86%** and **BERTScore F1 = 85.51%** on the held-out validation split (n=2,032), improving over the V5 baseline (29.24%) through four successive architectural refinements.

---

## 2. Architecture Evolution

### V5 — Baseline

- Conv1D + Bidirectional GRU EEG encoder
- Single mean-pooled EEG vector as prefix
- Prefix-tuned DistilGPT-2 decoder
- No eye-tracking or spectral features
- TF BLEU-1: **29.24%** | ROUGE-1: **33.92%**

### V8 — Multimodal Regional Encoder

Key additions over V5:

- **6 parallel GRU-Transformer RegionEncoders** — one per anatomical brain region (left temporal, left parietal, left parieto-occipital, central parietal, right parietal, right parieto-occipital)
- **MoCo Stage 0** contrastive pretraining (queue=128, condition-based hard negatives)
- **LoRA fine-tuning** on GPT-2 blocks [10, 11] (rank=8)
- **SR condition adapter** — three separate MLPs for NR / TSR / SR reading conditions
- **Eye-tracking encoder** (fixations, pupil size, duration)
- **Spectral band encoder** (alpha, beta, gamma, theta)
- **Word spectral encoder** (per-word band arrays)
- **Diagnosis finding:** `pool_attn` collapsed to uniform 1/256 in 4 of 6 regions — effectively mean-pooling despite the transformer stack
- TF BLEU-1: **30.40%** | BERTScore F1: **85.53%**

### V9 — Hierarchical Temporal Pooling (HTP)

Key fix over V8:

- **HierarchicalTemporalPooling** replaces the flat `pool_attn Linear(D,1)`:
  - Level 1: 32-way local softmax within 8 windows of 32 timesteps each
  - Level 2: 8-way segment softmax across windows
  - Gradient signal concentrated 8× vs the collapsed 256-way softmax
- **RegionEncoderV9** is a drop-in replacement for RegionEncoder
- **Stage 1 fix:** GPT-2 fully frozen, EEG encoder only (train/val gap reduced from 1.65 → ~0.13)
- **Stage 2 fix:** LoRA rank=4, block [11] only, encoder near-frozen (lr=1e-6)
- TF BLEU-1: **30.55–30.71%** | ROUGE-1: **35.96%**

### V9 + QML Hybrid — Quantum Fusion Projector

Key addition over V9:

- **QuantumFusionProjector (QFP)** inserted after `enc_proj_norm` in `_encode_eeg`:
  - Linear down: H=768 → 4
  - `AngleEmbedding` (RY rotations) encodes compressed EEG into 4-qubit state
  - 2× `StronglyEntanglingLayers` (CNOT ladders + rotation gates)
  - 4 Pauli-Z expectations → Linear up: 4 → H=768
  - LayerNorm residual connection
  - ~8,476 QML parameters (tiny on top of 147M total)
- **PennyLane** `lightning.qubit` simulator with adjoint differentiation
- 10-epoch fine-tune: QML_LR=3e-4, rest=1e-6, CosineAnnealingLR, patience=3
- TF BLEU-1: **30.86%** | ROUGE-1: **36.03%** | BERTScore F1: **85.51%**

---

## 3. Repository Structure

```
PROJECT1/
├── NR_files/                    # Raw .mat files — Normal Reading condition
├── TSR_files/                   # Raw .mat files — Timed Silent Reading
├── SR_files/                    # Raw .mat files — Speed Reading
├── processed_data/              # Intermediate processed pickles
├── plots/                       # All saved figures (see §10)
│   ├── attn_htp_NR.png
│   ├── attn_htp_SR.png
│   ├── attn_htp_TSR.png
│   ├── diag1_pool_attn_collapse.png
│   ├── diag2_v9_fusion_weights.png
│   ├── eeg_encoder.png
│   ├── plot_loss_curves.png
│   ├── plot_metrics_comparison.png
│   ├── plot_overfitting.png
│   ├── plot_per_condition_bleu.png
│   ├── plot_stage0_convergence.png
│   ├── plot_stage2_improvement.png
│   ├── plot_val_timeline.png
│   ├── prefix_token.png
│   ├── preprocess_pipeline.png
│   ├── processed_eeg.png
│   ├── qml_block.png
│   ├── system_architecture.png
│   ├── train_pipeline.png
│   ├── trial1.png
│   └── trial2.png
├── zuco_env/                    # Conda/venv environment
├── eeg_mean.npy                 # EEG z-score mean (from training set)
├── eeg_std.npy                  # EEG z-score std  (from training set)
├── final_best_v9.pt             # Best Stage 2 checkpoint
├── hybrid_qml_v9_best.pt        # Best QML hybrid checkpoint
├── model1_v9.py                 # Model architecture (V9 + QML classes)
├── final.ipynb                  # Main training + evaluation notebook
├── nat_eeg_agents_v9_updated.ipynb  # NVIDIA NIM three-agent analysis
├── my.ipynb                     # ZuCo .mat → pickle extractor
├── nat_v9_qml_results.json      # Agent pipeline output JSON
├── NR_data.pkl                  # Extracted NR rows (raw)
├── NR_lean.pkl                  # Processed NR rows (post-preprocessing)
├── SR_data.pkl
├── SR_lean.pkl
├── TSR_data.pkl
├── TSR_lean.pkl
├── scaler_eye.pkl               # Fitted StandardScaler for eye features
├── scaler_spec.pkl              # Fitted StandardScaler for spectral features
├── selected_channels.json       # Top-24 BioSemi channel indices
├── selected_channels.npy
├── stage0_v9.pt                 # Stage 0 MoCo checkpoint
└── stage1_best_v9.pt            # Stage 1 best checkpoint
```

### Key files

| File | Purpose |
|------|---------|
| `model1_v9.py` | All model classes: HTP, RegionEncoderV9, EEG2TextTransformerV9, QuantumFusionProjector, MoCo, training helpers |
| `final.ipynb` | 40-cell notebook: preprocessing → Stage 0/1/2 training → QML fine-tune → evaluation → diagnostics → plots |
| `nat_eeg_agents_v9_updated.ipynb` | 17-cell notebook: loads trained models, runs inference on val set, calls NVIDIA NIM three-agent pipeline |
| `my.ipynb` | 3-cell notebook: extracts raw ZuCo .mat files into structured Python pickles |
| `nat_v9_qml_results.json` | Saved agent outputs + full metric table (V5/V8/V9/QML) |

---

## 4. Dataset — ZuCo Corpus

### What is ZuCo?

ZuCo (Zurich Cognitive Language Processing Corpus) is a publicly available EEG+eye-tracking dataset recorded at the University of Zurich. Participants read natural English sentences on a screen while wearing a 128-channel BioSemi EEG cap. The dataset includes simultaneously recorded eye-tracking data (fixations, pupil size, gaze duration).

**Citation:** Hollenstein et al., "ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading", *Scientific Data*, 2018.

### Download

The dataset is available from the OSF repository:

```
https://osf.io/q3zws/
```

Download the three condition folders:

- `NR/` — Normal Reading (participants read for comprehension)
- `TSR/` — Timed Silent Reading (reading under time pressure)
- `SR/` — Speed Reading (fast reading, reduced fixations)

Place them as `NR_files/`, `TSR_files/`, `SR_files/` in the project root.

### Dataset statistics (after preprocessing)

| Condition | Rows | Subjects | Sentences |
|-----------|------|----------|-----------|
| NR | 3,887 | 12 | ~400 |
| TSR | 4,687 | 12 | ~400 |
| SR | 4,378 | 12 | ~400 |
| **Total** | **12,952** | **12** | **~700 unique** |

Train/val split: sentence-aware (no sentence appears in both sets), 85%/15%, seed=42. This gives ~10,920 train rows and ~2,032 val rows.

### Raw data format

Each `.mat` file contains one subject's recordings for one condition. Two file formats exist:

- **Z-prefix subjects** (e.g. `resultsZAB_NR.mat`) — MATLAB v5 format → loaded with `scipy.io.loadmat`
- **Y-prefix subjects** — MATLAB v7.3 / HDF5 format → loaded with `h5py`

The extractor in `my.ipynb` handles both automatically.

### What is extracted per sentence

```python
{
    'subject_id':   'ZAB',
    'condition':    0,          # 0=NR, 1=TSR, 2=SR
    'sentence_idx': 42,
    'sentence':     'Henry Ford founded ...',
    'raw_eeg':      np.array,  # (channels, timesteps) — full 105-ch sentence EEG
    'words': [
        {
            'word':  'Henry',
            'eeg':   np.array,  # fixedLenthEEG — 256 timestep word EEG
            'eye':   {'nFixations': 1.0, 'FFD': 180.0, 'TRT': 220.0, ...},
            'spec':  {'theta': [...], 'alpha1': [...], ...},  # band power arrays
        },
        ...
    ]
}
```

### Preprocessing pipeline (`final.ipynb` cells 3–14)

1. **Channel selection** — variance-based selection of top 24 channels from 105 (BioSemi 128, minus 23 reference/bad channels), using NR condition only to avoid data leakage
2. **Bandpass filter** — Butterworth 0.5–40 Hz, order 4 (removes DC drift and high-frequency noise)
3. **Downsampling** — 500 Hz → 64 Hz (TARGET_FS=64), TARGET_LEN=256 timesteps = 4 seconds
4. **Omission rate filtering** — rows with >60% missing electrode data removed
5. **PCA compression** — 24-component PCA applied to the 24 selected channels (N_PCA_COMPONENTS=24)
6. **EEG z-score normalisation** — online Welford mean/std computed on training set only, saved as `eeg_mean.npy` / `eeg_std.npy`
7. **Eye-tracking scaling** — StandardScaler on (n_fixations, mean_fix_duration, mean_pupilsize), saved as `scaler_eye.pkl`
8. **Spectral scaling** — StandardScaler on 8 mean band-power features, saved as `scaler_spec.pkl`
9. **Data augmentation** — pairs of EEG trials for the same sentence are averaged to create synthetic rows (~1,035 mixed rows added to training set)

Final EEG shape per row: `(256 timesteps × 24 channels)`.

---

## 5. Environment Setup

### Requirements

- Python 3.10+
- CUDA GPU (tested on RTX 3050 4GB — tight but workable with checkpointing)
- ~12 GB RAM for preprocessing

### Installation

```bash
# 1. Clone / download the project
cd PROJECT1

# 2. Create environment
python -m venv zuco_env
source zuco_env/bin/activate        # Linux/Mac
# zuco_env\Scripts\activate         # Windows

# 3. Core dependencies
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt    #for all dependencies needed at one run


# 4. NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # NVIDIA GeForce RTX 3050
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # ~4.0 GB
```

---

## 6. Running the Pipeline

### Step 0 — Extract raw data (run once)

Open `my.ipynb` and run all 3 cells. This reads every `.mat` file from `NR_files/`, `TSR_files/`, `SR_files/` and saves three pickle files:

```
NR_data.pkl   (~X MB)
TSR_data.pkl
SR_data.pkl
```

Both scipy and h5py extractors are called automatically based on subject ID prefix (Z → scipy, Y → h5py).

### Step 1 — Preprocessing + training (`final.ipynb`)

Run cells in order. On a fresh kernel, the minimum sequence to get to training is:

```
Cell 00  ENV_SETUP          — imports
Cell 01  QML_INSTALL        — installs pennylane
Cell 02  CONFIG             — dataset paths, hyperparameters
Cell 03  SIGNAL_UTILS       — filter/resample functions
Cell 04  DATA_LOAD_STEP1    — channel selection from NR
Cell 05  DATA_PROCESS_STEP2 — EEG processing per condition
Cell 06  DATA_MERGE_SPLIT   — load lean pickles, train/val split
Cell 07  EEG_SCALING        — Welford normaliser
Cell 08  EYE_SPEC_SCALING   — StandardScaler for eye + spectral
Cell 09  PIPELINE_SUMMARY   — verify shapes
Cell 13  RAM_CHECK
Cell 14  MIXED_ROWS         — data augmentation
Cell 15  DEVICE_SETUP       — torch device
Cell 16  QML_MODULE         — QuantumFusionProjector definition
Cell 17  DATASET_LOADER     — MultimodalEEGDataset, DataLoaders
Cell 18  MODEL_INIT_S0      — Stage 0 MoCo pretraining
Cell 19  MODEL_LOAD_S1      — Stage 1 teacher-forcing (GPT-2 frozen)
Cell 20  MODEL_LOAD_S2      — Stage 2 LoRA fine-tuning
Cell 21  EVAL_LOAD          — load best checkpoint, alpha sweep
Cell 22  BLEU_ROUGE_EVAL    — TF BLEU/ROUGE/BERTScore
Cell 23  QML_FINETUNE       — hybrid QML training + head-to-head
Cell 24  HYBRID_BERTSCORE   — BERTScore on classical + hybrid
```

**If checkpoints already exist** (skip retraining), jump directly to:

```
Cell 00, 01, 02, 15, 16, 17 → Cell 21
```

Cell 21 loads `stage1_best_v9.pt` → applies LoRA → loads `final_best_v9.pt` and sets `model` in scope. All diagnostic and plot cells then work.

### Step 2 — Diagnostics (`final.ipynb` cells 25–30)

| Cell | What it does |
|------|-------------|
| 25 (DIAG-1) | Attention entropy per region — detects 1/T collapse |
| 26 (DIAG-2) | Cross-region fusion weights heatmap per condition |
| 27 (DIAG-3) | SR adapter ablation (with vs without) |
| 28 (DIAG-4) | Component complexity table |
| 29 (DIAG-6) | TF vs free-generation BLEU gap |
| 30 (ATTN_VIZ) | HTP attention weight plots per condition |

### Step 3 — Publication plots (`final.ipynb` cells 31–39)

Run cells 31–39 in order. They depend on the training data arrays defined in cell 31. All seven figures are saved to the `plots/` folder.

### Step 4 — NVIDIA NIM agent analysis (`nat_eeg_agents_v9_updated.ipynb`)

See Section 8 for full instructions.

---

## 7. Results

### Four-model progression (val n=2,032)

| Metric | V5 | V8 | V9 classical | V9+QML |
|--------|----|----|-------------|--------|
| TF BLEU-1 | 29.24% | 30.40% | 30.55% | **30.86%** |
| TF BLEU-4 | — | 4.30% | 4.28% | **4.46%** |
| TF ROUGE-1 | 33.92% | 35.78% | 35.96% | **36.03%** |
| TF ROUGE-L | — | 30.68% | 30.58% | **30.83%** |
| FG BLEU-1 | — | — | 6.37% | 6.88% |
| TF/FG ratio | — | — | 4.80× | 4.49× |
| BERTScore F1 | — | 85.53% | 85.50% | **85.51%** |

### Per-condition BLEU-1 (V9+QML)

| Condition | V8 | V9+QML | Δ |
|-----------|-----|--------|---|
| NR (Normal Reading) | 30.90% | 31.30% | +0.40pp |
| TSR (Timed Silent) | 32.93% | 33.80% | +0.87pp |
| SR (Speed Reading) | 27.20% | 27.28% | +0.08pp |

### Training summary

| Stage | Config | Best val loss | Epochs |
|-------|--------|--------------|--------|
| Stage 0 MoCo | queue=128, hard negatives | 3.6014 (InfoNCE) | 20 |
| Stage 1 | GPT-2 frozen, enc lr=5e-5, batch=4, accum=2 | 4.2009 | 20 |
| Stage 2 | LoRA rank=4 block [11], enc lr=1e-6 | 4.1744 | 20 |
| QML | QFP 4-qubit, QML_LR=3e-4, rest=1e-6, batch=4 | 4.1733 | 10 |

### SR adapter ablation (DIAG-3)

| Condition | With adapter | Without | Δ |
|-----------|-------------|---------|---|
| NR | 32.48% | 30.09% | +2.40% |
| TSR | 31.30% | 35.82% | **−4.52%** |
| SR | 28.54% | 25.49% | +3.05% |

The adapter significantly helps SR (+3.05%) and NR (+2.40%) but hurts TSR (−4.52%), suggesting condition-specific overfitting in the TSR MLP.

### Key diagnostic findings

- **DIAG-1:** 4 of 6 regions had pool_attn entropy ratio >0.95 (collapsed to 1/256). V9 HTP reduced this through hierarchical softmax with smaller denominators (32-way local, 8-way segment).
- **DIAG-2:** `left_parieto_occipital` consistently dominates cross-region fusion across all three conditions. This is neurologically meaningful — it corresponds to the Visual Word Form Area (VWFA), the primary cortical region for visual reading.
- **DIAG-6:** TF/FG gap is 56.1% overall (NR: 73.2%, TSR: 46.7%, SR: 45.8%), confirming the EEG prefix is under-conditioning the LM — the model still partly relies on GPT-2's language prior.

---

## 8. NVIDIA NIM Agent Pipeline

`nat_eeg_agents_v9_updated.ipynb` runs a three-agent analysis pipeline powered by NVIDIA NIM (NVIDIA Inference Microservices). The agents use `meta/llama-3.1-70b-instruct` to produce structured research commentary.

### Three agents

**Scientist Agent** — given all metrics (V5/V8/V9/QML), training curves, attention diagnostics, and qualitative samples, produces a structured 7-section research analysis:
1. Dataset & Setup
2. Four-model progression — was each addition justified?
3. TF performance analysis
4. FG performance and TF/FG ratio
5. Per-condition analysis (NR/TSR/SR)
6. Attention diagnosis (HTP fix, VWFA dominance, SR patterns)
7. Future directions

**Critic Agent** — reads the Scientist's output and challenges: statistical significance, dataset size limitations, exposure bias, generalisation risks, quantum advantage claims.

**Explainer Agent** — rewrites the key findings in plain language accessible to a non-expert audience, with analogies and no jargon.

### Running the agent notebook

```bash
jupyter notebook nat_eeg_agents_v9_updated.ipynb
```

Run cells in order:

```
Cell 00  — install deps (nemo, openai)
Cell 01  — imports
Cell 02  — load model1_v9 from path
Cell 03  — load V9 classical model
Cell 04  — load QML hybrid model (QuantumFusionProjector)
Cell 05  — load val dataset from lean pickles
Cell 06  — build MultimodalEEGDataset
Cell 07  — attention weight analysis
Cell 08  — inference (TF + FG for both models, 254 batches)
Cell 09  — compute BLEU/ROUGE per model per condition
Cell 10  — corpus metrics aggregation
Cell 11  — qualitative samples (one per condition)
Cell 12  — assemble agent_stats dict
Cell 13  — define agent system prompts
Cell 14  — async call_nim() + run_pipeline()
Cell 15  — display agent outputs as Markdown
Cell 16  — save nat_v9_qml_results.json
```

**With API key set:** full LLM responses from Llama-3.1-70B.

**Without API key:** falls back to simulation mode — prints `[simulation — set NVIDIA_API_KEY]` and uses the agent label as a placeholder. All metrics and statistics still compute correctly.

### Output JSON

```json
{
  "stats": {
    "experiment": { "model_v9_classical": "...", "model_v9_qml": "..." },
    "live_metrics": {
      "v9_tf_bleu1_pct": 30.55,
      "qml_tf_bleu1_pct": 30.86,
      "delta_qml_vs_v9_bleu1": 0.31,
      ...
    },
    "baselines": { "v5": {...}, "v8": {...} },
    "attention_analysis": { "v9_classical": {...}, "v9_qml_hybrid": {...} }
  },
  "scientist": "## 1. DATASET & SETUP ...",
  "critic":    "## Critical Review ...",
  "explainer": "## Plain Language Summary ..."
}
```

---

## 9. Getting an NVIDIA API Key

The agent pipeline calls NVIDIA NIM via the OpenAI-compatible API. You need a free NVIDIA developer account.

### Step-by-step

1. **Create an NVIDIA developer account**
   Go to [https://developer.nvidia.com](https://developer.nvidia.com) and sign up with your email address. It is free.

2. **Access NVIDIA NIM**
   Go to [https://build.nvidia.com](https://build.nvidia.com). Sign in with your developer account.

3. **Choose a model**
   Search for `llama-3.1-70b-instruct` or browse the catalog. The agent notebook uses `meta/llama-3.1-70b-instruct`. Click on the model card.

4. **Generate an API key**
   On the model page, click **Get API Key**. This generates a key starting with `nvapi-`. Copy it immediately — it is only shown once.

5. **Set it in the notebook**
   In `nat_eeg_agents_v9_updated.ipynb`, find and replace:
   ```python
   NVIDIA_API_KEY = "nvapi-PASTE_YOUR_KEY_HERE"
   ```
   with your actual key:
   ```python
   NVIDIA_API_KEY = "nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

6. **Free tier limits**
   The free tier includes a generous number of tokens per month (typically 1,000 API calls or 40,000 tokens per minute on trial). The agent notebook makes 3 calls per run (Scientist, Critic, Explainer) so a single run uses approximately 5,000–8,000 tokens total.

7. **Alternative: environment variable (recommended)**
   Instead of hardcoding the key, set it as an environment variable:
   ```bash
   export NVIDIA_API_KEY="nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```
   Then in the notebook:
   ```python
   import os
   NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "nvapi-PASTE_YOUR_KEY_HERE")
   ```

### API endpoint used

```
https://integrate.api.nvidia.com/v1
```

The notebook uses the `openai` Python package with a custom `base_url`, so any model available on `build.nvidia.com` can be substituted by changing the `model=` string in `call_nim()`.

---

## 10. Plots Reference

All figures are saved to the `plots/` directory.

| File | Description |
|------|-------------|
| `plot_loss_curves.png` | Three-panel: Stage 0 MoCo InfoNCE, Stage 1 train/val, Stage 2 LoRA train/val |
| `plot_overfitting.png` | Two-panel overfitting diagnosis: Stage 1 gap (controlled at ~0.13 vs old 1.65), Stage 2 stable val < train |
| `plot_per_condition_bleu.png` | Grouped bar: V8 vs V9+HTP vs V9+QML per reading condition (NR/TSR/SR) |
| `plot_metrics_comparison.png` | All five metrics (BLEU-1/4, ROUGE-1/L, BERTScore) across V8→V9→QML |
| `plot_val_timeline.png` | Unified val loss timeline: Stage 1 (coral) + Stage 2 (amber) + QML (purple) with stage dividers |
| `plot_stage0_convergence.png` | MoCo loss over 20 epochs with rapid-descent and plateau annotations |
| `plot_stage2_improvement.png` | Two-panel: Stage 2 LoRA improvement over Stage 1 baseline + QML fine-tuning over Stage 2 baseline |
| `diag1_pool_attn_collapse.png` | Six-panel attention weight distribution — shows 4 collapsed regions (H/Hmax>0.95) and 1 selective (left_parieto_occipital, 0.663) |
| `diag2_v9_fusion_weights.png` | Heatmap: cross-region fusion attention weights per condition — left_parieto_occipital (VWFA) dominates |
| `attn_htp_NR.png` | HTP local attention weight profiles for Normal Reading condition |
| `attn_htp_TSR.png` | HTP local attention weight profiles for Timed Silent Reading |
| `attn_htp_SR.png` | HTP local attention weight profiles for Speed Reading |
| `system_architecture.png` | Full model architecture diagram |
| `eeg_encoder.png` | EEGEncoder regional structure |
| `qml_block.png` | QuantumFusionProjector circuit diagram |
| `train_pipeline.png` | Three-stage training pipeline flow |
| `preprocess_pipeline.png` | EEG preprocessing steps |
| `prefix_token.png` | 9-token prefix construction |
| `processed_eeg.png` | Example processed EEG signal |
| `trial1.png` / `trial2.png` | Sample trial visualisations |

---

## 11. Key Findings

### What worked

1. **HTP fixed temporal pooling collapse.** The flat `pool_attn Linear(D,1)` in V8 had 4 of 6 regions with entropy ratio >0.95 — indistinguishable from mean-pooling. HTP's two-level softmax (32-way local + 8-way segment) provides 8× more concentrated gradient signal.

2. **Left parieto-occipital dominance is neurologically valid.** The cross-region fusion MHA consistently assigns the highest weight to left parieto-occipital across all three reading conditions. This corresponds anatomically to the Visual Word Form Area (VWFA, fusiform gyrus), the cortical region responsible for recognising written words. This is a publishable neuroscience finding.

3. **Freezing GPT-2 in Stage 1 eliminated overfitting.** Old Stage 1 (GPT-2 blocks [10,11] unlocked): train/val gap = 1.65 at early stop epoch 7. Fixed Stage 1 (GPT-2 fully frozen): gap = ~0.13 at epoch 17. The Stage 1 task is to train the EEG encoder to produce meaningful prefixes — GPT-2 adaptation should happen in Stage 2 via LoRA.

4. **QML gives consistent marginal gains.** V9+QML improves BLEU-1 by +0.31pp over V9 classical and ROUGE-1 by +0.25pp with only 8,476 quantum parameters (0.006% of total). The gain is small but consistent across conditions and metrics, suggesting the 4-qubit entanglement captures a real inter-region correlation signal.

5. **SR adapter is condition-specific.** +3.05% for SR, +2.40% for NR, but −4.52% for TSR. This suggests the TSR MLP learned features that conflict with TSR's reading pattern, which is intermediate between focused NR and rapid SR.

### What remains open

1. **TF/FG gap (56.1% overall)** — the model still relies heavily on teacher-forced tokens. The EEG 9-token prefix does not exert enough pull to reliably condition free generation. Extending prefix length or adding cross-attention between prefix and GPT-2 KV cache is the highest-priority next step.

2. **Cross-subject generalisation** — the current sentence-aware split shares sentences across subjects. A leave-one-subject-out evaluation is needed to measure true subject-independent decoding.

3. **TSR adapter overfitting** — the SR adapter hurts TSR by 4.52pp. A mixture-of-experts router with harder condition boundaries may be more appropriate than fixed per-condition MLPs.

---

## 12. Citation

If you use this codebase or results, please cite:

```bibtex
@misc{eeg2text2025,
  title   = {EEG2Text: Brain-to-Language Decoding with Hierarchical
             Temporal Pooling and Quantum-Classical Hybrid Architecture},
  year    = {2025},
  note    = {EEG2TextTransformerV9 + QuantumFusionProjector on ZuCo corpus.
             TF BLEU-1 30.86\%, BERTScore F1 85.51\%.}
}
```

And the ZuCo dataset:

```bibtex
@article{hollenstein2018zuco,
  title   = {ZuCo, a simultaneous EEG and eye-tracking resource for
             natural sentence reading},
  author  = {Hollenstein, Nora and Rotsztejn, Jonathan and Troendle, Marius
             and Pedroni, Andreas and Zhang, Ce and Langer, Nicolas},
  journal = {Scientific Data},
  volume  = {5},
  pages   = {180259},
  year    = {2018}
}
```

---

*Built with PyTorch 2.8, PennyLane 0.44.1, HuggingFace Transformers, and NVIDIA NIM on RTX 3050.*
