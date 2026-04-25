# ── eeg_submission_schema.py ──────────────────────────────────────────────────
# Standard interface for any EEG-to-text researcher to submit their model's
# results and get them compared against the V9+QML baseline.
#
# A researcher only needs to fill in EEGModelSubmission — nothing else.
# The system handles computing deltas, running agents, and producing the report.
#
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ─────────────────────────────────────────────────────────────────────────────
# FROZEN BASELINES  (locked — do not change)
# These are the hardcoded numbers from final.ipynb / nat_eeg_agents_v9_product
# Any researcher's model is compared against these.
# ─────────────────────────────────────────────────────────────────────────────

V5_BASELINE = {
    "model":         "EEG2TextTransformerV5",
    "architecture":  "Conv1D + Bi-GRU + single mean-pool EEG vector, prefix-tuned DistilGPT2",
    "tf_bleu1_pct":  29.24,
    "tf_bleu4_pct":  None,
    "tf_rouge1_pct": 33.92,
    "tf_rougeL_pct": 30.06,
    "bertscore_f1":  None,
    "per_condition": {"NR": 30.70, "TSR": 32.78, "SR": 26.49},
    "note": "Single mean-pool EEG — no region decomp, no MoCo, no LoRA"
}

V8_BASELINE = {
    "model":         "EEG2TextTransformerV8",
    "architecture":  "6-region GRU-Transformer, MoCo Stage0, LoRA rank=8 GPT2 [10,11], SR adapter",
    "tf_bleu1_pct":  30.40,
    "tf_bleu4_pct":  4.30,
    "tf_rouge1_pct": 35.78,
    "tf_rougeL_pct": 30.68,
    "fg_bleu1_pct":  15.41,
    "bertscore_f1":  85.46,
    "per_condition": {"NR": 30.90, "TSR": 32.93, "SR": 27.20},
    "tf_fg_ratio":   1.97,
    "note": "pool_attn collapsed → 1/256; true cross-region in self.fusion MHA"
}

# V9+QML is the primary comparison target for external researchers
V9_QML_BASELINE = {
    "model":         "EEG2TextTransformerV9 + QuantumFusionProjector",
    "architecture":  (
        "V8 + HierarchicalTemporalPooling (local_attn + seg_attn), "
        "LoRA rank=4 alpha=16 block=[11], "
        "QFP: 4-qubit VQC residual post-sr_adapter, 10-epoch QML fine-tune"
    ),
    # These values are filled in at runtime from nat_v9_qml_results.json
    # or from the live agent_stats produced by the product notebook.
    # The schema validator checks these are present before running comparison.
    "tf_bleu1_pct":  None,   # populated from results JSON
    "tf_bleu4_pct":  None,
    "tf_rouge1_pct": None,
    "tf_rougeL_pct": None,
    "fg_bleu1_pct":  None,
    "bertscore_f1":  None,
    "tf_fg_ratio":   None,
    "per_condition": {"NR": None, "TSR": None, "SR": None},
    "note": "Loaded from nat_v9_qml_results.json produced by the product notebook"
}


def load_v9_qml_baseline(results_json_path: str = "nat_v9_qml_results.json") -> dict:
    """
    Load the live V9+QML numbers from the product notebook's output JSON.
    Call this before creating a comparison to ensure fresh baseline values.
    """
    try:
        with open(results_json_path) as f:
            data = json.load(f)
        lm = data["stats"]["live_metrics"]
        baseline = dict(V9_QML_BASELINE)
        baseline["tf_bleu1_pct"]  = lm["qml_tf_bleu1_pct"]
        baseline["tf_bleu4_pct"]  = lm["qml_tf_bleu4_pct"]
        baseline["tf_rouge1_pct"] = lm["qml_tf_rouge1_pct"]
        baseline["tf_rougeL_pct"] = lm["qml_tf_rougeL_pct"]
        baseline["fg_bleu1_pct"]  = lm["qml_fg_bleu1_pct"]
        baseline["tf_fg_ratio"]   = lm["qml_tf_fg_ratio"]
        baseline["per_condition"] = lm["qml_per_cond_bleu1"]
        print(f"✅ V9+QML baseline loaded: BLEU-1={baseline['tf_bleu1_pct']}%")
        return baseline
    except FileNotFoundError:
        print(f"⚠  {results_json_path} not found.")
        print("   Run nat_eeg_agents_v9_product.ipynb first to generate it,")
        print("   or manually set V9_QML_BASELINE values.")
        return V9_QML_BASELINE
    except KeyError as e:
        print(f"⚠  Unexpected JSON structure in {results_json_path}: {e}")
        return V9_QML_BASELINE


# ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION SCHEMA
# This is the ONLY thing an external researcher needs to fill in.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EEGModelSubmission:
    """
    Fill in your model's details and metrics.
    Then pass this to run_comparison_pipeline() to get a full analysis
    comparing your model against V9+QML, V8, and V5.

    Minimum required: model_name, architecture_description, tf_bleu1_pct, tf_rouge1_pct
    All per_condition values are optional but strongly recommended.

    Example usage:
    --------------
    from eeg_submission_schema import EEGModelSubmission, run_comparison_pipeline

    my_model = EEGModelSubmission(
        model_name        = "MyEEGTransformerV2",
        architecture_desc = "6-region encoder + cross-attention + LoRA rank=16 GPT-2",
        tf_bleu1_pct      = 32.1,
        tf_bleu4_pct      = 4.8,
        tf_rouge1_pct     = 37.2,
        tf_rougeL_pct     = 31.5,
        fg_bleu1_pct      = 16.4,
        bertscore_f1      = 85.9,
        tf_fg_ratio       = 1.96,
        per_condition_bleu1 = {"NR": 31.8, "TSR": 33.5, "SR": 28.2},
        dataset           = "ZuCo",
        val_split         = "sentence-aware TEST_SIZE=0.15 seed=42",
        n_val_samples     = 2032,
        dominant_region   = "Left_Temporal",
        notes             = "Increased LoRA rank; no QML component",
    )

    results = await run_comparison_pipeline(my_model)
    """

    # ── Required ──────────────────────────────────────────────────────────────
    model_name:           str   = "MyEEGModel"
    architecture_desc:    str   = "Describe your architecture here"

    # Core metrics (teacher-forcing)
    tf_bleu1_pct:         float = 0.0    # TF BLEU-1 %
    tf_rouge1_pct:        float = 0.0    # TF ROUGE-1 %

    # ── Recommended ───────────────────────────────────────────────────────────
    tf_bleu4_pct:         Optional[float] = None   # TF BLEU-4 %
    tf_rougeL_pct:        Optional[float] = None   # TF ROUGE-L %
    fg_bleu1_pct:         Optional[float] = None   # Free-generation BLEU-1 %
    bertscore_f1:         Optional[float] = None   # BERTScore F1 %
    tf_fg_ratio:          Optional[float] = None   # TF/FG ratio

    # Per-condition breakdown (ZuCo: NR=Normal Reading, TSR=Timed Silent, SR=Speed Reading)
    per_condition_bleu1:  Optional[dict]  = None   # {"NR": x, "TSR": y, "SR": z}

    # ── Context ───────────────────────────────────────────────────────────────
    dataset:              str   = "ZuCo"
    val_split:            str   = "sentence-aware TEST_SIZE=0.15 seed=42"
    n_val_samples:        Optional[int]   = None
    dominant_region:      Optional[str]   = None   # e.g. "Left_Temporal"
    eeg_channels:         Optional[int]   = None   # e.g. 24
    notes:                str   = ""

    # ── Attention (optional — include if you have it) ─────────────────────────
    attention_values:     Optional[dict]  = None   # region_name -> normalised weight
    attention_per_cond:   Optional[dict]  = None   # {NR: {region: w}, TSR: ..., SR: ...}

    def validate(self) -> list[str]:
        """Returns list of warnings. Empty = submission is valid."""
        warnings = []
        if self.tf_bleu1_pct <= 0:
            warnings.append("tf_bleu1_pct is 0 — did you forget to fill in your BLEU-1 score?")
        if self.tf_rouge1_pct <= 0:
            warnings.append("tf_rouge1_pct is 0 — did you forget to fill in your ROUGE-1 score?")
        if self.tf_bleu1_pct > 55:
            warnings.append(f"tf_bleu1_pct={self.tf_bleu1_pct}% is unusually high for ZuCo (typical range 20-50%)")
        if self.bertscore_f1 and (self.bertscore_f1 < 70 or self.bertscore_f1 > 98):
            warnings.append(f"bertscore_f1={self.bertscore_f1}% is outside plausible range 70-98%")
        if self.architecture_desc == "Describe your architecture here":
            warnings.append("architecture_desc is still the placeholder — please fill in your actual architecture")
        return warnings

    def to_agent_stats_slice(self, v9qml_baseline: dict) -> dict:
        """
        Convert submission into the agent_stats format expected by
        the comparison pipeline agents.
        """
        per_cond = self.per_condition_bleu1 or {}
        v9_per   = v9qml_baseline.get("per_condition", {})

        # Compute deltas vs V9+QML
        def delta(a, b):
            if a is None or b is None: return None
            return round(a - b, 2)

        return {
            "submitted_model": {
                "name":              self.model_name,
                "architecture":      self.architecture_desc,
                "dataset":           self.dataset,
                "val_split":         self.val_split,
                "n_val_samples":     self.n_val_samples,
                "dominant_region":   self.dominant_region,
                "eeg_channels":      self.eeg_channels,
                "notes":             self.notes,
            },
            "submitted_metrics": {
                "tf_bleu1_pct":   self.tf_bleu1_pct,
                "tf_bleu4_pct":   self.tf_bleu4_pct,
                "tf_rouge1_pct":  self.tf_rouge1_pct,
                "tf_rougeL_pct":  self.tf_rougeL_pct,
                "fg_bleu1_pct":   self.fg_bleu1_pct,
                "bertscore_f1":   self.bertscore_f1,
                "tf_fg_ratio":    self.tf_fg_ratio,
                "per_condition":  per_cond,
                "attention":      self.attention_values,
                "attention_per_cond": self.attention_per_cond,
            },
            "deltas_vs_v9qml": {
                "bleu1":   delta(self.tf_bleu1_pct,  v9qml_baseline.get("tf_bleu1_pct")),
                "rouge1":  delta(self.tf_rouge1_pct, v9qml_baseline.get("tf_rouge1_pct")),
                "rougeL":  delta(self.tf_rougeL_pct, v9qml_baseline.get("tf_rougeL_pct")),
                "bleu4":   delta(self.tf_bleu4_pct,  v9qml_baseline.get("tf_bleu4_pct")),
                "fg_bleu1": delta(self.fg_bleu1_pct, v9qml_baseline.get("fg_bleu1_pct")),
                "per_condition": {
                    c: delta(per_cond.get(c), v9_per.get(c))
                    for c in ["NR", "TSR", "SR"]
                },
            },
            "deltas_vs_v8": {
                "bleu1":  delta(self.tf_bleu1_pct,  V8_BASELINE["tf_bleu1_pct"]),
                "rouge1": delta(self.tf_rouge1_pct, V8_BASELINE["tf_rouge1_pct"]),
                "bleu4":  delta(self.tf_bleu4_pct,  V8_BASELINE["tf_bleu4_pct"]),
            },
            "deltas_vs_v5": {
                "bleu1":  delta(self.tf_bleu1_pct,  V5_BASELINE["tf_bleu1_pct"]),
                "rouge1": delta(self.tf_rouge1_pct, V5_BASELINE["tf_rouge1_pct"]),
            },
            "baselines": {
                "v5":     V5_BASELINE,
                "v8":     V8_BASELINE,
                "v9_qml": v9qml_baseline,
            },
        }

    def print_summary(self, v9qml_baseline: dict):
        """Print a quick comparison table before running agents."""
        v9b1 = v9qml_baseline.get("tf_bleu1_pct") or "?"
        v9r1 = v9qml_baseline.get("tf_rouge1_pct") or "?"
        d_b1 = round(self.tf_bleu1_pct - (v9qml_baseline.get("tf_bleu1_pct") or 0), 2)
        d_r1 = round(self.tf_rouge1_pct - (v9qml_baseline.get("tf_rouge1_pct") or 0), 2)
        print("\n" + "=" * 60)
        print(f"  Submission: {self.model_name}")
        print("=" * 60)
        print(f"  {'Model':<20} {'BLEU-1':>8}  {'ROUGE-1':>8}")
        print(f"  {'─'*20}  {'─'*8}  {'─'*8}")
        print(f"  {'V5 baseline':<20} {V5_BASELINE['tf_bleu1_pct']:>8.2f}%  {V5_BASELINE['tf_rouge1_pct']:>8.2f}%")
        print(f"  {'V8 baseline':<20} {V8_BASELINE['tf_bleu1_pct']:>8.2f}%  {V8_BASELINE['tf_rouge1_pct']:>8.2f}%")
        print(f"  {'V9+QML (ours)':<20} {str(v9b1)+('%' if v9b1!='?' else ''):>8}  {str(v9r1)+('%' if v9r1!='?' else ''):>8}")
        print(f"  {'YOUR MODEL':<20} {self.tf_bleu1_pct:>8.2f}%  {self.tf_rouge1_pct:>8.2f}%")
        print(f"  {'  Δ vs V9+QML':<20} {d_b1:>+8.2f}pp {d_r1:>+8.2f}pp")
        print("=" * 60 + "\n")
