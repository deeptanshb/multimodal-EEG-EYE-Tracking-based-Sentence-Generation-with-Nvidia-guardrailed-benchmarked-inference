# ── actions.py  —  Custom Python actions for EEG guardrails ──────────────────
# Registered as Colang actions, called from rails.co
# These run server-side (no LLM call) — fast and deterministic

import re
from nemoguardrails.actions import action


@action(name="check_metric_bounds")
async def check_metric_bounds(response: str) -> bool:
    """
    Extract any percentage values cited near metric keywords and verify
    they fall within plausible EEG-to-text ranges.

    Plausible ranges (based on ZuCo literature + our baselines):
      BLEU-1:    20 – 55%
      BLEU-4:     1 – 15%
      ROUGE-1:   25 – 55%
      ROUGE-L:   22 – 50%
      BERTScore: 78 – 96%
    """
    text = response.lower()

    # Require separator (=, :, space, 'of') before value to avoid matching
    # digit suffixes in metric names like "BLEU-1" where '1' is part of the name.
    # Also require value to be decimal OR 2+ digits — rules out lone '1', '4', etc.
    pattern = r'(bleu[-_]?[14]?|rouge[-_]?[1l]?|bertscore)[\s:=\(of]+(\d{1,3}\.\d+|\d{2,3}(?!\.))'
    matches = re.findall(pattern, text)

    RANGES = {
        "bleu-1": (20.0, 55.0),
        "bleu-4": (1.0,  15.0),   # BLEU-4 is naturally lower than BLEU-1
        "bleu_4": (1.0,  15.0),
        "bleu":   (20.0, 55.0),   # fallback for bare "bleu"
        "rouge":  (22.0, 58.0),
        "bertscore": (78.0, 96.5),
    }

    for metric_raw, value_str in matches:
        try:
            value = float(value_str)
        except ValueError:
            continue

        # Normalize metric name — prefer specific (bleu-1, bleu-4) over generic
        if "bert" in metric_raw:
            key = "bertscore"
        elif "rouge" in metric_raw:
            key = "rouge"
        elif "4" in metric_raw:
            key = "bleu-4"   # bleu-4 or bleu_4
        else:
            key = "bleu-1" if "1" in metric_raw else "bleu"

        lo, hi = RANGES[key]
        if value < lo or value > hi:
            print(f"  [guardrail] metric out of range: {metric_raw}={value}%  (expected {lo}-{hi}%)")
            return False

    return True


@action(name="self_check_relevance")
async def self_check_relevance(response: str, context: str = "EEG-to-text neuroscience research on ZuCo dataset") -> bool:
    """
    Lightweight keyword-based domain relevance check.
    A proper production version would call a small NIM classifier.
    Here we check that the response contains at least 3 domain terms.
    """
    domain_terms = [
        "eeg", "zuco", "bleu", "rouge", "bertscore", "attention", "encoder",
        "transformer", "gpt", "lora", "htp", "qml", "quantum", "neuroscience",
        "region", "temporal", "condition", "val", "baseline", "spectral",
        "waveform", "decoding", "moco", "stage", "checkpoint"
    ]

    text = response.lower()
    found = sum(1 for term in domain_terms if term in text)

    if found < 3:
        print(f"  [guardrail] domain relevance low: only {found} domain terms found in response")
        return False

    return True


@action(name="get_agent_role")
async def get_agent_role(system_prompt: str) -> str:
    """Extract agent role from the system prompt for dialog rail routing."""
    sp = system_prompt.lower()
    if "neuroscience and nlp researcher" in sp:
        return "scientist"
    elif "senior reviewer" in sp or "neurips" in sp:
        return "critic"
    elif "science communicator" in sp:
        return "qml_synthesiser"
    return "unknown"
