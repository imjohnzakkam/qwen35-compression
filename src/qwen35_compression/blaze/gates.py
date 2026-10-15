"""Blaze's gates. G1: is the ladder free, does a 6-bit body pay, and is a 4/8-bit menu enough on
the 4B. G1b: does a GPTQ-rounded 6-bit body reach the target G1's round-to-nearest one missed.
G5: does the 2B, at a standard W4A16 export's size, agree.

Drift results are drift_scores.py's (BF16's 1,501 replayed MATH-500 and MMLU-Pro answers).
Excess loss is a model's mean NLL of BF16's tokens minus BF16's own on the same tokens.
"""

from __future__ import annotations

from typing import Any

# (a) The ladder alone, BF16 body: within a whisker of BF16 (its noise floor flips 0.35%).
LADDER_MAX_FLIP_RATE = 0.006
LADDER_MAX_EXCESS = 0.001
# (b) The ladder with every body Linear at 6-bit round-to-nearest, no calibration. At or under
# this, a 6-bit option at Blaze-Q's ~6.8 bits per weight should reach 0.002 (7x under Glaze v2's
# 0.014), so the kernel pays. GPTQ would do better than round-to-nearest; this is the plain case.
SIX_BIT_MAX_EXCESS = 0.004
# G1b: the same 6-bit body GPTQ-rounded on generic text, held to the first draft's GPTQ mark.
GPTQ_SIX_BIT_MAX_EXCESS = 0.003
# G5: Blaze-Q at most a third of standard W4A16's excess loss, no larger, and no panel task
# more than a point below BF16.
G5_MAX_EXCESS_SHARE = 1 / 3
G5_MAX_PANEL_DROP = 1.0
# (c) Share of the FFN's summed price held by its most expensive half of weights (concentration).
MENU_MIN_CONCENTRATION = 0.75
# The error model: excess loss falls about 4x per added bit (records 3, 4 and 6).
FACTOR_PER_BIT = 4.0

A30_RUPEES_PER_HOUR = 38.88
RUPEES_PER_DOLLAR = 96.3  # the last rate on record, 2 October 2026


def drift_point(result: dict[str, Any]) -> dict[str, Any]:
    scores = result["scores"]["both"]["all"]
    return {
        "answers": scores["answers"],
        "tokens": scores["tokens"],
        "flip_rate": scores["flip_rate"],
        "mean_nll": scores["mean_nll"],
    }


def excess_loss(result: dict[str, Any], bf16: dict[str, Any]) -> float:
    model, base = drift_point(result), drift_point(bf16)
    if model["tokens"] != base["tokens"]:
        raise ValueError(f"scored on different tokens: {model['tokens']} and {base['tokens']}")
    return model["mean_nll"] - base["mean_nll"]


def g1_report(
    bf16: dict[str, Any],
    ladder: dict[str, Any],
    six_bit: dict[str, Any],
    four_bit: dict[str, Any] | None,
    concentration: float | None,
) -> dict[str, Any]:
    """The three checks, the measured bits rule when the 4-bit reference ran, and a decision."""
    ladder_point, six_point = drift_point(ladder), drift_point(six_bit)
    ladder_excess = excess_loss(ladder, bf16)
    six_excess = excess_loss(six_bit, bf16)
    checks = {
        "ladder_free": ladder_point["flip_rate"] <= LADDER_MAX_FLIP_RATE
        and ladder_excess <= LADDER_MAX_EXCESS,
        "six_bit_pays": six_excess <= SIX_BIT_MAX_EXCESS,
        "menu_enough": None if concentration is None else concentration >= MENU_MIN_CONCENTRATION,
    }
    report: dict[str, Any] = {
        "bf16": drift_point(bf16),
        "ladder": {**ladder_point, "excess_loss": ladder_excess},
        "six_bit": {**six_point, "excess_loss": six_excess},
        "ffn_concentration": concentration,
        "thresholds": {
            "ladder_max_flip_rate": LADDER_MAX_FLIP_RATE,
            "ladder_max_excess": LADDER_MAX_EXCESS,
            "six_bit_max_excess": SIX_BIT_MAX_EXCESS,
            "menu_min_concentration": MENU_MIN_CONCENTRATION,
        },
        "checks": checks,
    }
    if four_bit is not None:
        four_excess = excess_loss(four_bit, bf16)
        report["four_bit"] = {**drift_point(four_bit), "excess_loss": four_excess}
        # Two bits apart (4.125 and 6.125 bits per weight): the rule predicts a factor of 16.
        report["bits_rule"] = {
            "measured_factor": four_excess / six_excess if six_excess > 0 else None,
            "predicted_factor": FACTOR_PER_BIT**2,
        }
    report["decision"] = decide(checks)
    return report


def decide(checks: dict[str, Any]) -> str:
    if not checks["ladder_free"]:
        return "stop: retest the ladder once with 131,072 hot rows, then drop it if it still fails"
    six, menu = checks["six_bit_pays"], checks["menu_enough"]
    if six and menu is False:
        return "go: G2 needs the 6-bit option, so start the kernel track"
    if six:
        return "go: G2 on the 4/8-bit menu; a 6-bit kernel would add more"
    if menu is False:
        return "stop on the 4B: neither route reaches the target; check the 2B (G5) before more"
    return "go: G2 on the 4/8-bit menu; skip the 6-bit kernel"


def g1b_report(
    bf16: dict[str, Any], gptq: dict[str, Any], rtn_excess: float | None = None
) -> dict[str, Any]:
    """The 6-bit GPTQ body (with the ladder) against BF16, beside G1's round-to-nearest one."""
    excess = excess_loss(gptq, bf16)
    checks = {"six_bit_gptq_pays": excess <= GPTQ_SIX_BIT_MAX_EXCESS}
    if checks["six_bit_gptq_pays"]:
        decision = "go: the 6-bit option pays with GPTQ; G2 needs the 6-bit kernel"
    elif excess <= SIX_BIT_MAX_EXCESS:
        decision = "borderline: GPTQ meets G1's plain mark but not the GPTQ one; ask before G2"
    else:
        decision = "stop the 6-bit route on the 4B: GPTQ does not reach the target"
    return {
        "bf16": drift_point(bf16),
        "six_bit_gptq": {**drift_point(gptq), "excess_loss": excess},
        "six_bit_rtn_excess": rtn_excess,
        "gptq_gain": rtn_excess / excess if rtn_excess and excess > 0 else None,
        "thresholds": {
            "six_bit_gptq_max_excess": GPTQ_SIX_BIT_MAX_EXCESS,
            "six_bit_max_excess": SIX_BIT_MAX_EXCESS,
        },
        "checks": checks,
        "decision": decision,
    }


def g5_report(
    bf16: dict[str, Any],
    w4a16: dict[str, Any],
    blaze_q: dict[str, Any],
    sizes: dict[str, int],
    panel_drops: dict[str, float] | None,
) -> dict[str, Any]:
    """Blaze-Q against standard W4A16 on the 2B: drift share, size, and (when run) the panel.

    `sizes` holds `w4a16` and `blaze_q` bytes; `panel_drops` is BF16's score minus Blaze-Q's,
    in points, per panel task.
    """
    w4_excess = excess_loss(w4a16, bf16)
    q_excess = excess_loss(blaze_q, bf16)
    checks: dict[str, Any] = {
        "excess_share": q_excess <= G5_MAX_EXCESS_SHARE * w4_excess,
        "no_larger": sizes["blaze_q"] <= sizes["w4a16"],
        "panel": None
        if panel_drops is None
        else all(drop <= G5_MAX_PANEL_DROP for drop in panel_drops.values()),
    }
    if checks["excess_share"] and checks["no_larger"] and checks["panel"] is not False:
        decision = "go: the 2B agrees; Blaze is not a 4B-only result"
    else:
        decision = "stop: Blaze stays a 4B-only result"
    return {
        "bf16": drift_point(bf16),
        "w4a16": {**drift_point(w4a16), "excess_loss": w4_excess},
        "blaze_q": {**drift_point(blaze_q), "excess_loss": q_excess},
        "excess_share": q_excess / w4_excess if w4_excess > 0 else None,
        "sizes": sizes,
        "panel_drops": panel_drops,
        "thresholds": {
            "max_excess_share": G5_MAX_EXCESS_SHARE,
            "max_panel_drop": G5_MAX_PANEL_DROP,
        },
        "checks": checks,
        "decision": decision,
    }


def cost(hours: float) -> dict[str, float]:
    rupees = hours * A30_RUPEES_PER_HOUR
    return {
        "hours": hours,
        "rupees": round(rupees, 2),
        "dollars": round(rupees / RUPEES_PER_DOLLAR, 2),
    }
