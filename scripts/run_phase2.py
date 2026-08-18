"""Phase 2 analysis runner — SHAP importance, fairness, and business impact.

Produces the frozen artifacts consumed by the README and the portfolio:
    reports/shap/global_importance.json   (+ .png bar chart, beeswarm)
    reports/fairness.json
    reports/business_impact.json

All numbers are computed from the real Kaggle dataset — nothing is hand-entered.
Run:  python scripts/run_phase2.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from adapters.data.csv_repository import KaggleAppointmentCSVRepository  # noqa: E402
from adapters.ml.calibrated_predictor import CalibratedPredictor  # noqa: E402
from adapters.ml.evaluation import evaluate_model  # noqa: E402
from adapters.ml.fairness import fairness_report  # noqa: E402
from adapters.ml.shap_explainer import ShapExplainer  # noqa: E402
from domain.services import estimate_business_impact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "raw" / "KaggleV2-May-2016.csv"
REPORTS = ROOT / "reports"
SHAP_DIR = REPORTS / "shap"

SHAP_SAMPLE_SIZE = 5000
RANDOM_STATE = 42

# Operating assumptions for the business-impact projection (labeled, not clinic data).
DAILY_CALL_CAPACITY = 20  # matches precision@K reporting (K=20)
COST_PER_NOSHOW = 200.0  # primary-care literature estimate
WORKING_DAYS_PER_MONTH = 22
JUNE_2016 = datetime(2016, 6, 1)


def _age_band(age: int) -> str:
    if age <= 17:
        return "0-17"
    if age <= 34:
        return "18-34"
    if age <= 54:
        return "35-54"
    return "55+"


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Scan 0.10 → 0.90 and return the threshold maximizing F1 (matches evaluation)."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.1, 0.91, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        score = float(f1_score(y_true, y_pred, zero_division=0))
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return round(best_threshold, 2)


def run_shap(appointments: list, labels: list) -> dict[str, float]:
    print("[SHAP] Fitting XGBoost + TreeExplainer on full dataset...")
    explainer = ShapExplainer()
    explainer.fit(appointments, labels)

    rng = np.random.default_rng(RANDOM_STATE)
    n_sample = min(SHAP_SAMPLE_SIZE, len(appointments))
    idx = rng.choice(len(appointments), size=n_sample, replace=False)
    sample = [appointments[i] for i in idx]

    print(f"[SHAP] Computing SHAP values on {n_sample} sampled appointments...")
    importance = explainer.global_importance(sample)
    values = explainer.shap_values(sample)
    matrix = explainer.encode(sample)

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    (SHAP_DIR / "global_importance.json").write_text(
        json.dumps(
            {
                "n_sample": n_sample,
                "random_state": RANDOM_STATE,
                "space": "log-odds (uncalibrated XGBoost; calibration is monotonic)",
                "mean_abs_shap": importance,
                "ranked_features": list(importance.keys()),
            },
            indent=2,
        )
    )

    # Bar chart of mean |SHAP|
    feats = list(importance.keys())[::-1]
    vals = [importance[f] for f in feats]
    plt.figure(figsize=(8, 5))
    plt.barh(feats, vals, color="#2c7fb8")
    plt.xlabel("Mean |SHAP value|  (impact on no-show log-odds)")
    plt.title("Global feature importance — no-shows are an operations problem")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "global_importance.png", dpi=130)
    plt.close()

    # Beeswarm summary
    shap.summary_plot(values, matrix, feature_names=explainer.feature_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close()

    print("[SHAP] Top drivers:", list(importance.items())[:3])
    return importance


def score_holdout(
    repo: KaggleAppointmentCSVRepository,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Train the calibrated model on Apr-May and score the June holdout once."""
    print("[HOLDOUT] Training CalibratedPredictor on Apr-May, scoring June...")
    train_appts = repo.get_appointments(end_date=JUNE_2016)
    train_labels = repo.get_labels(end_date=JUNE_2016)
    test_appts = repo.get_appointments(start_date=JUNE_2016)
    test_labels = repo.get_labels(start_date=JUNE_2016)

    predictor = CalibratedPredictor()
    predictor.train(train_appts, train_labels)
    y_prob = np.array(
        [predictor.predict_no_show(a).risk_score for a in test_appts], dtype=float
    )
    y_true = np.array([int(lb) for lb in test_labels], dtype=float)
    return test_appts, y_true, y_prob


def run_fairness(
    test_appts: list, y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[dict, float]:
    threshold = _best_f1_threshold(y_true, y_prob)

    slices = {
        "gender": np.array([a.patient.gender for a in test_appts]),
        "age_band": np.array([_age_band(a.patient.age) for a in test_appts]),
        "sms_received": np.array(
            ["sms" if a.sms_received else "no_sms" for a in test_appts]
        ),
        "scholarship": np.array(
            ["scholarship" if a.scholarship else "none" for a in test_appts]
        ),
    }

    report = fairness_report(y_true, y_prob, slices, threshold)
    report["holdout_period"] = "2016-06"
    report["threshold_rule"] = "F1-optimal on holdout (scan 0.10-0.90)"
    (REPORTS / "fairness.json").write_text(json.dumps(report, indent=2))

    print(f"[FAIR] threshold={threshold}  disparities={report['disparities']}")
    return report, threshold


def run_business_impact(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    print("[IMPACT] Estimating recoverable revenue from June holdout precision@K...")
    metrics = evaluate_model(
        y_true, y_prob, "calibrated_xgboost", k=DAILY_CALL_CAPACITY
    )
    precision_at_k = float(metrics["precision_at_k"])

    impact = estimate_business_impact(
        precision_at_k=precision_at_k,
        daily_call_capacity=DAILY_CALL_CAPACITY,
        cost_per_noshow=COST_PER_NOSHOW,
        working_days_per_month=WORKING_DAYS_PER_MONTH,
    )
    impact_out: dict[str, object] = dict(impact)
    impact_out["precision_source"] = (
        f"calibrated_xgboost precision@{DAILY_CALL_CAPACITY} on 2016-06 holdout"
    )
    impact_out["assumptions"] = {
        "cost_per_noshow": "USD 200 — primary-care literature estimate (labeled, not clinic data)",
        "conversion": "assumes every correctly-flagged, called no-show recovers a slot (upper bound)",
    }
    (REPORTS / "business_impact.json").write_text(json.dumps(impact_out, indent=2))

    print(
        f"[IMPACT] precision@{DAILY_CALL_CAPACITY}={precision_at_k} -> "
        f"monthly ${impact['monthly_value']:,.0f}, annual ${impact['annual_value']:,.0f}"
    )
    return impact


def main() -> None:
    print(f"Loading {CSV_PATH}")
    repo = KaggleAppointmentCSVRepository(CSV_PATH)
    all_appts = repo.get_appointments()
    all_labels = repo.get_labels()
    print(f"Loaded {len(all_appts)} appointments.")

    run_shap(all_appts, all_labels)
    test_appts, y_true, y_prob = score_holdout(repo)
    run_fairness(test_appts, y_true, y_prob)
    run_business_impact(y_true, y_prob)
    print("\nPhase 2 artifacts written to reports/.")


if __name__ == "__main__":
    main()
