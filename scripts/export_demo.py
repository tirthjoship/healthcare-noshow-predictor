"""Export a lightweight, precomputed demo dataset for the Streamlit app.

Streamlit Community Cloud has no access to the (gitignored) raw Kaggle CSV, and
training on 71K rows at startup is too heavy for the free tier. So we precompute
here — calibrated risk scores + per-patient SHAP contributions for a sample of
the June holdout — and commit a small CSV the app just renders.

Run:  python scripts/export_demo.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from adapters.data.csv_repository import KaggleAppointmentCSVRepository
from adapters.ml.calibrated_predictor import CalibratedPredictor
from adapters.ml.shap_explainer import ShapExplainer

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "raw" / "KaggleV2-May-2016.csv"
DEMO_DIR = ROOT / "demo"
JUNE_2016 = datetime(2016, 6, 1)
N_DEMO = 1500
RANDOM_STATE = 42


def main() -> None:
    print(f"Loading {CSV_PATH}")
    repo = KaggleAppointmentCSVRepository(CSV_PATH)

    train_appts = repo.get_appointments(end_date=JUNE_2016)
    train_labels = repo.get_labels(end_date=JUNE_2016)
    test_appts = repo.get_appointments(start_date=JUNE_2016)
    test_labels = repo.get_labels(start_date=JUNE_2016)

    print("Training calibrated model + fitting SHAP explainer on Apr-May...")
    predictor = CalibratedPredictor()
    predictor.train(train_appts, train_labels)
    explainer = ShapExplainer()
    explainer.fit(train_appts, train_labels)

    rng = np.random.default_rng(RANDOM_STATE)
    n = min(N_DEMO, len(test_appts))
    idx = sorted(rng.choice(len(test_appts), size=n, replace=False).tolist())
    sample = [test_appts[i] for i in idx]
    sample_labels = [test_labels[i] for i in idx]

    print(f"Scoring {n} holdout appointments + computing SHAP contributions...")
    shap_matrix = explainer.shap_values(sample)
    feature_names = explainer.feature_names

    rows: list[dict[str, object]] = []
    for j, (appt, label) in enumerate(zip(sample, sample_labels)):
        outcome = predictor.predict_no_show(appt)
        row: dict[str, object] = {
            "appointment_id": appt.appointment_id,
            "patient_id": appt.patient.patient_id,
            "appointment_day": appt.appointment_day.date().isoformat(),
            "age": appt.patient.age,
            "gender": appt.patient.gender,
            "neighbourhood": appt.neighbourhood,
            "lead_time_days": appt.lead_time_days,
            "sms_received": appt.sms_received,
            "scholarship": appt.scholarship,
            "hypertension": appt.hypertension,
            "diabetes": appt.diabetes,
            "risk_score": round(outcome.risk_score, 4),
            "risk_category": outcome.risk_category,
            "actual_no_show": int(bool(label)),
        }
        for f, val in zip(feature_names, shap_matrix[j]):
            row[f"shap_{f}"] = round(float(val), 4)
        rows.append(row)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEMO_DIR / "scored_appointments.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
