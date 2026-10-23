"""Export the small files behind the Overview tab of the Streamlit app.

Writes two committed files from the (gitignored) raw Kaggle CSV:
    demo/june_scores.csv.gz      one row per June 2016 holdout appointment, scored
                                 by the calibrated model trained on April to May
    demo/weekday_lead_rates.csv  no-show rate and count by weekday x lead bucket,
                                 all 110K appointments

Nothing is hand-entered. Run:
    python scripts/export_dashboard.py [path/to/KaggleV2-May-2016.csv]
or set NOSHOW_CSV_PATH.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from adapters.data.csv_repository import KaggleAppointmentCSVRepository
from adapters.ml.calibrated_predictor import CalibratedPredictor
from adapters.visualization.overview import LEAD_BUCKETS, lead_bucket

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "raw" / "KaggleV2-May-2016.csv"
DEMO_DIR = ROOT / "demo"
JUNE_2016 = datetime(2016, 6, 1)


def _weekday(day: datetime) -> str:
    return day.strftime("%a")


def june_scores(repo: KaggleAppointmentCSVRepository) -> pd.DataFrame:
    """Train on April to May exactly as run_phase2.score_holdout, score June."""
    train_appts = repo.get_appointments(end_date=JUNE_2016)
    train_labels = repo.get_labels(end_date=JUNE_2016)
    test_appts = repo.get_appointments(start_date=JUNE_2016)
    test_labels = repo.get_labels(start_date=JUNE_2016)

    predictor = CalibratedPredictor()
    predictor.train(train_appts, train_labels)
    risk = np.array([predictor.predict_no_show(a).risk_score for a in test_appts])

    frame = pd.DataFrame(
        {
            "appointment_day": [
                a.appointment_day.date().isoformat() for a in test_appts
            ],
            "age": [a.patient.age for a in test_appts],
            "neighbourhood": [a.neighbourhood for a in test_appts],
            "lead_time_days": [a.lead_time_days for a in test_appts],
            "risk_score": np.round(risk, 4),
            "actual_no_show": [int(lb) for lb in test_labels],
        }
    )
    # Rank within each day from the UNROUNDED score: rounding to 4 dp creates ties
    # that would change the top-K. Ties keep the original (repository) order.
    frame["day_rank"] = (
        pd.Series(risk)
        .groupby(frame["appointment_day"])
        .rank(method="first", ascending=False)
        .astype(int)
    )
    # Weekday, age band and lead bucket are derived at load time (overview.enrich).
    return frame.sort_values(["appointment_day", "day_rank"]).reset_index(drop=True)


def weekday_lead_rates(repo: KaggleAppointmentCSVRepository) -> pd.DataFrame:
    """No-show rate and count by appointment weekday x lead bucket, full data."""
    appts = repo.get_appointments()
    labels = repo.get_labels()
    frame = pd.DataFrame(
        {
            "weekday": [_weekday(a.appointment_day) for a in appts],
            "lead_bucket": [lead_bucket(a.lead_time_days) for a in appts],
            "no_show": [int(lb) for lb in labels],
        }
    )
    grouped = (
        frame.groupby(["weekday", "lead_bucket"])["no_show"]
        .agg(appointments="size", no_shows="sum")
        .reset_index()
    )
    grouped["no_show_rate"] = (grouped["no_shows"] / grouped["appointments"]).round(4)
    order = {name: i for i, name in enumerate(LEAD_BUCKETS)}
    grouped["_o"] = grouped["lead_bucket"].map(order)
    result: pd.DataFrame = grouped.sort_values(["weekday", "_o"]).drop(columns="_o")
    return result


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = Path(arg or os.environ.get("NOSHOW_CSV_PATH") or DEFAULT_CSV)
    print(f"Loading {csv_path}")
    repo = KaggleAppointmentCSVRepository(csv_path)

    scores = june_scores(repo)
    rates = weekday_lead_rates(repo)
    DEMO_DIR.mkdir(exist_ok=True)
    scores.to_csv(
        DEMO_DIR / "june_scores.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    rates.to_csv(DEMO_DIR / "weekday_lead_rates.csv", index=False)

    total = int(rates["appointments"].sum())
    misses = int(rates["no_shows"].sum())
    same = rates[rates["lead_bucket"] == LEAD_BUCKETS[0]]
    print(
        f"June rows {len(scores)}  June base rate {scores['actual_no_show'].mean():.4f}"
    )
    print(f"Full rows {total}  full no-show rate {misses / total:.4f}")
    print("Same-day rate " f"{same['no_shows'].sum() / same['appointments'].sum():.4f}")


if __name__ == "__main__":
    main()
