"""Use cases: orchestration of domain and adapters."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from adapters.ml.evaluation import evaluate_model
from adapters.ml.feature_encoder import FeatureEncoder
from domain.models import Appointment, NoShowOutcome
from domain.ports import AppointmentRepository, NoShowPredictorPort


def predict_no_show(
    appointment: Appointment,
    predictor: NoShowPredictorPort,
) -> NoShowOutcome:
    """Orchestrate no-show prediction for one appointment."""
    return predictor.predict_no_show(appointment)


def train_and_evaluate(
    repository: AppointmentRepository,
    predictor_factories: dict[str, type],
    output_path: Path,
    k: int = 20,
) -> dict:
    """Train all models with GroupKFold CV + temporal holdout.

    Args:
        repository: Data source implementing AppointmentRepository.
        predictor_factories: Map of name → predictor class.
        output_path: Path to save model_metrics.json.
        k: Top-K for precision@K.

    Returns:
        Nested dict of metrics per model per validation strategy.
    """
    # Load all data
    all_appointments = repository.get_appointments()
    all_labels = repository.get_labels()
    n_rows = len(all_appointments)
    n_patients = len({a.patient.patient_id for a in all_appointments})

    results = {
        "dataset": "KaggleV2-May-2016.csv",
        "n_rows": n_rows,
        "n_patients": n_patients,
        "validation": {
            "groupkfold": {},
            "temporal_holdout": {},
        },
    }

    # === GroupKFold CV ===
    groups = np.array([a.patient.patient_id for a in all_appointments])
    y_all = np.array([int(lb) for lb in all_labels])
    gkf = GroupKFold(n_splits=5)

    for model_name, predictor_cls in predictor_factories.items():
        fold_metrics = []

        for train_idx, test_idx in gkf.split(np.zeros(n_rows), y_all, groups):
            train_appts = [all_appointments[i] for i in train_idx]
            train_labels = [all_labels[i] for i in train_idx]
            test_appts = [all_appointments[i] for i in test_idx]

            # Train predictor
            predictor = predictor_cls()
            predictor.train(train_appts, train_labels)

            # Predict on test fold
            y_prob = np.array(
                [predictor.predict_no_show(appt).risk_score for appt in test_appts]
            )
            y_true = y_all[test_idx]

            metrics = evaluate_model(y_true, y_prob, model_name, k=k)
            fold_metrics.append(metrics)

        # Aggregate across folds
        metric_keys = ["auc", "f1", "brier", "precision_at_k", "ece"]
        aggregated = {}
        for key in metric_keys:
            values = [fm[key] for fm in fold_metrics]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            aggregated[key] = f"{mean_val:.4f} +/- {std_val:.4f}"

        results["validation"]["groupkfold"][model_name] = aggregated

    # === Temporal Holdout ===
    june_start = datetime(2016, 6, 1)
    train_appts = repository.get_appointments(end_date=june_start)
    train_labels = repository.get_labels(end_date=june_start)
    test_appts = repository.get_appointments(start_date=june_start)
    test_labels = repository.get_labels(start_date=june_start)

    if test_appts:
        y_test = np.array([int(lb) for lb in test_labels])

        for model_name, predictor_cls in predictor_factories.items():
            predictor = predictor_cls()
            predictor.train(train_appts, train_labels)

            y_prob = np.array(
                [predictor.predict_no_show(appt).risk_score for appt in test_appts]
            )
            metrics = evaluate_model(y_test, y_prob, model_name, k=k)
            metrics.pop("model_name", None)
            results["validation"]["temporal_holdout"][model_name] = metrics

        results["validation"]["temporal_holdout"]["train_period"] = "2016-04 to 2016-05"
        results["validation"]["temporal_holdout"]["test_period"] = "2016-06"

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))

    return results
