"""Run the full training pipeline and save metrics."""

from pathlib import Path

from adapters.data.csv_repository import KaggleAppointmentCSVRepository
from adapters.ml.calibrated_predictor import CalibratedPredictor
from adapters.ml.logistic_predictor import LogisticPredictor
from adapters.ml.xgboost_predictor import XGBoostPredictor
from application.use_cases import train_and_evaluate

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "raw" / "KaggleV2-May-2016.csv"
OUTPUT_PATH = ROOT / "reports" / "model_metrics.json"


def main() -> None:
    print(f"Loading data from {CSV_PATH}")
    repo = KaggleAppointmentCSVRepository(CSV_PATH)

    predictor_factories = {
        "logistic": LogisticPredictor,
        "xgboost": XGBoostPredictor,
        "calibrated_xgboost": CalibratedPredictor,
    }

    print("Training and evaluating models...")
    print("  - 5-fold GroupKFold CV")
    print("  - Temporal holdout (June 2016)")
    results = train_and_evaluate(
        repository=repo,
        predictor_factories=predictor_factories,
        output_path=OUTPUT_PATH,
        k=20,
    )

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("\n=== GroupKFold CV ===")
    for model, metrics in results["validation"]["groupkfold"].items():
        print(f"\n{model}:")
        for key, val in metrics.items():
            print(f"  {key}: {val}")

    if results["validation"]["temporal_holdout"]:
        print("\n=== Temporal Holdout (June) ===")
        for model, metrics in results["validation"]["temporal_holdout"].items():
            if isinstance(metrics, dict):
                print(f"\n{model}:")
                for key, val in metrics.items():
                    print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
