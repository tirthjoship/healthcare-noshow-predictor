"""Use cases: orchestration of domain and adapters."""

from datetime import datetime

from domain.models import Encounter, RiskOutcome
from domain.ports import PatientDataRepository, RiskPredictorPort


def train_model(
    data_repository: PatientDataRepository,
    predictor: RiskPredictorPort,
    start_date: datetime,
    end_date: datetime,
) -> dict[str, float]:
    """Orchestrate model training with leakage-safe data.

    Args:
        data_repository: Source implementing PatientDataRepository.
        predictor: Model implementing RiskPredictorPort.
        start_date: Start of training period.
        end_date: End of training period.

    Returns:
        Performance metrics (e.g. precision, recall, AUC-PR).
    """
    encounters = data_repository.get_encounters(start_date, end_date)
    # Outcomes loaded per repository contract; then predictor.train(...)
    # TODO: load outcomes and call predictor.train(encounters, outcomes)
    return {}


def predict_readmission_risk(
    encounter: Encounter,
    predictor: RiskPredictorPort,
) -> RiskOutcome:
    """Orchestrate risk prediction for one encounter.

    Args:
        encounter: Encounter with pre-discharge features.
        predictor: Trained model implementing RiskPredictorPort.

    Returns:
        RiskOutcome from the predictor.
    """
    return predictor.predict_readmission_risk(encounter)
