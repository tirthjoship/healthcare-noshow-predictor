"""Use cases: orchestration of domain and adapters."""

from datetime import datetime

from domain.models import Appointment, NoShowOutcome
from domain.ports import AppointmentRepository, NoShowPredictorPort


def train_model(
    data_repository: AppointmentRepository,
    predictor: NoShowPredictorPort,
    start_date: datetime,
    end_date: datetime,
) -> dict[str, float]:
    """Orchestrate model training with leakage-safe data.

    Args:
        data_repository: Source implementing AppointmentRepository.
        predictor: Model implementing NoShowPredictorPort.
        start_date: Start of training period.
        end_date: End of training period.

    Returns:
        Performance metrics (e.g. AUC, F1).
    """
    appointments = data_repository.get_appointments(start_date, end_date)
    labels = data_repository.get_labels(start_date, end_date)
    predictor.train(appointments, labels)
    return {}


def predict_no_show(
    appointment: Appointment,
    predictor: NoShowPredictorPort,
) -> NoShowOutcome:
    """Orchestrate no-show prediction for one appointment.

    Args:
        appointment: Appointment with pre-appointment features.
        predictor: Trained model implementing NoShowPredictorPort.

    Returns:
        NoShowOutcome from the predictor.
    """
    return predictor.predict_no_show(appointment)
