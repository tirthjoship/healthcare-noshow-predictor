"""Domain services: pure business logic for no-show risk.

No external dependencies. Uses only pre-appointment features.
"""

from .models import Appointment


def baseline_no_show_risk_flag(appointment: Appointment) -> str:
    """Rule-based baseline risk category using only scheduling-time features.

    Heuristic based on EDA findings:
    - Longer lead time → higher no-show risk
    - Younger patients → higher no-show risk
    - SMS received correlates with higher risk (confound: sent to high-risk)
    - No scholarship/chronic conditions → slightly lower risk

    Args:
        appointment: Appointment with pre-appointment features only.

    Returns:
        One of 'Low Risk', 'Medium Risk', 'High Risk'.
    """
    risk_factors = [
        appointment.lead_time_days >= 14,
        appointment.lead_time_days >= 30,
        appointment.patient.age < 35,
        appointment.sms_received == 1,
        appointment.scholarship == 1,
    ]
    risk_count = sum(risk_factors)
    if risk_count >= 3:
        return "High Risk"
    if risk_count >= 1:
        return "Medium Risk"
    return "Low Risk"
