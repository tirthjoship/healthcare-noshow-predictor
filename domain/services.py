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


def estimate_business_impact(
    precision_at_k: float,
    daily_call_capacity: int,
    cost_per_noshow: float,
    working_days_per_month: int = 22,
) -> dict[str, float]:
    """Project monthly/annual revenue recoverable from a ranked call list (ADR-004).

    The clinic can make ``daily_call_capacity`` reminder calls per day and works
    the top-K highest-risk patients. ``precision_at_k`` of those flagged patients
    truly no-show, so each is an opportunity to confirm, reschedule, or backfill
    the slot. Each recovered slot is valued at ``cost_per_noshow``.

    This is a labeled *what-if* projection, not a measured outcome:
    - ``cost_per_noshow`` is a primary-care literature estimate, not clinic data.
    - It assumes every correctly-flagged no-show that is called converts to a
      recovered slot — an upper bound. A real pilot would multiply by a
      call-to-recovery conversion rate.

    Args:
        precision_at_k: Fraction of flagged (top-K) patients who truly no-show,
            in [0, 1] (e.g. calibrated model's precision@K on holdout).
        daily_call_capacity: Number of reminder calls the team makes per day (K).
        cost_per_noshow: Dollar value of one recovered appointment slot.
        working_days_per_month: Clinic working days per month.

    Returns:
        Dict of the labeled inputs plus recoverable slots and projected value.

    Raises:
        ValueError: If precision_at_k is outside [0, 1] or any count is negative.
    """
    if not 0.0 <= precision_at_k <= 1.0:
        raise ValueError(f"precision_at_k must be in [0, 1], got {precision_at_k}")
    if daily_call_capacity < 0:
        raise ValueError(
            f"daily_call_capacity must be non-negative, got {daily_call_capacity}"
        )
    if cost_per_noshow < 0:
        raise ValueError(f"cost_per_noshow must be non-negative, got {cost_per_noshow}")
    if working_days_per_month < 0:
        raise ValueError(
            f"working_days_per_month must be non-negative, got {working_days_per_month}"
        )

    recoverable_slots_per_day = daily_call_capacity * precision_at_k
    recoverable_slots_per_month = recoverable_slots_per_day * working_days_per_month
    monthly_value = recoverable_slots_per_month * cost_per_noshow
    annual_value = monthly_value * 12

    return {
        "precision_at_k": round(precision_at_k, 4),
        "daily_call_capacity": float(daily_call_capacity),
        "cost_per_noshow": float(cost_per_noshow),
        "working_days_per_month": float(working_days_per_month),
        "recoverable_slots_per_day": round(recoverable_slots_per_day, 4),
        "recoverable_slots_per_month": round(recoverable_slots_per_month, 2),
        "monthly_value": round(monthly_value, 2),
        "annual_value": round(annual_value, 2),
    }
