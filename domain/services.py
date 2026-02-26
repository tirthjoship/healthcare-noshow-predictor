"""Domain services: pure business logic for readmission risk.

No external dependencies. Uses only pre-discharge features.
"""

from .models import Encounter

# High-risk diagnosis prefixes (ICD-10) for rule-based baseline.
HIGH_RISK_CONDITIONS: frozenset[str] = frozenset({
    "I50",  # Heart failure
    "J44",  # COPD
    "E11",  # Type 2 diabetes with complications
    "N18",  # Chronic kidney disease
})


def baseline_readmission_risk_flag(encounter: Encounter) -> str:
    """Rule-based baseline risk category using only pre-discharge features.

    Args:
        encounter: Encounter with pre-discharge features only.

    Returns:
        One of 'Low Risk', 'Medium Risk', 'High Risk'.
    """
    diagnosis_prefix = encounter.primary_diagnosis[:3]
    high_risk_conditions = [
        diagnosis_prefix in HIGH_RISK_CONDITIONS,
        encounter.comorbidity_count >= 3,
        encounter.prior_admissions_count >= 2,
        encounter.encounter_type == "Emergency",
        encounter.length_of_stay_scheduled >= 7,
    ]
    risk_factor_count = sum(high_risk_conditions)
    if risk_factor_count >= 3:
        return "High Risk"
    if risk_factor_count >= 1:
        return "Medium Risk"
    return "Low Risk"
