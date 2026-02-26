"""Domain exceptions for patient readmission risk.

All domain-level errors inherit from DomainError.
"""


class DomainError(Exception):
    """Base exception for all domain-level errors."""

    pass


class InvalidAdmissionDataError(DomainError):
    """Raised when encounter/admission data violates business invariants.

    Examples: negative age or counts, invalid admission type or gender code,
    missing required fields.
    """

    pass


class InvalidRiskAssessmentError(DomainError):
    """Raised when risk outcome data is invalid.

    Examples: risk score outside [0, 1], invalid risk category,
    missing required fields.
    """

    pass


class DataLeakageError(DomainError):
    """Raised when post-discharge features are detected in training or prediction.

    Prevents models from using future-looking clinical data that would
    not be available at prediction time.
    """

    pass
