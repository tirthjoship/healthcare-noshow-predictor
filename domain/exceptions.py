"""Domain exceptions for appointment no-show prediction.

All domain-level errors inherit from DomainError.
"""


class DomainError(Exception):
    """Base exception for all domain-level errors."""

    pass


class InvalidAppointmentDataError(DomainError):
    """Raised when appointment data violates business invariants.

    Examples: negative age, negative lead time, invalid gender code,
    appointment day before scheduled day.
    """

    pass


class InvalidNoShowPredictionError(DomainError):
    """Raised when no-show prediction data is invalid.

    Examples: risk score outside [0, 1], invalid risk category.
    """

    pass


class DataLeakageError(DomainError):
    """Raised when post-appointment features are detected in training or prediction.

    All features must be knowable at scheduling time. Any feature derived
    from appointment outcome (e.g. actual attendance) is leakage.
    """

    pass
