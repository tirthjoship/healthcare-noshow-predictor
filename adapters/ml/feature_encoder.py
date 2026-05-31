"""FeatureEncoder — converts Appointment domain objects to numpy feature matrix.

Target encoding for neighbourhood is fit on training labels only (ADR-007).
"""

from __future__ import annotations

import numpy as np

from domain.models import Appointment


class FeatureEncoder:
    """Encode Appointment objects into a fixed 10-column numpy array.

    Neighbourhood is target-encoded (mean no-show rate per neighbourhood in
    training data). Unseen neighbourhoods receive the global training mean.
    Must call fit() or fit_transform() before transform().
    """

    FEATURE_NAMES: list[str] = [
        "age",
        "gender_enc",
        "lead_time_days",
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "handicap",
        "sms_received",
        "neighbourhood_enc",
    ]

    def __init__(self) -> None:
        self._neighbourhood_map: dict[str, float] | None = None
        self._global_mean: float | None = None

    def fit(self, appointments: list[Appointment], labels: list[bool]) -> None:
        """Compute per-neighbourhood mean no-show rate from training data.

        Args:
            appointments: Training appointments.
            labels: Corresponding no-show labels (True = no-show).
        """
        neighbourhood_totals: dict[str, list[float]] = {}
        for appt, label in zip(appointments, labels):
            neighbourhood_totals.setdefault(appt.neighbourhood, []).append(float(label))

        self._neighbourhood_map = {
            nb: float(np.mean(vals)) for nb, vals in neighbourhood_totals.items()
        }
        all_labels = [float(lb) for lb in labels]
        self._global_mean = float(np.mean(all_labels)) if all_labels else 0.0

    def transform(self, appointments: list[Appointment]) -> np.ndarray:
        """Convert appointments to a (N, 10) numpy array.

        Args:
            appointments: Appointments to encode.

        Returns:
            Float64 array of shape (len(appointments), 10).

        Raises:
            RuntimeError: If fit() has not been called first.
        """
        if self._neighbourhood_map is None or self._global_mean is None:
            raise RuntimeError(
                "FeatureEncoder must be fitted before calling transform(). "
                "Call fit() or fit_transform() first."
            )
        rows: list[list[float]] = []
        for appt in appointments:
            gender_enc = 1.0 if appt.patient.gender == "F" else 0.0
            neighbourhood_enc = self._neighbourhood_map.get(
                appt.neighbourhood, self._global_mean
            )
            row = [
                float(appt.patient.age),
                gender_enc,
                float(appt.lead_time_days),
                float(appt.scholarship),
                float(appt.hypertension),
                float(appt.diabetes),
                float(appt.alcoholism),
                float(appt.handicap),
                float(appt.sms_received),
                neighbourhood_enc,
            ]
            rows.append(row)
        return np.array(rows, dtype=np.float64)

    def fit_transform(
        self, appointments: list[Appointment], labels: list[bool]
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            appointments: Training appointments.
            labels: Corresponding no-show labels.

        Returns:
            Float64 array of shape (len(appointments), 10).
        """
        self.fit(appointments, labels)
        return self.transform(appointments)

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of feature column names."""
        return self.FEATURE_NAMES
