"""KaggleAppointmentCSVRepository — CSV adapter for Kaggle medical appointments dataset.

Implements the AppointmentRepository port. Parses the Kaggle dataset CSV,
constructs domain objects, and enforces the leakage guard via ALLOWED_COLUMNS.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from domain.exceptions import DataLeakageError
from domain.models import Appointment, Patient

# Exact columns present in the Kaggle dataset. Any extra column is potential leakage.
ALLOWED_COLUMNS: frozenset[str] = frozenset(
    {
        "PatientId",
        "AppointmentID",
        "Gender",
        "ScheduledDay",
        "AppointmentDay",
        "Age",
        "Neighbourhood",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
        "No-show",
    }
)


class KaggleAppointmentCSVRepository:
    """Load appointment data from the Kaggle medical appointments CSV.

    The Kaggle dataset uses non-standard column spellings:
    - "Hipertension" (typo for hypertension) → mapped to hypertension field
    - "Handcap" (typo for handicap) → mapped to handicap field
    - PatientId stored as float → converted to str(int())

    Leakage guard: any column outside ALLOWED_COLUMNS raises DataLeakageError
    on construction, before any data is returned.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._df = self._load_and_validate(Path(csv_path))

    def _load_and_validate(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)

        # Leakage guard: reject unexpected columns
        actual_columns = frozenset(df.columns)
        unexpected = actual_columns - ALLOWED_COLUMNS
        if unexpected:
            raise DataLeakageError(
                f"Unexpected columns detected (potential leakage): {unexpected}"
            )

        # Parse datetime columns; convert to UTC-aware
        for col in ("ScheduledDay", "AppointmentDay"):
            df[col] = pd.to_datetime(df[col], utc=True)

        # Compute lead_time_days, clip at 0 (negative = data error)
        df["lead_time_days"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days.clip(
            lower=0
        )

        return df

    def _build_appointment(self, row: pd.Series) -> Appointment:
        patient = Patient(
            patient_id=str(int(row["PatientId"])),
            age=int(row["Age"]),
            gender=str(row["Gender"]),
        )
        return Appointment(
            appointment_id=str(int(row["AppointmentID"])),
            patient=patient,
            scheduled_day=row["ScheduledDay"].to_pydatetime(),
            appointment_day=row["AppointmentDay"].to_pydatetime(),
            neighbourhood=str(row["Neighbourhood"]),
            scholarship=int(row["Scholarship"]),
            hypertension=int(row["Hipertension"]),  # dataset typo → domain field name
            diabetes=int(row["Diabetes"]),
            alcoholism=int(row["Alcoholism"]),
            handicap=int(row["Handcap"]),  # dataset typo → domain field name
            sms_received=int(row["SMS_received"]),
            lead_time_days=int(row["lead_time_days"]),
        )

    def _filter(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pd.DataFrame:
        df = self._df
        if start_date is not None:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            df = df[df["AppointmentDay"] >= start_date]
        if end_date is not None:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            df = df[df["AppointmentDay"] < end_date]
        return df

    def get_appointments(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Appointment]:
        """Return Appointment objects filtered by AppointmentDay range."""
        df = self._filter(start_date, end_date)
        return [self._build_appointment(row) for _, row in df.iterrows()]

    def get_labels(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[bool]:
        """Return no-show labels aligned with get_appointments (True = no-show)."""
        df = self._filter(start_date, end_date)
        return [val == "Yes" for val in df["No-show"]]
