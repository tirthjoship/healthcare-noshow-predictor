"""Tests for KaggleAppointmentCSVRepository."""

from datetime import datetime, timezone


from adapters.data.csv_repository import KaggleAppointmentCSVRepository
from domain.models import Appointment, Patient


class TestCSVRepositoryLoading:
    """Tests for basic CSV loading and object construction."""

    def test_loads_all_rows(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments()
        assert len(appointments) == 6

    def test_returns_appointment_objects(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments()
        assert all(isinstance(a, Appointment) for a in appointments)

    def test_parses_patient_fields(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments()
        # First row: PatientId=1.0 → "1", Age=62, Gender=F
        patient = appointments[0].patient
        assert isinstance(patient, Patient)
        assert patient.patient_id == "1"
        assert patient.age == 62
        assert patient.gender == "F"

    def test_parses_appointment_fields(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments()
        appt = appointments[0]
        assert appt.appointment_id == "1001"
        assert appt.neighbourhood == "JARDIM DA PENHA"
        assert appt.scholarship == 0
        assert appt.hypertension == 1
        assert appt.diabetes == 0
        assert appt.alcoholism == 0
        assert appt.handicap == 0
        assert appt.sms_received == 0

    def test_computes_lead_time(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments()
        # Row 2: ScheduledDay=2016-04-30T10:00, AppointmentDay=2016-05-05T00:00
        # Timedelta = 4 days 14 hours → .days truncates to 4
        appt = appointments[1]
        assert appt.lead_time_days == 4

    def test_clips_negative_lead_time(self, sample_csv, tmp_path):
        # Row 1: AppointmentDay == ScheduledDay (same day, could be negative after
        # stripping time). Let's create a specific row with appointment before scheduled.
        csv_file = tmp_path / "negative_lead.csv"
        csv_file.write_text(
            "PatientId,AppointmentID,Gender,ScheduledDay,AppointmentDay,Age,"
            "Neighbourhood,Scholarship,Hipertension,Diabetes,Alcoholism,Handcap,"
            "SMS_received,No-show\n"
            "5.0,2001,M,2016-05-10T10:00:00Z,2016-05-08T00:00:00Z,30,"
            "CENTRO,0,0,0,0,0,0,No\n",
            encoding="utf-8",
        )
        repo = KaggleAppointmentCSVRepository(csv_file)
        appointments = repo.get_appointments()
        assert appointments[0].lead_time_days == 0


class TestCSVRepositoryLabels:
    """Tests for label alignment and mapping."""

    def test_aligned_length(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        appointments = repo.get_appointments()
        labels = repo.get_labels()
        assert len(labels) == len(appointments)

    def test_labels_are_booleans(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        labels = repo.get_labels()
        assert all(isinstance(label, bool) for label in labels)

    def test_yes_maps_to_true(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        labels = repo.get_labels()
        # Row 2 (index 1): No-show=Yes → True
        assert labels[1] is True

    def test_no_maps_to_false(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        labels = repo.get_labels()
        # Row 1 (index 0): No-show=No → False
        assert labels[0] is False


class TestCSVRepositoryDateFiltering:
    """Tests for date-range filtering in get_appointments and get_labels."""

    def test_filter_by_start_date_june_only(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        start = datetime(2016, 6, 1, tzinfo=timezone.utc)
        appointments = repo.get_appointments(start_date=start)
        # Rows 5 and 6 have AppointmentDay in June
        assert len(appointments) == 2
        for appt in appointments:
            assert appt.appointment_day >= start

    def test_filter_by_end_date_before_may_1(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        end = datetime(2016, 5, 1, tzinfo=timezone.utc)
        appointments = repo.get_appointments(end_date=end)
        # Only row 1: AppointmentDay=2016-04-29
        assert len(appointments) == 1
        for appt in appointments:
            assert appt.appointment_day < end

    def test_labels_filtered_same_as_appointments(self, sample_csv):
        repo = KaggleAppointmentCSVRepository(sample_csv)
        start = datetime(2016, 6, 1, tzinfo=timezone.utc)
        appointments = repo.get_appointments(start_date=start)
        labels = repo.get_labels(start_date=start)
        assert len(labels) == len(appointments)
        # Row 5 (index 4): No-show=No → False; Row 6 (index 5): No-show=Yes → True
        assert labels == [False, True]
