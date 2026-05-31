"""Shared test fixtures for healthcare-noshow-predictor."""

import textwrap

import pytest

SAMPLE_CSV_CONTENT = textwrap.dedent("""\
    PatientId,AppointmentID,Gender,ScheduledDay,AppointmentDay,Age,Neighbourhood,Scholarship,Hipertension,Diabetes,Alcoholism,Handcap,SMS_received,No-show
    1.0,1001,F,2016-04-29T18:38:08Z,2016-04-29T00:00:00Z,62,JARDIM DA PENHA,0,1,0,0,0,0,No
    1.0,1002,F,2016-04-30T10:00:00Z,2016-05-05T00:00:00Z,62,JARDIM DA PENHA,0,1,0,0,0,1,Yes
    2.0,1003,M,2016-04-29T08:00:00Z,2016-05-10T00:00:00Z,25,CENTRO,1,0,0,0,0,0,No
    3.0,1004,F,2016-05-01T12:00:00Z,2016-05-15T00:00:00Z,35,RESISTÊNCIA,0,0,1,0,1,1,Yes
    4.0,1005,M,2016-05-02T09:00:00Z,2016-06-01T00:00:00Z,45,CENTRO,0,0,0,1,0,0,No
    4.0,1006,M,2016-05-03T14:00:00Z,2016-06-05T00:00:00Z,45,CENTRO,0,0,0,1,0,1,Yes
    """)


@pytest.fixture
def sample_csv(tmp_path):
    """Write the 6-row sample CSV to a temp file and return its path."""
    csv_file = tmp_path / "appointments.csv"
    csv_file.write_text(SAMPLE_CSV_CONTENT, encoding="utf-8")
    return csv_file
