# Coding Standards

## Python

- Python 3.12+
- Formatting: `black` (line-length 88)
- Type checking: `mypy` with strict mode enabled
- Linting: `ruff`
- Import sorting: `isort` (profile: black)
- No bare `except` — use specific exception types
- Type hints on all public function signatures
- Prefer `X | None` over `Optional[X]` (Python 3.12 syntax)

## Naming Conventions

- **Variables and functions**: `snake_case` (e.g., `get_appointments`, `predict_no_show`)
- **Classes**: `PascalCase` (e.g., `AppointmentRepository`, `Patient`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `RISK_THRESHOLD`, `LEAKAGE_COLUMNS`)
- **Modules**: `snake_case` (e.g., `csv_repository.py`, `use_cases.py`)
- **Test functions**: `test_<description>` (e.g., `test_no_show_high_lead_time`)
- **Private methods**: prefix with `_` (e.g., `_parse_scheduled_day`, `_encode_neighbourhood`)

## Architecture Rules (NON-NEGOTIABLE)

- Hexagonal architecture: `domain/` → `adapters/` → `application/`
- `domain/` has ZERO imports from `adapters/`, `application/`, or external frameworks
- `domain/` imports ONLY: `typing`, `dataclasses`, `datetime`, `enum`, `collections.abc`
- Adapters implement domain port `Protocol` interfaces from `domain/ports.py`
- `application/` orchestrates domain + adapters — it is the composition root
- New external tool = new adapter. Never put framework code in `domain/`
- Domain models use `@dataclass(frozen=True)` for immutable entities

## Data Integrity Rules (NON-NEGOTIABLE)

- **Leakage rule:** All features must be knowable at scheduling time (pre-appointment)
- `DataLeakageError` raised if post-appointment features detected
- Target encoding (neighbourhood) fit on training data ONLY — never on full dataset
- GroupKFold by PatientId — same patient never in train and test
- SMS_received is intervention feature — include but document confound (ADR-006)

## Testing Rules (NON-NEGOTIABLE)

- Tests use small fixtures — NEVER load the full dataset
- Property-based tests with Hypothesis for domain invariants
- pytest with `-v --tb=short` default
- Test categories to cover:
  - **Happy path**: valid input, expected output
  - **Error path**: invalid input, correct exception raised
  - **Boundary**: exact edge of a condition
  - **Edge case**: extreme but valid input
- One logical assertion per test function
- Use `pytest.raises` for expected exceptions
- Use `pytest.approx` for float comparisons
- Use `tmp_path` fixture for file I/O tests

## Project Layout

```
domain/                 Pure business logic
├── models.py           Patient, Appointment, NoShowOutcome (frozen dataclasses)
├── ports.py            AppointmentRepository, NoShowPredictorPort (Protocols)
├── services.py         Baseline no-show risk heuristic
└── exceptions.py       DataLeakageError, InvalidAppointmentDataError

adapters/               External connections
├── data/               KaggleAppointmentCSVRepository
├── ml/                 Logistic, XGBoost, CalibratedXGBoost adapters
└── visualization/      (Phase 2 — Streamlit components)

application/            Orchestration
└── use_cases.py        train_model, predict_no_show

tests/                  Mirrors source layout

notebooks/              EDA only — no production logic
docs/adr/               Architecture Decision Records (ADR-001 through ADR-013)
data/raw/               Untouched source data (gitignored)
data/interim/           Intermediate artifacts (gitignored)
data/processed/         Model-ready data (gitignored)
reports/                EDA gate, model metrics, fairness
```

## Git (NON-NEGOTIABLE)

- Commit format: `feat:` / `fix:` / `docs:` / `chore:` / `test:` followed by lowercase description, no period
- Keep commits small and focused
- Never commit directly to `main` or `dev` — use feature branches
- Branch naming: `feat/<slug>` or `fix/<slug>`
- PR target: `dev` (confirm with user before targeting `main`)
- Never commit secrets, raw data, model artifacts, or `.env` files
- Prefer new commits over `--amend` on pushed branches

## Commands

```bash
# Test
pytest -v --tb=short
pytest -v --cov=domain --cov=adapters --cov=application --tb=short

# Lint and format
pre-commit run --all-files

# Type check
mypy domain/ adapters/ application/ --strict

# Environment
pip install -e ".[dev]"
```

## Strong Preferences

These are not hard stops but should be followed unless there is a clear reason not to:

- Use structured logging over `print()` (loguru when added)
- Avoid heavyweight dependencies without justification (e.g., no TensorFlow for tabular data)
- Prefer `X | None` over `Optional[X]` for modern Python 3.12 annotations
- Type hints on private functions too when practical
