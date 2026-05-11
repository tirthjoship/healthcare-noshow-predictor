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

- **Variables and functions**: `snake_case` (e.g., `get_admissions`, `compute_readmission_risk`)
- **Classes**: `PascalCase` (e.g., `AdmissionRepository`, `Patient`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `RISK_THRESHOLD`, `EXCLUDED_COLUMNS`)
- **Modules**: `snake_case` (e.g., `csv_repository.py`, `use_cases.py`)
- **Test functions**: `test_<description>` (e.g., `test_readmission_within_30_days`)
- **Private methods**: prefix with `_` (e.g., `_parse_admission_date`, `_safe_float`)

## Architecture Rules (NON-NEGOTIABLE)

- Hexagonal architecture: `domain/` → `adapters/` → `application/`
- `domain/` has ZERO imports from `adapters/`, `application/`, or external frameworks
- `domain/` imports ONLY: `typing`, `dataclasses`, `datetime`, `enum`, `collections.abc`
- Adapters implement domain port `Protocol` interfaces from `domain/ports.py`
- `application/` orchestrates domain + adapters — it is the composition root
- New external tool = new adapter. Never put framework code in `domain/`
- Domain models use `@dataclass(frozen=True)` for immutable entities

## Data Integrity Rules (NON-NEGOTIABLE)

- Domain-specific leakage columns and evaluation rules to be defined after brainstorming
- General rule: all feature engineering must use only pre-discharge information
- Evaluate models appropriately for class distribution (check imbalance before choosing metrics)

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
├── models.py           Frozen dataclasses
├── ports.py            Protocol interfaces
├── services.py         Business rules
└── exceptions.py       Domain-specific errors

adapters/               External connections
├── data/               Data source connectors
├── ml/                 Model adapters
└── visualization/      Charting adapters

application/            Orchestration
└── use_cases.py        Wires domain + adapters for business workflows

tests/                  Mirrors source layout

notebooks/              Exploration and EDA only — no production logic
data/raw/               Untouched source data (gitignored)
data/interim/           Intermediate artifacts (gitignored)
data/processed/         Model-ready data (gitignored)
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
conda activate patient-readmission
```

## Strong Preferences

These are not hard stops but should be followed unless there is a clear reason not to:

- Use structured logging over `print()` (loguru when added)
- Avoid heavyweight dependencies without justification (e.g., no TensorFlow for tabular data)
- Prefer `X | None` over `Optional[X]` for modern Python 3.12 annotations
- Type hints on private functions too when practical
