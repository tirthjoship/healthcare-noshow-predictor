# Requirements Document: Patient Readmission Risk Engine

## Introduction

This document specifies the requirements for initializing a new ML project (patient-readmission-risk-engine) that replicates the architectural and quality standards from the supply-chain-optimization-ml golden template. The project will implement a Hexagonal Architecture with strict governance standards, comprehensive testing infrastructure, and production-grade documentation practices for healthcare ML systems.

## Glossary

- **Project_Initializer**: The system component responsible for creating the project structure and configuration files
- **Golden_Template**: The reference implementation at /Users/tirthjoshi/My Data Science Projects/ML_Portfolio_Projects/supply-chain-optimization-ml
- **Hexagonal_Architecture**: Ports and Adapters architectural pattern with inward-pointing dependencies
- **Domain_Layer**: Pure Python business logic with zero external dependencies (domain/ directory)
- **Adapter_Layer**: External connections for data, ML models, and visualization (adapters/ directory)
- **Application_Layer**: Use case orchestration bridging domain and adapters (application/ directory)
- **Digital_Constitution**: The governance configuration enforcing Python 3.12, strict typing, and code quality standards
- **Pre_Commit_Hooks**: Automated code quality checks (Black, Ruff, isort, Mypy) executed on git commit
- **Property_Based_Testing**: Testing approach using Hypothesis library to validate invariants across generated inputs
- **Audit_Trail**: LOCAL_PROJECT_STORY.md documentation with Director's Cut narrative detail
- **Healthcare_Domain**: The patient readmission risk prediction business domain
- **CSVRepository**: Data adapter implementing PatientDataRepository port for loading admission data from CSV files
- **RiskPredictorPort**: Port interface for ML model adapters that predict readmission risk
- **DataLeakageError**: Custom exception raised when post-discharge features are detected in training or prediction data
- **LEAKAGE_COLUMNS**: Constant defining prohibited post-discharge features that would cause data leakage

## Requirements

### Requirement 1: Replicate Physical Folder Structure

**User Story:** As a ML systems architect, I want the exact folder structure from the golden template replicated, so that the project enforces Hexagonal Architecture separation of concerns from day one.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create a domain/ directory containing __init__.py, models.py, ports.py, services.py, and exceptions.py files
2. THE Project_Initializer SHALL create an adapters/ directory with subdirectories data/, ml/, and visualization/, each containing __init__.py
3. THE Project_Initializer SHALL create an application/ directory containing __init__.py and use_cases.py files
4. THE Project_Initializer SHALL create a tests/ directory containing __init__.py
5. THE Project_Initializer SHALL create a data/ directory with subdirectories raw/, interim/, and processed/
6. THE Project_Initializer SHALL create a notebooks/ directory for exploratory analysis
7. WHEN the folder structure is created, THE Domain_Layer SHALL have zero imports from Adapter_Layer or Application_Layer (inward-pointing dependencies only)

**Boeing Alignment:** Quality/Reliability

### Requirement 2: Replicate Digital Constitution Configuration

**User Story:** As a ML systems architect, I want the pyproject.toml configuration replicated with healthcare-specific adaptations, so that Python 3.12 strict typing and code quality standards are enforced automatically.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create a pyproject.toml file with requires-python = ">=3.12"
2. THE Project_Initializer SHALL configure [tool.mypy] with strict = true and python_version = "3.12"
3. THE Project_Initializer SHALL configure [tool.black] with line-length = 88 and target-version = ["py312"]
4. THE Project_Initializer SHALL configure [tool.isort] with profile = "black"
5. THE Project_Initializer SHALL declare core dependencies: pandas>=2.0.0, scikit-learn>=1.3.0, xgboost>=2.0.0, mlflow>=2.0.0, hypothesis>=6.0.0, pytest>=7.0.0
6. THE Project_Initializer SHALL declare dev dependencies: pytest-cov>=4.0.0, black>=24.0.0, isort>=5.13.0, mypy>=1.8.0, ruff>=0.1.0
7. THE Project_Initializer SHALL configure [tool.pytest.ini_options] with testpaths = ["tests"], pythonpath = ["."], and addopts = "-v"
8. THE Project_Initializer SHALL set project name to "patient_readmission_risk" and version to "0.1.0"

**Boeing Alignment:** Quality/Reliability

### Requirement 3: Replicate Pre-Commit Hook Configuration

**User Story:** As a ML systems architect, I want the .pre-commit-config.yaml replicated exactly, so that Black, isort, Mypy, and Ruff automatically enforce code quality on every commit.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create a .pre-commit-config.yaml file with four hooks: black, isort, mypy, and ruff
2. THE Project_Initializer SHALL configure the black hook with rev: 24.1.1 and language_version: python3.12
3. THE Project_Initializer SHALL configure the isort hook with rev: 5.13.2 and args: ["--profile", "black"]
4. THE Project_Initializer SHALL configure the mypy hook with rev: v1.8.0 and args: ["--strict"]
5. THE Project_Initializer SHALL configure the ruff hook with rev: v0.1.15
6. FOR ALL four hooks, THE Project_Initializer SHALL use the exact repository URLs from the golden template

**Boeing Alignment:** Quality/Reliability

### Requirement 4: Replicate Conda Environment Configuration

**User Story:** As a ML systems architect, I want the environment.yml replicated with healthcare-specific naming, so that the Python 3.12 environment with all dependencies can be reproduced identically.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create an environment.yml file with name: patient-readmission-ml
2. THE Project_Initializer SHALL specify channels: conda-forge and defaults
3. THE Project_Initializer SHALL declare conda dependencies: python=3.12, pip, jupyterlab>=4.0, pandas>=2.0.0, numpy>=1.26, scikit-learn>=1.3.0, matplotlib>=3.8, seaborn>=0.13
4. THE Project_Initializer SHALL declare pip dependencies: xgboost>=2.0.0, mlflow>=2.0.0, hypothesis>=6.0.0, shap>=0.44, streamlit>=1.30, black>=24.0.0, ruff>=0.1.0, mypy>=1.8.0, isort>=5.13, pytest>=7.0.0, pytest-cov>=4.0.0, pre-commit>=3.6
5. THE Project_Initializer SHALL match all version constraints exactly to the golden template

**Boeing Alignment:** Quality/Reliability

### Requirement 5: Create Healthcare Domain Models

**User Story:** As a ML systems architect, I want domain models for healthcare entities created, so that the domain layer represents patient, admission, and readmission risk concepts in pure Python.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create domain/models.py with dataclass entities for Patient, Admission, and RiskAssessment
2. WHEN defining domain models, THE Project_Initializer SHALL use frozen=True for immutable value objects
3. THE Project_Initializer SHALL include comprehensive Google-style docstrings for all dataclasses and fields
4. THE Project_Initializer SHALL use type hints for all fields (str, int, float, datetime, Optional types)
5. THE Project_Initializer SHALL include __post_init__ validation methods where business invariants must be enforced
6. THE Project_Initializer SHALL ensure domain/models.py has zero imports from pandas, numpy, scikit-learn, or any external ML libraries

**Boeing Alignment:** Safety/Risk

### Requirement 6: Create Healthcare Domain Ports

**User Story:** As a ML systems architect, I want abstract port interfaces defined, so that adapters can be swapped without modifying domain logic.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create domain/ports.py with Protocol classes for PatientDataRepository and RiskPredictorPort
2. THE Project_Initializer SHALL define PatientDataRepository protocol with methods get_admissions() returning list[Admission]
3. THE Project_Initializer SHALL define RiskPredictorPort protocol with method predict_readmission_risk() accepting Admission and returning RiskAssessment
4. THE Project_Initializer SHALL include comprehensive Google-style docstrings explaining the contract for each protocol method
5. THE Project_Initializer SHALL use typing.Protocol for all port definitions

**Boeing Alignment:** Quality/Reliability

### Requirement 7: Create Healthcare Domain Services

**User Story:** As a ML systems architect, I want domain service functions for business logic created, so that rule-based risk assessment logic exists independently of ML models.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create domain/services.py with pure Python functions for baseline risk assessment
2. THE Project_Initializer SHALL implement a baseline_readmission_risk_flag() function using only pre-discharge features: admission_type, primary_diagnosis, comorbidity_count, length_of_stay_scheduled, prior_admissions_count
3. THE Project_Initializer SHALL include comprehensive Google-style docstrings with Args, Returns, and Example sections
4. THE Project_Initializer SHALL ensure domain/services.py has zero imports from pandas, numpy, scikit-learn, or any external ML libraries
5. THE Project_Initializer SHALL define HIGH_RISK_CONDITIONS as a frozenset constant for immutable reference data

**Boeing Alignment:** Safety/Risk

### Requirement 8: Create Healthcare Domain Exceptions

**User Story:** As a ML systems architect, I want custom domain exceptions defined, so that domain errors are explicit and traceable.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create domain/exceptions.py with a base DomainError exception class
2. THE Project_Initializer SHALL define InvalidAdmissionDataError exception inheriting from DomainError
3. THE Project_Initializer SHALL define InvalidRiskAssessmentError exception inheriting from DomainError
4. THE Project_Initializer SHALL include comprehensive Google-style docstrings explaining when each exception should be raised

**Boeing Alignment:** Quality/Reliability

### Requirement 9: Create Testing Infrastructure

**User Story:** As a ML systems architect, I want the testing infrastructure replicated, so that TDD workflow with property-based testing is enabled from day one.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create tests/__init__.py as an empty file
2. THE Project_Initializer SHALL create tests/test_domain_models.py with skeleton test functions for Patient, Admission, and RiskAssessment validation
3. THE Project_Initializer SHALL create tests/test_domain_services.py with skeleton test functions for baseline risk assessment logic
4. THE Project_Initializer SHALL create tests/test_properties.py with Hypothesis strategy examples for property-based testing
5. WHEN test files are created, THE Project_Initializer SHALL include import statements for pytest, hypothesis, and domain modules
6. THE Project_Initializer SHALL include docstring comments explaining the TDD workflow and property-based testing approach

**Boeing Alignment:** Quality/Reliability

### Requirement 10: Create Audit Trail Documentation

**User Story:** As a ML systems architect, I want the LOCAL_PROJECT_STORY.md template created, so that the Director's Cut narrative documentation standard is established.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create LOCAL_PROJECT_STORY.md with the exact structure from the golden template
2. THE Project_Initializer SHALL include sections: The Expert Architect's Opening, The Interconnected Team, Phase 0-5 structure, UBC MDS Alignment table
3. THE Project_Initializer SHALL adapt the narrative content to healthcare/patient readmission context while preserving the tone and detail level
4. THE Project_Initializer SHALL include the "What We Settled On" session format for recording architectural decisions
5. THE Project_Initializer SHALL include the Technology Decisions and What We Decided NOT to Use sections
6. THE Project_Initializer SHALL preserve the exact markdown formatting, table structures, and section hierarchy from the golden template

**Boeing Alignment:** Business Impact

### Requirement 11: Create Git Configuration Files

**User Story:** As a ML systems architect, I want .gitignore and README.md created, so that the repository follows professional git hygiene standards.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create a .gitignore file excluding: __pycache__/, .pytest_cache/, .hypothesis/, .mypy_cache/, *.pyc, .DS_Store, data/raw/*, data/interim/*, data/processed/*, mlruns/, LOCAL_PROJECT_STORY.md, .kiro/
2. THE Project_Initializer SHALL create a README.md with sections: Project Overview, Architecture, Technology Stack, Setup Instructions, Testing, and Project Status
3. THE Project_Initializer SHALL adapt README content to patient readmission risk context
4. THE Project_Initializer SHALL follow the golden template's writing style: no em dashes, no contractions, short paragraphs, no hyperbole
5. THE Project_Initializer SHALL include a badge placeholder for GitHub Actions CI status

**Boeing Alignment:** Quality/Reliability

### Requirement 12: Create Adapter Layer Skeleton

**User Story:** As a ML systems architect, I want adapter layer skeleton files created, so that the structure for CSV data loading, ML model training, and visualization is established.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create adapters/data/__init__.py and adapters/data/csv_repository.py with a skeleton CSVRepository class
2. THE Project_Initializer SHALL create adapters/ml/__init__.py and adapters/ml/xgboost_predictor.py with a skeleton XGBoostPredictor class
3. THE Project_Initializer SHALL create adapters/visualization/__init__.py with placeholder for future Streamlit or plotting adapters
4. WHEN adapter classes are created, THE Project_Initializer SHALL include type hints showing they implement the corresponding domain ports
5. THE Project_Initializer SHALL include comprehensive Google-style docstrings explaining the adapter's responsibility
6. THE Project_Initializer SHALL include TODO comments indicating where implementation logic will be added in future phases

**Boeing Alignment:** Quality/Reliability

### Requirement 13: Create Application Layer Skeleton

**User Story:** As a ML systems architect, I want application layer skeleton created, so that use case orchestration structure is established.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create application/use_cases.py with skeleton functions for train_model() and predict_readmission_risk()
2. WHEN use case functions are created, THE Project_Initializer SHALL include type hints for parameters and return values
3. THE Project_Initializer SHALL include comprehensive Google-style docstrings explaining the use case workflow
4. THE Project_Initializer SHALL include TODO comments indicating where domain service and adapter calls will be orchestrated

**Boeing Alignment:** Business Impact

### Requirement 14: Validate Hexagonal Architecture Compliance

**User Story:** As a ML systems architect, I want architectural compliance validated, so that inward-pointing dependencies are guaranteed before any implementation begins.

#### Acceptance Criteria

1. WHEN all files are created, THE Project_Initializer SHALL verify that domain/ files import only from typing, dataclasses, datetime, and other domain modules
2. WHEN all files are created, THE Project_Initializer SHALL verify that adapters/ files import from domain/ but domain/ never imports from adapters/
3. WHEN all files are created, THE Project_Initializer SHALL verify that application/ files import from both domain/ and adapters/ for orchestration
4. IF any dependency violation is detected, THEN THE Project_Initializer SHALL raise an InvalidArchitectureError with a descriptive message

**Boeing Alignment:** Safety/Risk

### Requirement 15: Create Initial Notebook Template

**User Story:** As a ML systems architect, I want an initial EDA notebook template created, so that exploratory analysis follows the golden template's structure.

#### Acceptance Criteria

1. THE Project_Initializer SHALL create notebooks/eda_initial.ipynb with markdown cells for: Data Loading, Data Quality Assessment, Target Variable Analysis, Feature Distribution Analysis, and Statistical Tests
2. THE Project_Initializer SHALL include code cell templates for loading data via the CSVRepository adapter
3. THE Project_Initializer SHALL include markdown guidance on avoiding data leakage (e.g., excluding post-discharge features)
4. THE Project_Initializer SHALL include placeholder cells for chi-squared tests and clustering analysis matching the golden template's approach

**Boeing Alignment:** Business Impact

### Requirement 16: Enforce Data Leakage Prevention Protocol

**User Story:** As a ML systems architect, I want explicit leakage prevention enforced in the data adapter, so that no future-looking clinical features contaminate the training or prediction pipeline.

#### Acceptance Criteria

1. THE CSVRepository adapter SHALL prune all post-discharge features including: discharge_status, post_discharge_medications, readmission_occurred, days_until_readmission
2. WHEN loading admission data, THE CSVRepository SHALL validate that only pre-discharge features are present in the feature matrix
3. IF any leakage column is detected in the loaded data, THEN THE CSVRepository SHALL raise a DataLeakageError with the column name
4. THE domain/services.py baseline_readmission_risk_flag() function SHALL use ONLY pre-discharge features: admission_type, primary_diagnosis, comorbidity_count, length_of_stay_scheduled, prior_admissions_count
5. THE Project_Initializer SHALL create a LEAKAGE_COLUMNS constant in adapters/data/csv_repository.py listing all prohibited features
6. THE testing suite SHALL include a property-based test verifying that no leakage columns appear in any feature matrix passed to the ML model
7. WHILE in a clinical setting, THE RiskPredictorPort SHALL flag a patient as 'High Risk' only IF the predicted probability of readmission exceeds a 0.75 threshold to ensure high precision and minimize alarm fatigue

**Boeing Alignment:** Safety/Risk

---

## Cross-Project Technical Decision Log (TDL)

**Decision ID:** TDL-001-Healthcare-Architecture-Standards
**Date:** 2026-02-25
**Status:** Approved
**Scope:** patient-readmission-risk-engine, multi-modal-stock-recommender (future)

### Context

Both healthcare and finance ML projects in the portfolio must maintain identical architectural DNA to ensure reproducibility, auditability, and cross-domain knowledge transfer.

### Decision

Enforce the following standards across ALL portfolio projects:

1. **Hexagonal Architecture**: Inward-pointing dependencies with domain/ having zero external framework imports
2. **Digital Constitution**: Python 3.12, Mypy --strict, Black, Ruff, isort, pre-commit hooks
3. **Property-Based Testing**: Hypothesis library with 90%+ coverage target
4. **Leakage Prevention**: Explicit LEAKAGE_COLUMNS constants in all data adapters with validation tests
5. **EARS Notation**: All acceptance criteria use WHILE/WHEN/IF patterns for unambiguous testability
6. **Boeing Alignment**: Every requirement categorized as Safety/Risk, Quality/Reliability, or Business Impact

### Rationale

- **Reproducibility**: Identical pyproject.toml and pre-commit configs eliminate "works on my machine" issues
- **Auditability**: EARS notation provides legally defensible requirement traceability for healthcare/finance domains
- **Safety**: Leakage prevention protocols prevent models from learning spurious patterns that fail in production
- **Maintainability**: Hexagonal Architecture allows swapping data sources (CSV → database) without touching domain logic

### Consequences

- **Positive**: Portfolio demonstrates industrial-grade systems engineering to employers (Amazon, Walmart, VGH)
- **Positive**: Cross-project code review becomes trivial due to identical structure
- **Negative**: Initial scaffolding time increases by ~2 hours per project (acceptable trade-off)

### Compliance Verification

Each project MUST pass the following audits before Phase 1 completion:

1. `mypy --strict` returns zero errors
2. All tests in tests/test_properties.py pass with Hypothesis
3. Architecture compliance test verifies domain/ has zero adapter imports
4. Leakage prevention test verifies LEAKAGE_COLUMNS are excluded from feature matrices
