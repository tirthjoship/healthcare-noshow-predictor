# Implementation Plan: Patient Readmission Risk Engine

## Overview

This implementation plan breaks down the Patient Readmission Risk Engine into actionable coding tasks following a phased development approach. The system implements Hexagonal Architecture with strict dependency inversion, data leakage prevention protocols, and industrial-grade testing standards using Python 3.12.

The implementation follows a 6-phase roadmap: Foundation (scaffolding), Domain Layer, Testing Infrastructure, Adapter Layer, Application Layer, and Documentation. Each task references specific requirements and design sections for traceability.

## Tasks

- [ ] 1. Phase 0: Foundation and Project Scaffolding
  - [ ] 1.1 Create project directory structure
    - Create domain/ directory with __init__.py, models.py, ports.py, services.py, exceptions.py
    - Create adapters/ directory with subdirectories data/, ml/, visualization/, each with __init__.py
    - Create application/ directory with __init__.py, use_cases.py
    - Create tests/ directory with __init__.py
    - Create data/ directory with subdirectories raw/, interim/, processed/
    - Create notebooks/ directory
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ] 1.2 Create pyproject.toml configuration file
    - Set project name to "patient_readmission_risk" and version to "0.1.0"
    - Configure requires-python = ">=3.12"
    - Declare core dependencies: pandas>=2.0.0, scikit-learn>=1.3.0, xgboost>=2.0.0, mlflow>=2.0.0, hypothesis>=6.0.0, pytest>=7.0.0
    - Declare dev dependencies: pytest-cov>=4.0.0, black>=24.0.0, isort>=5.13.0, mypy>=1.8.0, ruff>=0.1.0
    - Configure [tool.mypy] with strict = true and python_version = "3.12"
    - Configure [tool.black] with line-length = 88 and target-version = ["py312"]
    - Configure [tool.isort] with profile = "black"
    - Configure [tool.pytest.ini_options] with testpaths = ["tests"], pythonpath = ["."], addopts = "-v"
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ] 1.3 Create .pre-commit-config.yaml configuration file
    - Configure black hook with rev: 24.1.1, language_version: python3.12, repo: https://github.com/psf/black
    - Configure isort hook with rev: 5.13.2, args: ["--profile", "black"], repo: https://github.com/pycqa/isort
    - Configure mypy hook with rev: v1.8.0, args: ["--strict"], repo: https://github.com/pre-commit/mirrors-mypy
    - Configure ruff hook with rev: v0.1.15, repo: https://github.com/astral-sh/ruff-pre-commit
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ] 1.4 Create environment.yml conda configuration file
    - Set name: patient-readmission-ml
    - Specify channels: conda-forge, defaults
    - Declare conda dependencies: python=3.12, pip, jupyterlab>=4.0, pandas>=2.0.0, numpy>=1.26, scikit-learn>=1.3.0, matplotlib>=3.8, seaborn>=0.13
    - Declare pip dependencies: xgboost>=2.0.0, mlflow>=2.0.0, hypothesis>=6.0.0, shap>=0.44, streamlit>=1.30, black>=24.0.0, ruff>=0.1.0, mypy>=1.8.0, isort>=5.13, pytest>=7.0.0, pytest-cov>=4.0.0, pre-commit>=3.6
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ] 1.5 Create .gitignore file
    - Exclude Python artifacts: __pycache__/, .pytest_cache/, .hypothesis/, .mypy_cache/, *.pyc
    - Exclude IDE files: .DS_Store, .vscode/, .idea/
    - Exclude data directories: data/raw/*, data/interim/*, data/processed/*
    - Exclude ML artifacts: mlruns/, models/*.pkl, models/*.joblib
    - Exclude local documentation: LOCAL_PROJECT_STORY.md
    - Exclude Kiro directory: .kiro/
    - Exclude Jupyter checkpoints: .ipynb_checkpoints/
    - _Requirements: 11.1_

  - [ ] 1.6 Create README.md with project overview
    - Include sections: Project Overview, Architecture, Technology Stack, Setup Instructions, Testing, Project Status
    - Adapt content to patient readmission risk context
    - Include Hexagonal Architecture diagram description
    - Include badge placeholder for GitHub Actions CI status
    - Follow writing style: no em dashes, no contractions, short paragraphs, no hyperbole
    - _Requirements: 11.2, 11.3, 11.4, 11.5_

- [ ] 2. Phase 1: Domain Layer Implementation
  - [ ] 2.1 Implement domain/models.py with Patient, Admission, RiskAssessment dataclasses
    - Create Patient dataclass with fields: patient_id (str), age (int), gender (str), insurance_type (str)
    - Set frozen=True for immutability on all dataclasses
    - Add __post_init__ validation: age >= 0, gender in {'M', 'F', 'Other'}
    - Create Admission dataclass with fields: admission_id, patient, admission_date, admission_type, primary_diagnosis, comorbidity_count, length_of_stay_scheduled, prior_admissions_count, lab_abnormalities_count, medication_count
    - Add __post_init__ validation: admission_type in {'Emergency', 'Elective', 'Urgent'}, all counts >= 0
    - Create RiskAssessment dataclass with fields: admission_id, risk_score, risk_category, assessment_timestamp, model_version, explanation (Optional)
    - Add __post_init__ validation: risk_score in [0.0, 1.0], risk_category in {'Low Risk', 'Medium Risk', 'High Risk'}
    - Include comprehensive Google-style docstrings for all dataclasses and fields
    - Use type hints for all fields
    - Ensure zero imports from pandas, numpy, scikit-learn, or external ML libraries
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 2.2 Write unit tests for domain models
    - Test Patient validation: negative age raises InvalidAdmissionDataError
    - Test Patient validation: invalid gender raises InvalidAdmissionDataError
    - Test Patient immutability: modifying frozen dataclass raises FrozenInstanceError
    - Test Admission validation: invalid admission_type raises InvalidAdmissionDataError
    - Test Admission validation: negative counts raise InvalidAdmissionDataError
    - Test RiskAssessment validation: risk_score outside [0, 1] raises InvalidRiskAssessmentError
    - Test RiskAssessment validation: invalid risk_category raises InvalidRiskAssessmentError
    - _Requirements: 5.5, 9.2_

  - [ ] 2.3 Implement domain/ports.py with PatientDataRepository and RiskPredictorPort protocols
    - Create PatientDataRepository protocol with method get_admissions(start_date, end_date) -> list[Admission]
    - Create RiskPredictorPort protocol with methods predict_readmission_risk(admission) -> RiskAssessment and train(admissions, outcomes) -> None
    - Use typing.Protocol for all port definitions
    - Include comprehensive Google-style docstrings explaining contract for each protocol method
    - Document that implementations must enforce leakage prevention
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ] 2.4 Implement domain/exceptions.py with custom exception hierarchy
    - Create base DomainError exception class
    - Create InvalidAdmissionDataError inheriting from DomainError
    - Create InvalidRiskAssessmentError inheriting from DomainError
    - Create DataLeakageError inheriting from DomainError
    - Include comprehensive Google-style docstrings explaining when each exception should be raised
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ] 2.5 Implement domain/services.py with baseline risk assessment logic
    - Define HIGH_RISK_CONDITIONS as frozenset containing ICD-10 codes: 'I50', 'J44', 'E11', 'N18', 'I25', 'J18', 'I63', 'K70'
    - Implement baseline_readmission_risk_flag(admission: Admission) -> str function
    - Use only pre-discharge features: admission_type, primary_diagnosis, comorbidity_count, length_of_stay_scheduled, prior_admissions_count
    - Implement risk logic: High Risk if >= 3 risk factors, Medium Risk if >= 1 risk factor, Low Risk otherwise
    - Risk factors: diagnosis in HIGH_RISK_CONDITIONS, comorbidity_count >= 3, prior_admissions_count >= 2, admission_type == 'Emergency', length_of_stay_scheduled >= 7
    - Include comprehensive Google-style docstrings with Args, Returns, and Example sections
    - Ensure zero imports from pandas, numpy, scikit-learn, or external ML libraries
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 16.4_

  - [ ]* 2.6 Write unit tests for domain services
    - Test baseline_readmission_risk_flag returns 'High Risk' for emergency admission with heart failure diagnosis
    - Test baseline_readmission_risk_flag returns 'Medium Risk' for elective admission with 2 comorbidities
    - Test baseline_readmission_risk_flag returns 'Low Risk' for elective admission with no risk factors
    - Test baseline_readmission_risk_flag with edge cases: zero comorbidities, empty diagnosis
    - _Requirements: 7.2, 9.3_

- [ ] 3. Phase 2: Testing Infrastructure and Property-Based Tests
  - [ ] 3.1 Create tests/test_domain_models.py with unit test skeleton
    - Import pytest, hypothesis, and domain.models
    - Create test functions for Patient, Admission, RiskAssessment validation
    - Include docstring comments explaining TDD workflow
    - _Requirements: 9.2, 9.5_

  - [ ] 3.2 Create tests/test_domain_services.py with unit test skeleton
    - Import pytest and domain.services
    - Create test functions for baseline_readmission_risk_flag
    - Include docstring comments explaining baseline logic testing approach
    - _Requirements: 9.3, 9.5_

  - [ ] 3.3 Implement tests/test_properties.py with 15 property-based tests
    - [ ] 3.3.1 Write property test for Hexagonal Architecture Dependency Inversion
      - **Property 1: Hexagonal Architecture Dependency Inversion**
      - **Validates: Requirements 1.7, 5.6, 7.4, 14.1**
      - Use AST parsing to verify domain/ files import only from typing, dataclasses, datetime, enum, or other domain modules
      - Verify no imports from adapters/, application/, pandas, numpy, scikit-learn, xgboost
      - _Requirements: 1.7, 5.6, 7.4, 14.1_

    - [ ]* 3.3.2 Write property test for Adapter-Domain Dependency Flow
      - **Property 2: Adapter-Domain Dependency Flow**
      - **Validates: Requirements 14.2**
      - Use AST parsing to verify adapters/ files can import from domain/
      - Verify domain/ files never import from adapters/
      - _Requirements: 14.2_

    - [ ]* 3.3.3 Write property test for Immutable Domain Models
      - **Property 3: Immutable Domain Models**
      - **Validates: Requirements 5.2**
      - Use Hypothesis to generate valid patient data
      - Verify all dataclasses in domain/models.py have frozen=True
      - Verify modifying fields raises FrozenInstanceError
      - _Requirements: 5.2_

    - [ ]* 3.3.4 Write property test for Type-Annotated Domain Fields
      - **Property 4: Type-Annotated Domain Fields**
      - **Validates: Requirements 5.4**
      - Use AST parsing to verify all fields in domain/models.py dataclasses have type annotations
      - _Requirements: 5.4_

    - [ ]* 3.3.5 Write property test for Protocol-Based Port Definitions
      - **Property 5: Protocol-Based Port Definitions**
      - **Validates: Requirements 6.5**
      - Use AST parsing to verify all classes in domain/ports.py inherit from typing.Protocol
      - _Requirements: 6.5_

    - [ ]* 3.3.6 Write property test for Comprehensive Docstring Coverage
      - **Property 6: Comprehensive Docstring Coverage**
      - **Validates: Requirements 5.3, 6.4, 7.3, 8.4, 12.5, 13.3**
      - Use AST parsing to verify all public classes, functions, methods in domain/, adapters/, application/ have docstrings
      - _Requirements: 5.3, 6.4, 7.3, 8.4, 12.5, 13.3_

    - [ ]* 3.3.7 Write property test for Type-Annotated Use Case Functions
      - **Property 7: Type-Annotated Use Case Functions**
      - **Validates: Requirements 13.2**
      - Use AST parsing to verify all functions in application/use_cases.py have parameter and return type annotations
      - _Requirements: 13.2_

    - [ ]* 3.3.8 Write property test for Type-Annotated Adapter Implementations
      - **Property 8: Type-Annotated Adapter Implementations**
      - **Validates: Requirements 12.4**
      - Use AST parsing to verify adapter classes indicate which port protocol they implement
      - _Requirements: 12.4_

    - [ ]* 3.3.9 Write property test for Required Test Imports
      - **Property 9: Required Test Imports**
      - **Validates: Requirements 9.5**
      - Use AST parsing to verify all Python files in tests/ import pytest
      - _Requirements: 9.5_

    - [ ]* 3.3.10 Write property test for Leakage Column Exclusion
      - **Property 10: Leakage Column Exclusion**
      - **Validates: Requirements 16.1, 16.2, 16.3**
      - Use Hypothesis to generate DataFrames with and without leakage columns
      - Verify CSVRepository raises DataLeakageError when leakage columns detected
      - Verify LEAKAGE_COLUMNS constant includes: discharge_status, post_discharge_medications, readmission_occurred, days_until_readmission
      - _Requirements: 16.1, 16.2, 16.3_

    - [ ]* 3.3.11 Write property test for High-Risk Threshold Precision
      - **Property 11: High-Risk Threshold Precision**
      - **Validates: Requirements 16.7**
      - Use Hypothesis to generate risk scores in [0.0, 1.0]
      - Verify risk categorization: 'High Risk' if >= 0.75, 'Medium Risk' if >= 0.40 and < 0.75, 'Low Risk' if < 0.40
      - _Requirements: 16.7_

    - [ ]* 3.3.12 Write property test for Risk Score Bounds Validation
      - **Property 12: Risk Score Bounds Validation**
      - **Validates: Requirements 5.5**
      - Use Hypothesis to generate risk scores outside [0.0, 1.0]
      - Verify RiskAssessment raises InvalidRiskAssessmentError during __post_init__
      - _Requirements: 5.5_

    - [ ]* 3.3.13 Write property test for Pre-Commit Hook Repository URLs
      - **Property 13: Pre-Commit Hook Repository URLs**
      - **Validates: Requirements 3.6**
      - Parse .pre-commit-config.yaml and verify repository URLs match official repositories
      - Verify black uses https://github.com/psf/black
      - Verify isort uses https://github.com/pycqa/isort
      - Verify mypy uses https://github.com/pre-commit/mirrors-mypy
      - Verify ruff uses https://github.com/astral-sh/ruff-pre-commit
      - _Requirements: 3.6_

    - [ ]* 3.3.14 Write property test for Baseline Risk Assessment Feature Restriction
      - **Property 14: Baseline Risk Assessment Feature Restriction**
      - **Validates: Requirements 7.2, 16.4**
      - Use AST parsing to verify baseline_readmission_risk_flag only accesses pre-discharge features
      - Verify function accesses: admission_type, primary_diagnosis, comorbidity_count, length_of_stay_scheduled, prior_admissions_count
      - Verify function does NOT access: discharge_status, post_discharge_medications, readmission_occurred
      - _Requirements: 7.2, 16.4_

    - [ ]* 3.3.15 Write property test for Domain Exception Inheritance
      - **Property 15: Domain Exception Inheritance**
      - **Validates: Requirements 8.2, 8.3**
      - Use AST parsing to verify all exception classes in domain/exceptions.py (except DomainError) inherit from DomainError
      - _Requirements: 8.2, 8.3_

  - [ ] 3.4 Create tests/test_architecture.py for architecture compliance validation
    - Implement test_domain_has_no_adapter_imports to verify domain/ never imports from adapters/
    - Implement test_domain_has_no_external_ml_imports to verify domain/ never imports pandas, numpy, scikit-learn, xgboost
    - Implement test_adapters_can_import_domain to verify adapters/ can import from domain/
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 4. Checkpoint - Ensure domain layer and tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Phase 3: Adapter Layer Implementation
  - [ ] 5.1 Implement adapters/data/csv_repository.py with CSVRepository class
    - Define LEAKAGE_COLUMNS constant as frozenset: 'discharge_status', 'post_discharge_medications', 'readmission_occurred', 'days_until_readmission', 'discharge_disposition', 'follow_up_scheduled', 'readmission_date'
    - Create CSVRepository class implementing PatientDataRepository port
    - Implement __init__(file_path: str) with file existence validation
    - Implement get_admissions(start_date, end_date) method
    - Load CSV using pandas and validate no leakage columns present
    - Raise DataLeakageError with column names if leakage detected
    - Convert DataFrame rows to Admission domain objects
    - Apply date range filters if specified
    - Include comprehensive Google-style docstrings
    - Include type hints showing implementation of PatientDataRepository port
    - _Requirements: 12.1, 12.4, 12.5, 16.1, 16.2, 16.3, 16.5_

  - [ ]* 5.2 Write unit tests for CSVRepository
    - Test CSVRepository raises DataLeakageError when CSV contains discharge_status column
    - Test CSVRepository raises DataLeakageError when CSV contains readmission_occurred column
    - Test CSVRepository successfully loads CSV with only pre-discharge features
    - Test CSVRepository applies date range filters correctly
    - Test CSVRepository raises InvalidAdmissionDataError for invalid data
    - _Requirements: 16.2, 16.3_

  - [ ] 5.3 Implement adapters/ml/xgboost_predictor.py with XGBoostPredictor class
    - Create XGBoostPredictor class implementing RiskPredictorPort
    - Define HIGH_RISK_THRESHOLD = 0.75 and MEDIUM_RISK_THRESHOLD = 0.40 as class constants
    - Implement __init__(model_version: str) to initialize model version
    - Implement predict_readmission_risk(admission: Admission) -> RiskAssessment method
    - Extract pre-discharge features from Admission object
    - Generate probability prediction using XGBoost model
    - Apply thresholds: 'High Risk' if >= 0.75, 'Medium Risk' if >= 0.40, 'Low Risk' otherwise
    - Return RiskAssessment domain object with risk_score, risk_category, assessment_timestamp, model_version
    - Implement train(admissions: list[Admission], outcomes: list[bool]) method
    - Extract feature matrix from Admission objects
    - Configure XGBoost with objective='binary:logistic', eval_metric='aucpr', max_depth=6, learning_rate=0.1, n_estimators=100
    - Train model and store feature names for validation
    - Include comprehensive Google-style docstrings
    - Include type hints showing implementation of RiskPredictorPort
    - _Requirements: 12.2, 12.4, 12.5, 16.7_

  - [ ]* 5.4 Write unit tests for XGBoostPredictor
    - Test XGBoostPredictor raises InvalidRiskAssessmentError when predict called before train
    - Test XGBoostPredictor categorizes risk_score=0.80 as 'High Risk'
    - Test XGBoostPredictor categorizes risk_score=0.50 as 'Medium Risk'
    - Test XGBoostPredictor categorizes risk_score=0.30 as 'Low Risk'
    - Test XGBoostPredictor train method successfully trains on valid data
    - _Requirements: 16.7_

  - [ ] 5.5 Create adapters/visualization/__init__.py placeholder
    - Create empty __init__.py file with docstring indicating future Streamlit dashboard location
    - Include TODO comment for Phase 5 implementation
    - _Requirements: 12.3_

- [ ] 6. Phase 4: Application Layer Implementation
  - [ ] 6.1 Implement application/use_cases.py with train_model use case
    - Implement train_model(data_repository, predictor, start_date, end_date) -> dict[str, float] function
    - Load historical admissions from data_repository.get_admissions()
    - Load corresponding readmission outcomes
    - Call predictor.train(admissions, outcomes)
    - Evaluate model performance and return metrics dictionary
    - Include comprehensive Google-style docstrings with Args, Returns, Raises sections
    - Include type hints for all parameters and return value
    - _Requirements: 13.1, 13.2, 13.3_

  - [ ] 6.2 Implement application/use_cases.py with predict_readmission_risk use case
    - Implement predict_readmission_risk(admission, predictor) -> RiskAssessment function
    - Validate admission data
    - Call predictor.predict_readmission_risk(admission)
    - Optionally generate baseline risk assessment for comparison logging
    - Return RiskAssessment domain object
    - Include comprehensive Google-style docstrings with Args, Returns, Raises sections
    - Include type hints for all parameters and return value
    - _Requirements: 13.1, 13.2, 13.3_

  - [ ]* 6.3 Write unit tests for application use cases
    - Test train_model successfully orchestrates training workflow
    - Test train_model raises DataLeakageError when training data contains leakage
    - Test predict_readmission_risk returns valid RiskAssessment
    - Test predict_readmission_risk raises InvalidAdmissionDataError for invalid admission
    - _Requirements: 13.1_

  - [ ]* 6.4 Write integration tests for end-to-end workflows
    - Test end-to-end training workflow: load data, train model, evaluate metrics
    - Test end-to-end prediction workflow: create admission, predict risk, validate result
    - Test integration between CSVRepository and domain models
    - Test integration between XGBoostPredictor and domain models
    - _Requirements: 13.1_

- [ ] 7. Checkpoint - Ensure all implementation tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Phase 5: Documentation and Audit Trail
  - [ ] 8.1 Create LOCAL_PROJECT_STORY.md with Director's Cut narrative
    - Include section: The Expert Architect's Opening with project vision and motivation
    - Include section: The Interconnected Team with stakeholder analysis (clinicians, data scientists, patients)
    - Include section: Phase 0-5 Development Roadmap with detailed implementation timeline
    - Include section: UBC MDS Alignment Table mapping to data science competencies
    - Include section: What We Settled On with architectural decision records
    - Include section: Technology Decisions with rationale for each technology choice
    - Include section: What We Decided NOT to Use with rejected alternatives and reasoning
    - Adapt narrative content to healthcare/patient readmission context
    - Preserve exact markdown formatting, table structures, and section hierarchy from golden template
    - Follow writing style: no em dashes, no contractions, short paragraphs, no hyperbole
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ] 8.2 Create notebooks/eda_initial.ipynb template
    - Create markdown cells for: Data Loading, Data Quality Assessment, Target Variable Analysis, Feature Distribution Analysis, Statistical Tests
    - Include code cell templates for loading data via CSVRepository adapter
    - Include markdown guidance on avoiding data leakage (excluding post-discharge features)
    - Include placeholder cells for chi-squared tests and clustering analysis
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

  - [ ] 8.3 Update README.md with comprehensive setup instructions
    - Add detailed setup instructions: conda env create, conda activate, pre-commit install
    - Add testing instructions: pytest commands, coverage reports, property test execution
    - Add development workflow: Black, isort, Mypy, Ruff commands
    - Add architecture diagram with dependency flow visualization
    - Add technology stack table with versions and rationale
    - _Requirements: 11.2, 11.3, 11.4_

  - [ ] 8.4 Create GitHub Actions CI pipeline configuration
    - Create .github/workflows/ci.yml file
    - Configure quality job: Black check, isort check, Ruff check, Mypy check
    - Configure test job: unit tests with coverage, property tests with Hypothesis statistics, architecture tests
    - Configure to run on push to main/develop branches and pull requests
    - Include coverage upload to Codecov
    - _Requirements: Design section on Continuous Integration_

- [ ] 9. Final checkpoint - Verify all requirements satisfied
  - Run full test suite with coverage report
  - Verify 90%+ overall coverage, 95%+ domain layer coverage
  - Run all property tests with Hypothesis statistics
  - Run architecture compliance tests
  - Verify pre-commit hooks pass on all files
  - Verify Mypy --strict passes with zero errors
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability to requirements.md
- Property tests validate universal correctness properties from design.md
- Checkpoints ensure incremental validation at phase boundaries
- All code must pass Mypy --strict type checking before moving to next phase
- Data leakage prevention is enforced at adapter boundaries with explicit validation
- High precision threshold (0.75) minimizes clinical alarm fatigue
- Hexagonal Architecture ensures domain logic remains independent of external frameworks
