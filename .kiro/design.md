# Design Document: Patient Readmission Risk Engine

## Overview

The Patient Readmission Risk Engine is a healthcare ML system that predicts 30-day hospital readmission risk using pre-discharge clinical features. The system implements Hexagonal Architecture (Ports and Adapters pattern) to ensure domain logic remains independent of external frameworks, enabling testability, maintainability, and regulatory compliance.

### Core Objectives

- Predict readmission risk using only pre-discharge features to prevent data leakage
- Maintain high precision (0.75 threshold) to minimize clinical alarm fatigue
- Provide explainable predictions for clinical decision support
- Enable adapter swapping (CSV → database, XGBoost → other models) without domain changes
- Enforce industrial-grade code quality through Python 3.12 strict typing and property-based testing

### System Context

The system operates in a clinical workflow where healthcare providers need readmission risk assessments before patient discharge. The model must never use post-discharge information (discharge status, post-discharge medications, actual readmission outcomes) during training or prediction to ensure real-world validity.

## Architecture

### Hexagonal Architecture Pattern

The system follows the Ports and Adapters pattern with strict inward-pointing dependencies:

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│                   (Use Case Orchestration)                   │
│                                                              │
│  train_model()           predict_readmission_risk()         │
└──────────────────┬───────────────────┬──────────────────────┘
                   │                   │
                   ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                      Domain Layer                            │
│                  (Pure Business Logic)                       │
│                                                              │
│  Models: Patient, Admission, RiskAssessment                 │
│  Ports: PatientDataRepository, RiskPredictorPort            │
│  Services: baseline_readmission_risk_flag()                 │
│  Exceptions: DomainError, DataLeakageError                  │
└──────────────────┬───────────────────┬──────────────────────┘
                   ▲                   ▲
                   │                   │
┌──────────────────┴───────────────────┴──────────────────────┐
│                     Adapter Layer                            │
│              (External System Connections)                   │
│                                                              │
│  Data: CSVRepository (implements PatientDataRepository)     │
│  ML: XGBoostPredictor (implements RiskPredictorPort)        │
│  Visualization: Streamlit dashboards (future)               │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Rules

1. Domain layer has ZERO imports from adapters or application layers
2. Domain layer imports only: typing, dataclasses, datetime, enum, and other domain modules
3. Adapter layer imports from domain layer to implement port interfaces
4. Application layer imports from both domain and adapters for orchestration
5. Tests can import from all layers for integration testing

### Benefits for Healthcare Context

- **Regulatory Compliance**: Domain logic is testable without external dependencies
- **Adapter Swapping**: Replace CSV with FHIR API without touching domain code
- **Model Swapping**: Replace XGBoost with neural networks without touching domain code
- **Audit Trail**: Pure Python domain logic is human-readable for clinical validation
- **Leakage Prevention**: Data validation logic lives in adapters, enforced at system boundaries

## Components and Interfaces

### Domain Layer Components

#### 1. Domain Models (domain/models.py)

**Patient Dataclass**
```python
@dataclass(frozen=True)
class Patient:
    """Immutable value object representing a hospital patient.
    
    Attributes:
        patient_id: Unique identifier for the patient
        age: Patient age in years (must be >= 0)
        gender: Patient gender ('M', 'F', 'Other')
        insurance_type: Insurance category ('Medicare', 'Medicaid', 'Private', 'Uninsured')
    """
    patient_id: str
    age: int
    gender: str
    insurance_type: str
    
    def __post_init__(self) -> None:
        """Validate business invariants."""
        if self.age < 0:
            raise InvalidAdmissionDataError(f"Age must be non-negative, got {self.age}")
        if self.gender not in {'M', 'F', 'Other'}:
            raise InvalidAdmissionDataError(f"Invalid gender: {self.gender}")
```

**Admission Dataclass**
```python
@dataclass(frozen=True)
class Admission:
    """Immutable value object representing a hospital admission with pre-discharge features only.
    
    Attributes:
        admission_id: Unique identifier for this admission
        patient: The patient being admitted
        admission_date: Date and time of admission
        admission_type: Type of admission ('Emergency', 'Elective', 'Urgent')
        primary_diagnosis: ICD-10 code for primary diagnosis
        comorbidity_count: Number of documented comorbidities (Charlson index)
        length_of_stay_scheduled: Planned length of stay in days
        prior_admissions_count: Number of admissions in past 12 months
        lab_abnormalities_count: Number of abnormal lab results during stay
        medication_count: Number of medications prescribed during stay
    """
    admission_id: str
    patient: Patient
    admission_date: datetime
    admission_type: str
    primary_diagnosis: str
    comorbidity_count: int
    length_of_stay_scheduled: int
    prior_admissions_count: int
    lab_abnormalities_count: int
    medication_count: int
    
    def __post_init__(self) -> None:
        """Validate business invariants."""
        if self.admission_type not in {'Emergency', 'Elective', 'Urgent'}:
            raise InvalidAdmissionDataError(f"Invalid admission type: {self.admission_type}")
        if self.comorbidity_count < 0:
            raise InvalidAdmissionDataError("Comorbidity count must be non-negative")
        if self.length_of_stay_scheduled < 0:
            raise InvalidAdmissionDataError("Length of stay must be non-negative")
```

**RiskAssessment Dataclass**
```python
@dataclass(frozen=True)
class RiskAssessment:
    """Immutable value object representing a readmission risk prediction.
    
    Attributes:
        admission_id: Reference to the admission being assessed
        risk_score: Predicted probability of 30-day readmission (0.0 to 1.0)
        risk_category: Human-readable category ('Low Risk', 'Medium Risk', 'High Risk')
        assessment_timestamp: When this assessment was generated
        model_version: Version identifier of the prediction model used
        explanation: Optional SHAP-based feature importance explanation
    """
    admission_id: str
    risk_score: float
    risk_category: str
    assessment_timestamp: datetime
    model_version: str
    explanation: Optional[dict[str, float]] = None
    
    def __post_init__(self) -> None:
        """Validate business invariants."""
        if not 0.0 <= self.risk_score <= 1.0:
            raise InvalidRiskAssessmentError(f"Risk score must be in [0, 1], got {self.risk_score}")
        if self.risk_category not in {'Low Risk', 'Medium Risk', 'High Risk'}:
            raise InvalidRiskAssessmentError(f"Invalid risk category: {self.risk_category}")
```

#### 2. Domain Ports (domain/ports.py)

**PatientDataRepository Protocol**
```python
class PatientDataRepository(Protocol):
    """Port interface for loading patient admission data from external sources.
    
    This protocol defines the contract that data adapters must implement.
    Implementations must enforce leakage prevention by excluding post-discharge features.
    """
    
    def get_admissions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[Admission]:
        """Load admission records within the specified date range.
        
        Args:
            start_date: Optional filter for admissions on or after this date
            end_date: Optional filter for admissions before this date
            
        Returns:
            List of Admission objects with only pre-discharge features
            
        Raises:
            DataLeakageError: If any post-discharge features are detected in source data
            InvalidAdmissionDataError: If data fails validation
        """
        ...
```

**RiskPredictorPort Protocol**
```python
class RiskPredictorPort(Protocol):
    """Port interface for ML models that predict readmission risk.
    
    This protocol defines the contract that ML adapters must implement.
    Implementations must use only pre-discharge features from Admission objects.
    """
    
    def predict_readmission_risk(self, admission: Admission) -> RiskAssessment:
        """Predict 30-day readmission risk for a given admission.
        
        Args:
            admission: Admission object containing pre-discharge features only
            
        Returns:
            RiskAssessment with risk score, category, and optional explanation
            
        Raises:
            InvalidAdmissionDataError: If admission data is invalid
        """
        ...
    
    def train(self, admissions: list[Admission], outcomes: list[bool]) -> None:
        """Train the model on historical admission data.
        
        Args:
            admissions: List of historical admissions with pre-discharge features
            outcomes: List of boolean outcomes (True = readmitted within 30 days)
            
        Raises:
            DataLeakageError: If training data contains post-discharge features
        """
        ...
```

#### 3. Domain Services (domain/services.py)

**Baseline Risk Assessment Function**
```python
# High-risk diagnosis codes (ICD-10 codes associated with readmission)
HIGH_RISK_CONDITIONS: frozenset[str] = frozenset({
    'I50',    # Heart failure
    'J44',    # COPD
    'E11',    # Type 2 diabetes with complications
    'N18',    # Chronic kidney disease
    'I25',    # Chronic ischemic heart disease
    'J18',    # Pneumonia
    'I63',    # Cerebral infarction
    'K70',    # Alcoholic liver disease
})

def baseline_readmission_risk_flag(admission: Admission) -> str:
    """Rule-based baseline risk assessment using clinical heuristics.
    
    This function provides a non-ML baseline for comparison and fallback.
    It uses only pre-discharge features available at admission time.
    
    Args:
        admission: Admission object with pre-discharge features
        
    Returns:
        Risk category: 'Low Risk', 'Medium Risk', or 'High Risk'
        
    Example:
        >>> patient = Patient('P001', 75, 'M', 'Medicare')
        >>> admission = Admission(
        ...     'A001', patient, datetime.now(), 'Emergency',
        ...     'I50', comorbidity_count=4, length_of_stay_scheduled=7,
        ...     prior_admissions_count=2, lab_abnormalities_count=3,
        ...     medication_count=8
        ... )
        >>> baseline_readmission_risk_flag(admission)
        'High Risk'
    """
    # Extract primary diagnosis prefix (first 3 characters of ICD-10)
    diagnosis_prefix = admission.primary_diagnosis[:3]
    
    # High risk criteria
    high_risk_conditions = [
        diagnosis_prefix in HIGH_RISK_CONDITIONS,
        admission.comorbidity_count >= 3,
        admission.prior_admissions_count >= 2,
        admission.admission_type == 'Emergency',
        admission.length_of_stay_scheduled >= 7,
    ]
    
    # Count how many high-risk factors are present
    risk_factor_count = sum(high_risk_conditions)
    
    if risk_factor_count >= 3:
        return 'High Risk'
    elif risk_factor_count >= 1:
        return 'Medium Risk'
    else:
        return 'Low Risk'
```

#### 4. Domain Exceptions (domain/exceptions.py)

```python
class DomainError(Exception):
    """Base exception for all domain-level errors."""
    pass

class InvalidAdmissionDataError(DomainError):
    """Raised when admission data violates business invariants.
    
    Examples:
        - Negative age or comorbidity count
        - Invalid admission type or gender code
        - Missing required fields
    """
    pass

class InvalidRiskAssessmentError(DomainError):
    """Raised when risk assessment data is invalid.
    
    Examples:
        - Risk score outside [0, 1] range
        - Invalid risk category
        - Missing required fields
    """
    pass

class DataLeakageError(DomainError):
    """Raised when post-discharge features are detected in training or prediction data.
    
    This exception prevents models from learning spurious patterns that would fail
    in production where post-discharge information is unavailable.
    
    Examples:
        - discharge_status column present in training data
        - readmission_occurred column present in feature matrix
        - days_until_readmission column present in prediction input
    """
    pass
```

### Adapter Layer Components

#### 1. CSV Data Adapter (adapters/data/csv_repository.py)

**CSVRepository Class**
```python
# Post-discharge features that must be excluded to prevent leakage
LEAKAGE_COLUMNS: frozenset[str] = frozenset({
    'discharge_status',
    'post_discharge_medications',
    'readmission_occurred',
    'days_until_readmission',
    'discharge_disposition',
    'follow_up_scheduled',
    'readmission_date',
})

class CSVRepository:
    """Adapter for loading patient admission data from CSV files.
    
    Implements PatientDataRepository port with leakage prevention validation.
    """
    
    def __init__(self, file_path: str) -> None:
        """Initialize repository with CSV file path.
        
        Args:
            file_path: Path to CSV file containing admission data
        """
        self.file_path = file_path
        self._validate_file_exists()
    
    def get_admissions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list[Admission]:
        """Load admissions from CSV with leakage validation.
        
        Implementation:
        1. Load CSV using pandas
        2. Validate no leakage columns present
        3. Filter by date range if specified
        4. Convert rows to Admission domain objects
        5. Validate each Admission object
        
        Raises:
            DataLeakageError: If any column in LEAKAGE_COLUMNS is present
            InvalidAdmissionDataError: If data fails domain validation
        """
        df = pd.read_csv(self.file_path)
        
        # Leakage prevention check
        detected_leakage = set(df.columns) & LEAKAGE_COLUMNS
        if detected_leakage:
            raise DataLeakageError(
                f"Post-discharge features detected: {detected_leakage}. "
                f"These columns must be removed to prevent data leakage."
            )
        
        # Convert to domain objects (implementation details omitted)
        admissions = self._convert_to_admissions(df)
        
        # Apply date filters if specified
        if start_date:
            admissions = [a for a in admissions if a.admission_date >= start_date]
        if end_date:
            admissions = [a for a in admissions if a.admission_date < end_date]
        
        return admissions
```

#### 2. XGBoost ML Adapter (adapters/ml/xgboost_predictor.py)

**XGBoostPredictor Class**
```python
class XGBoostPredictor:
    """Adapter for XGBoost-based readmission risk prediction.
    
    Implements RiskPredictorPort with high-precision threshold tuning.
    """
    
    # High precision threshold to minimize false alarms in clinical setting
    HIGH_RISK_THRESHOLD: float = 0.75
    MEDIUM_RISK_THRESHOLD: float = 0.40
    
    def __init__(self, model_version: str = "v1.0") -> None:
        """Initialize predictor with model version identifier."""
        self.model_version = model_version
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: list[str] = []
    
    def predict_readmission_risk(self, admission: Admission) -> RiskAssessment:
        """Predict readmission risk using trained XGBoost model.
        
        Implementation:
        1. Extract pre-discharge features from Admission object
        2. Validate features match training schema
        3. Generate probability prediction
        4. Apply threshold to determine risk category
        5. Generate SHAP explanation (optional)
        6. Return RiskAssessment domain object
        """
        if self.model is None:
            raise InvalidRiskAssessmentError("Model not trained. Call train() first.")
        
        # Extract features (only pre-discharge)
        features = self._extract_features(admission)
        
        # Predict probability
        risk_score = self.model.predict_proba(features)[0, 1]
        
        # Determine category based on clinical thresholds
        if risk_score >= self.HIGH_RISK_THRESHOLD:
            risk_category = 'High Risk'
        elif risk_score >= self.MEDIUM_RISK_THRESHOLD:
            risk_category = 'Medium Risk'
        else:
            risk_category = 'Low Risk'
        
        return RiskAssessment(
            admission_id=admission.admission_id,
            risk_score=float(risk_score),
            risk_category=risk_category,
            assessment_timestamp=datetime.now(),
            model_version=self.model_version,
        )
    
    def train(self, admissions: list[Admission], outcomes: list[bool]) -> None:
        """Train XGBoost model on historical data.
        
        Implementation:
        1. Extract feature matrix from Admission objects
        2. Validate no leakage in features
        3. Configure XGBoost hyperparameters
        4. Train with cross-validation
        5. Log metrics to MLflow
        """
        # Extract features and validate
        X = self._extract_feature_matrix(admissions)
        y = np.array(outcomes)
        
        # Configure model for high precision
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',  # Precision-recall AUC
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
        )
        
        # Train model
        self.model.fit(X, y)
        
        # Store feature names for validation
        self.feature_names = list(X.columns)
    
    def _extract_features(self, admission: Admission) -> pd.DataFrame:
        """Extract pre-discharge features from Admission object.
        
        Features extracted:
        - patient.age
        - patient.gender (one-hot encoded)
        - patient.insurance_type (one-hot encoded)
        - admission_type (one-hot encoded)
        - primary_diagnosis (encoded as diagnosis category)
        - comorbidity_count
        - length_of_stay_scheduled
        - prior_admissions_count
        - lab_abnormalities_count
        - medication_count
        """
        # Implementation details omitted for brevity
        pass
```

### Application Layer Components

#### Use Cases (application/use_cases.py)

**train_model() Use Case**
```python
def train_model(
    data_repository: PatientDataRepository,
    predictor: RiskPredictorPort,
    start_date: datetime,
    end_date: datetime,
) -> dict[str, float]:
    """Orchestrate model training workflow.
    
    Workflow:
    1. Load historical admissions from repository
    2. Load corresponding readmission outcomes
    3. Train predictor using domain objects
    4. Evaluate model performance
    5. Log metrics to MLflow
    6. Return performance metrics
    
    Args:
        data_repository: Data source implementing PatientDataRepository
        predictor: ML model implementing RiskPredictorPort
        start_date: Start of training data period
        end_date: End of training data period
        
    Returns:
        Dictionary of performance metrics (precision, recall, AUC-PR)
        
    Raises:
        DataLeakageError: If training data contains post-discharge features
    """
    # Load training data
    admissions = data_repository.get_admissions(start_date, end_date)
    
    # Load outcomes (implementation depends on data source)
    outcomes = data_repository.get_outcomes([a.admission_id for a in admissions])
    
    # Train model
    predictor.train(admissions, outcomes)
    
    # Evaluate and return metrics
    metrics = evaluate_model(predictor, admissions, outcomes)
    return metrics
```

**predict_readmission_risk() Use Case**
```python
def predict_readmission_risk(
    admission: Admission,
    predictor: RiskPredictorPort,
) -> RiskAssessment:
    """Orchestrate risk prediction workflow.
    
    Workflow:
    1. Validate admission data
    2. Generate ML-based risk assessment
    3. Generate baseline risk assessment for comparison
    4. Return risk assessment
    
    Args:
        admission: Admission object with pre-discharge features
        predictor: Trained ML model implementing RiskPredictorPort
        
    Returns:
        RiskAssessment domain object
        
    Raises:
        InvalidAdmissionDataError: If admission data is invalid
    """
    # Generate ML prediction
    ml_assessment = predictor.predict_readmission_risk(admission)
    
    # Generate baseline for comparison (optional logging)
    baseline_category = baseline_readmission_risk_flag(admission)
    
    return ml_assessment
```

## Data Models

### Patient Entity

Represents a hospital patient with demographic information.

Fields:
- patient_id (str): Unique identifier
- age (int): Age in years, must be >= 0
- gender (str): 'M', 'F', or 'Other'
- insurance_type (str): 'Medicare', 'Medicaid', 'Private', or 'Uninsured'

Invariants:
- Age must be non-negative
- Gender must be one of the valid codes
- All fields are immutable (frozen dataclass)

### Admission Entity

Represents a hospital admission with pre-discharge features only.

Fields:
- admission_id (str): Unique identifier
- patient (Patient): Reference to patient entity
- admission_date (datetime): Date and time of admission
- admission_type (str): 'Emergency', 'Elective', or 'Urgent'
- primary_diagnosis (str): ICD-10 code
- comorbidity_count (int): Number of comorbidities (Charlson index)
- length_of_stay_scheduled (int): Planned length of stay in days
- prior_admissions_count (int): Admissions in past 12 months
- lab_abnormalities_count (int): Number of abnormal lab results
- medication_count (int): Number of medications prescribed

Invariants:
- Admission type must be valid
- All counts must be non-negative
- All fields are immutable (frozen dataclass)

Leakage Prevention:
- NO discharge_status field
- NO post_discharge_medications field
- NO readmission_occurred field
- NO days_until_readmission field

### RiskAssessment Entity

Represents a readmission risk prediction result.

Fields:
- admission_id (str): Reference to admission
- risk_score (float): Probability in [0.0, 1.0]
- risk_category (str): 'Low Risk', 'Medium Risk', or 'High Risk'
- assessment_timestamp (datetime): When assessment was generated
- model_version (str): Model version identifier
- explanation (Optional[dict[str, float]]): SHAP feature importance values

Invariants:
- Risk score must be in [0, 1]
- Risk category must be valid
- All fields are immutable (frozen dataclass)

Thresholds:
- High Risk: risk_score >= 0.75 (high precision to minimize alarm fatigue)
- Medium Risk: 0.40 <= risk_score < 0.75
- Low Risk: risk_score < 0.40


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property Reflection

After analyzing all 70+ acceptance criteria, I identified the following consolidations to eliminate redundancy:

- Multiple criteria checking specific configuration values (pyproject.toml, environment.yml, .pre-commit-config.yaml) are examples, not properties
- Multiple criteria checking file/directory existence are examples, not properties
- Criteria about "all dataclasses must have X" and "all functions must have Y" can be combined into comprehensive properties
- Architecture compliance checks (1.7, 14.1, 14.2) are related and can be consolidated
- Leakage prevention checks (16.1, 16.2, 16.3) are related and can be consolidated
- Docstring requirements across different modules (5.3, 6.4, 7.3, 8.4, 12.5, 13.3) can be consolidated

The following properties represent the unique, testable invariants that provide comprehensive validation coverage:

### Property 1: Hexagonal Architecture Dependency Inversion

For any Python file in the domain/ directory, all import statements must reference only: typing, dataclasses, datetime, enum, or other domain/ modules. No imports from adapters/, application/, pandas, numpy, scikit-learn, xgboost, or any external ML/data libraries are permitted.

**Validates: Requirements 1.7, 5.6, 7.4, 14.1**

### Property 2: Adapter-Domain Dependency Flow

For any Python file in the adapters/ directory, imports from domain/ are permitted, but for any Python file in the domain/ directory, imports from adapters/ are forbidden. This ensures unidirectional dependency flow.

**Validates: Requirements 14.2**

### Property 3: Immutable Domain Models

For any dataclass defined in domain/models.py, the frozen=True parameter must be set to ensure immutability of value objects.

**Validates: Requirements 5.2**

### Property 4: Type-Annotated Domain Fields

For any field in any dataclass in domain/models.py, a type annotation must be present (str, int, float, datetime, Optional, or other valid types).

**Validates: Requirements 5.4**

### Property 5: Protocol-Based Port Definitions

For any class defined in domain/ports.py, it must inherit from typing.Protocol to ensure proper port interface definition.

**Validates: Requirements 6.5**

### Property 6: Comprehensive Docstring Coverage

For any public class, function, or method in domain/, adapters/, or application/ layers, a docstring must be present explaining its purpose and contract.

**Validates: Requirements 5.3, 6.4, 7.3, 8.4, 12.5, 13.3**

### Property 7: Type-Annotated Use Case Functions

For any function defined in application/use_cases.py, all parameters and the return value must have type annotations.

**Validates: Requirements 13.2**

### Property 8: Type-Annotated Adapter Implementations

For any class defined in adapters/ that implements a domain port, type hints must indicate which port protocol is being implemented (either through inheritance or structural typing).

**Validates: Requirements 12.4**

### Property 9: Required Test Imports

For any Python file in tests/ directory, import statements for pytest must be present to enable test discovery and execution.

**Validates: Requirements 9.5**

### Property 10: Leakage Column Exclusion

For any DataFrame or feature matrix loaded by CSVRepository.get_admissions(), none of the columns in LEAKAGE_COLUMNS (discharge_status, post_discharge_medications, readmission_occurred, days_until_readmission) may be present. If detected, a DataLeakageError must be raised with the offending column names.

**Validates: Requirements 16.1, 16.2, 16.3**

### Property 11: High-Risk Threshold Precision

For any risk score value in the range [0.0, 1.0], the risk categorization logic must classify it as 'High Risk' if and only if the score is >= 0.75, 'Medium Risk' if >= 0.40 and < 0.75, and 'Low Risk' if < 0.40.

**Validates: Requirements 16.7**

### Property 12: Risk Score Bounds Validation

For any RiskAssessment object created, the risk_score field must be in the range [0.0, 1.0], otherwise an InvalidRiskAssessmentError must be raised during __post_init__ validation.

**Validates: Requirements 5.5 (implicit from RiskAssessment validation)**

### Property 13: Pre-Commit Hook Repository URLs

For any hook defined in .pre-commit-config.yaml, the repository URL must match the official repository for that tool (e.g., black must use https://github.com/psf/black).

**Validates: Requirements 3.6**

### Property 14: Baseline Risk Assessment Feature Restriction

For any Admission object passed to baseline_readmission_risk_flag(), the function must compute risk category using only the pre-discharge features: admission_type, primary_diagnosis, comorbidity_count, length_of_stay_scheduled, and prior_admissions_count. No post-discharge features may be accessed.

**Validates: Requirements 7.2, 16.4**

### Property 15: Domain Exception Inheritance

For any exception class defined in domain/exceptions.py (except DomainError itself), it must inherit from DomainError to maintain exception hierarchy.

**Validates: Requirements 8.2, 8.3**

## Error Handling

### Domain-Level Error Handling

The domain layer defines a hierarchy of exceptions that represent business rule violations:

```
DomainError (base)
├── InvalidAdmissionDataError
│   ├── Negative age or counts
│   ├── Invalid admission type
│   ├── Invalid gender code
│   └── Missing required fields
├── InvalidRiskAssessmentError
│   ├── Risk score out of [0, 1] range
│   ├── Invalid risk category
│   └── Missing required fields
└── DataLeakageError
    ├── Post-discharge columns in training data
    ├── Post-discharge columns in prediction data
    └── Leakage columns in feature matrix
```

### Error Handling Strategy by Layer

**Domain Layer:**
- Raise domain exceptions immediately when invariants are violated
- Use __post_init__ validation in dataclasses for eager validation
- Provide descriptive error messages with actual vs expected values

**Adapter Layer:**
- Catch external library exceptions (pandas, xgboost) and translate to domain exceptions
- Validate data at system boundaries before converting to domain objects
- Log detailed error context for debugging while raising clean domain exceptions

**Application Layer:**
- Catch domain exceptions and translate to user-facing error messages
- Log exceptions with full context for audit trail
- Return error responses with appropriate HTTP status codes (if web API)

### Leakage Prevention Error Handling

The CSVRepository adapter implements defensive validation:

```python
def get_admissions(self, ...) -> list[Admission]:
    df = pd.read_csv(self.file_path)
    
    # Leakage detection
    detected_leakage = set(df.columns) & LEAKAGE_COLUMNS
    if detected_leakage:
        raise DataLeakageError(
            f"Post-discharge features detected: {sorted(detected_leakage)}. "
            f"These columns must be removed to prevent data leakage. "
            f"Allowed features: {sorted(ALLOWED_FEATURES)}"
        )
    
    # Continue with safe data processing...
```

### Model Prediction Error Handling

The XGBoostPredictor adapter handles model errors gracefully:

```python
def predict_readmission_risk(self, admission: Admission) -> RiskAssessment:
    if self.model is None:
        raise InvalidRiskAssessmentError(
            "Model not trained. Call train() before prediction."
        )
    
    try:
        features = self._extract_features(admission)
        risk_score = self.model.predict_proba(features)[0, 1]
    except Exception as e:
        raise InvalidRiskAssessmentError(
            f"Model prediction failed for admission {admission.admission_id}: {e}"
        ) from e
    
    # Continue with risk categorization...
```

## Testing Strategy

### Dual Testing Approach

The system employs both unit testing and property-based testing for comprehensive coverage:

**Unit Tests:**
- Specific examples demonstrating correct behavior
- Edge cases (empty data, boundary values, invalid inputs)
- Integration points between layers
- Error condition handling

**Property Tests:**
- Universal properties that hold for all inputs
- Comprehensive input coverage through randomization
- Invariant validation across generated test cases
- Minimum 100 iterations per property test

### Property-Based Testing with Hypothesis

The project uses the Hypothesis library for property-based testing in Python:

**Configuration:**
```python
# tests/test_properties.py
from hypothesis import given, settings, strategies as st

@settings(max_examples=100)  # Minimum 100 iterations
@given(
    age=st.integers(min_value=0, max_value=120),
    gender=st.sampled_from(['M', 'F', 'Other']),
    insurance=st.sampled_from(['Medicare', 'Medicaid', 'Private', 'Uninsured'])
)
def test_property_patient_creation(age, gender, insurance):
    """Feature: patient-readmission-risk-engine, Property 3: Immutable Domain Models
    
    For any valid patient data, creating a Patient object should succeed
    and the object should be immutable (frozen).
    """
    patient = Patient(f"P{age}", age, gender, insurance)
    
    # Verify immutability
    with pytest.raises(FrozenInstanceError):
        patient.age = 99
```

### Test Organization

**tests/test_domain_models.py:**
- Unit tests for Patient, Admission, RiskAssessment validation
- Edge cases: negative values, invalid codes, boundary conditions
- Example: test_patient_negative_age_raises_error()

**tests/test_domain_services.py:**
- Unit tests for baseline_readmission_risk_flag()
- Edge cases: empty diagnosis, zero comorbidities
- Example: test_baseline_high_risk_emergency_admission()

**tests/test_properties.py:**
- Property tests for all 15 correctness properties
- Hypothesis strategies for generating test data
- Example: test_property_hexagonal_architecture_dependency_inversion()

**tests/test_adapters.py:**
- Unit tests for CSVRepository and XGBoostPredictor
- Leakage detection tests with mock data
- Example: test_csv_repository_raises_leakage_error()

**tests/test_architecture.py:**
- Architecture compliance tests
- Import graph validation
- Example: test_domain_has_no_adapter_imports()

### Property Test Tagging

Each property test must include a comment tag referencing the design document:

```python
@settings(max_examples=100)
@given(csv_data=st.data())
def test_property_leakage_column_exclusion(csv_data):
    """Feature: patient-readmission-risk-engine, Property 10: Leakage Column Exclusion
    
    For any DataFrame loaded by CSVRepository, if it contains leakage columns,
    a DataLeakageError must be raised.
    """
    # Test implementation...
```

### Coverage Targets

- Overall code coverage: 90%+
- Domain layer coverage: 95%+ (critical business logic)
- Adapter layer coverage: 85%+ (external integrations)
- Application layer coverage: 90%+ (orchestration logic)

### Test Execution

```bash
# Run all tests with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Run only property tests
pytest tests/test_properties.py -v

# Run with Hypothesis verbosity
pytest tests/test_properties.py -v --hypothesis-show-statistics

# Run architecture compliance tests
pytest tests/test_architecture.py -v
```

### Continuous Integration

GitHub Actions workflow will run on every commit:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov hypothesis
      - name: Run pre-commit hooks
        run: pre-commit run --all-files
      - name: Run tests
        run: pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Technology Stack

### Core Technologies

**Python 3.12**
- Rationale: Latest stable version with improved type system, performance optimizations, and better error messages
- Benefits: Strict typing support, pattern matching, improved asyncio
- Trade-offs: Requires recent environment, some libraries may lag in support

**XGBoost 2.0+**
- Rationale: State-of-art gradient boosting for tabular healthcare data
- Benefits: High accuracy, built-in feature importance, handles missing values
- Trade-offs: Less interpretable than logistic regression, requires tuning

**MLflow 2.0+**
- Rationale: Experiment tracking and model registry for reproducibility
- Benefits: Version control for models, parameter tracking, deployment support
- Trade-offs: Adds infrastructure complexity, requires storage backend

**Hypothesis 6.0+**
- Rationale: Property-based testing for comprehensive edge case coverage
- Benefits: Automatic test case generation, shrinking for minimal failing examples
- Trade-offs: Steeper learning curve than traditional unit testing

**SHAP 0.44+**
- Rationale: Model explainability for clinical decision support
- Benefits: Theoretically grounded feature importance, local and global explanations
- Trade-offs: Computationally expensive for large datasets

### Development Tools

**Black 24.0+**
- Rationale: Opinionated code formatter for consistency
- Configuration: line-length=88, target-version=py312
- Benefits: Zero configuration debates, automatic formatting

**Mypy 1.8+**
- Rationale: Static type checking for early error detection
- Configuration: --strict mode for maximum safety
- Benefits: Catches type errors before runtime, improves IDE support

**Ruff 0.1+**
- Rationale: Fast Python linter (10-100x faster than Flake8)
- Benefits: Combines multiple tools (Flake8, isort, pyupgrade)
- Trade-offs: Newer tool, less mature than alternatives

**isort 5.13+**
- Rationale: Automatic import sorting for consistency
- Configuration: profile="black" for compatibility
- Benefits: Reduces merge conflicts, improves readability

**pre-commit 3.6+**
- Rationale: Automated code quality checks on every commit
- Benefits: Prevents bad code from entering repository
- Trade-offs: Adds friction to commit process (acceptable for quality)

### Data Science Tools

**Pandas 2.0+**
- Rationale: DataFrame manipulation for CSV data loading
- Benefits: Rich API, wide adoption, good performance
- Trade-offs: Only used in adapter layer, not domain

**NumPy 1.26+**
- Rationale: Numerical operations for feature engineering
- Benefits: Fast array operations, scientific computing foundation
- Trade-offs: Only used in adapter layer, not domain

**Scikit-learn 1.3+**
- Rationale: Preprocessing, metrics, and baseline models
- Benefits: Consistent API, comprehensive toolkit
- Trade-offs: Less performant than XGBoost for boosting

**Matplotlib 3.8+ / Seaborn 0.13+**
- Rationale: Visualization for EDA and model evaluation
- Benefits: Publication-quality plots, statistical visualizations
- Trade-offs: Verbose API, requires styling effort

### Infrastructure

**Conda / Mamba**
- Rationale: Environment management with binary dependencies
- Benefits: Reproducible environments, handles non-Python dependencies
- Trade-offs: Slower than pip, larger environment size

**Jupyter Lab 4.0+**
- Rationale: Interactive notebooks for EDA and experimentation
- Benefits: Inline visualizations, iterative development
- Trade-offs: Not suitable for production code

**Streamlit 1.30+ (Future)**
- Rationale: Interactive dashboards for model deployment
- Benefits: Pure Python, rapid prototyping, built-in widgets
- Trade-offs: Limited customization, not for high-scale production

## Configuration Management

### pyproject.toml Structure

The pyproject.toml file serves as the digital constitution for the project:

```toml
[project]
name = "patient_readmission_risk"
version = "0.1.0"
description = "Healthcare ML system for predicting 30-day hospital readmission risk"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "mlflow>=2.0.0",
    "hypothesis>=6.0.0",
    "pytest>=7.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest-cov>=4.0.0",
    "black>=24.0.0",
    "isort>=5.13.0",
    "mypy>=1.8.0",
    "ruff>=0.1.0",
]

[tool.mypy]
strict = true
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.black]
line-length = 88
target-version = ["py312"]
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v --cov=. --cov-report=html --cov-report=term"
```

### Pre-Commit Configuration

The .pre-commit-config.yaml enforces code quality on every commit:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: ["--strict"]
        additional_dependencies: [types-all]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.15
    hooks:
      - id: ruff
```

### Conda Environment Configuration

The environment.yml ensures reproducible Python 3.12 environment:

```yaml
name: patient-readmission-ml
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - jupyterlab>=4.0
  - pandas>=2.0.0
  - numpy>=1.26
  - scikit-learn>=1.3.0
  - matplotlib>=3.8
  - seaborn>=0.13
  - pip:
      - xgboost>=2.0.0
      - mlflow>=2.0.0
      - hypothesis>=6.0.0
      - shap>=0.44
      - streamlit>=1.30
      - black>=24.0.0
      - ruff>=0.1.0
      - mypy>=1.8.0
      - isort>=5.13
      - pytest>=7.0.0
      - pytest-cov>=4.0.0
      - pre-commit>=3.6
```

### Git Configuration

The .gitignore excludes generated files and sensitive data:

```
# Python
__pycache__/
.pytest_cache/
.hypothesis/
.mypy_cache/
*.pyc
*.pyo
*.pyd
.Python

# IDE
.vscode/
.idea/
.DS_Store

# Data (never commit raw data)
data/raw/*
data/interim/*
data/processed/*

# ML artifacts
mlruns/
models/*.pkl
models/*.joblib

# Documentation (local only)
LOCAL_PROJECT_STORY.md

# Kiro
.kiro/

# Jupyter
.ipynb_checkpoints/
```

## Deployment Considerations

### Development Phases

The project follows a phased development approach documented in LOCAL_PROJECT_STORY.md:

**Phase 0: Foundation (Current)**
- Project scaffolding with Hexagonal Architecture
- Digital constitution (pyproject.toml, pre-commit hooks)
- Domain models, ports, and services
- Testing infrastructure with Hypothesis

**Phase 1: Data Pipeline**
- CSVRepository implementation with leakage validation
- EDA notebook with statistical analysis
- Data quality checks and validation
- Feature engineering pipeline

**Phase 2: Baseline Model**
- Implement baseline_readmission_risk_flag() logic
- Unit tests for baseline model
- Performance benchmarking
- Documentation of baseline approach

**Phase 3: ML Model Training**
- XGBoostPredictor implementation
- Hyperparameter tuning with cross-validation
- MLflow experiment tracking
- Model evaluation with precision-recall curves

**Phase 4: Model Explainability**
- SHAP value computation
- Feature importance visualization
- Local explanation generation
- Clinical validation of explanations

**Phase 5: Deployment Preparation**
- Streamlit dashboard for predictions
- Model serialization and versioning
- API endpoint design (future)
- Production monitoring strategy (future)

### Local Development Workflow

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate patient-readmission-ml

# 2. Install pre-commit hooks
pre-commit install

# 3. Run tests
pytest --cov=. --cov-report=html

# 4. Check type safety
mypy domain/ adapters/ application/

# 5. Format code
black .
isort .

# 6. Run linter
ruff check .
```

### GitHub Actions CI Pipeline

The CI pipeline runs on every push and pull request:

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run Black
        run: black --check .
      
      - name: Run isort
        run: isort --check-only .
      
      - name: Run Ruff
        run: ruff check .
      
      - name: Run Mypy
        run: mypy domain/ adapters/ application/
  
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run unit tests
        run: pytest tests/ --cov=. --cov-report=xml
      
      - name: Run property tests
        run: pytest tests/test_properties.py -v --hypothesis-show-statistics
      
      - name: Run architecture tests
        run: pytest tests/test_architecture.py -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Future Cloud Deployment Architecture

**AWS Deployment (Future Phase 6+):**
- Lambda functions for prediction API
- S3 for model artifacts and data storage
- SageMaker for model training and hosting
- CloudWatch for monitoring and logging
- API Gateway for RESTful endpoints

**Azure Deployment (Alternative):**
- Azure Functions for prediction API
- Blob Storage for artifacts
- Azure ML for model management
- Application Insights for monitoring

**Deployment Constraints:**
- HIPAA compliance required for healthcare data
- Model versioning and rollback capability
- A/B testing infrastructure for model comparison
- Real-time prediction latency < 200ms
- Audit logging for all predictions

### Clinical Validity Requirements

**High Precision Threshold (0.75):**
- Rationale: Minimize false positives to prevent alarm fatigue
- Trade-off: Lower recall (miss some true readmissions)
- Clinical justification: False alarms erode clinician trust

**Explainability Requirements:**
- SHAP values for every prediction
- Feature importance visualization
- Local explanation for individual patients
- Clinical validation of feature contributions

**Audit Trail Requirements:**
- Log every prediction with timestamp
- Store model version used for each prediction
- Record input features for reproducibility
- Enable retrospective analysis of model performance

**Regulatory Compliance:**
- Document model development process
- Validate on held-out test set
- Monitor for model drift in production
- Establish model retraining cadence

## Documentation Standards

### LOCAL_PROJECT_STORY.md

The project maintains a Director's Cut narrative documentation following the golden template structure:

**Sections:**
1. The Expert Architect's Opening: Project vision and motivation
2. The Interconnected Team: Stakeholder analysis (clinicians, data scientists, patients)
3. Phase 0-5 Development Roadmap: Detailed implementation timeline
4. UBC MDS Alignment Table: Mapping to data science competencies
5. What We Settled On: Architectural decision records
6. Technology Decisions: Rationale for each technology choice
7. What We Decided NOT to Use: Rejected alternatives with reasoning

**Writing Style:**
- No em dashes, no contractions
- Short paragraphs (2-3 sentences)
- No hyperbole or superlatives
- Grounded in facts and reality
- Technical depth with narrative flow

### README.md

The README provides quick-start guidance:

**Sections:**
1. Project Overview: One-paragraph summary
2. Architecture: Hexagonal Architecture diagram
3. Technology Stack: Key technologies with versions
4. Setup Instructions: Step-by-step environment setup
5. Testing: How to run tests and interpret results
6. Project Status: Current phase and next milestones

### Code Documentation

**Docstring Standards:**
- Google-style docstrings for all public APIs
- Args, Returns, Raises sections
- Example usage for complex functions
- Type hints in signatures, not docstrings

**Example:**
```python
def baseline_readmission_risk_flag(admission: Admission) -> str:
    """Rule-based baseline risk assessment using clinical heuristics.
    
    This function provides a non-ML baseline for comparison and fallback.
    It uses only pre-discharge features available at admission time.
    
    Args:
        admission: Admission object with pre-discharge features
        
    Returns:
        Risk category: 'Low Risk', 'Medium Risk', or 'High Risk'
        
    Example:
        >>> patient = Patient('P001', 75, 'M', 'Medicare')
        >>> admission = Admission(
        ...     'A001', patient, datetime.now(), 'Emergency',
        ...     'I50', comorbidity_count=4, length_of_stay_scheduled=7,
        ...     prior_admissions_count=2, lab_abnormalities_count=3,
        ...     medication_count=8
        ... )
        >>> baseline_readmission_risk_flag(admission)
        'High Risk'
    """
    # Implementation...
```

---

## Design Summary

This design document translates 16 requirements with 70+ acceptance criteria into a concrete technical architecture. The system implements Hexagonal Architecture with strict dependency inversion, ensuring domain logic remains independent of external frameworks. Data leakage prevention is enforced at system boundaries through explicit validation. Property-based testing with Hypothesis provides comprehensive correctness guarantees. The technology stack prioritizes type safety (Python 3.12, Mypy strict), code quality (Black, Ruff, pre-commit), and clinical validity (0.75 precision threshold, SHAP explainability).

The design is ready for implementation following the phased development roadmap documented in LOCAL_PROJECT_STORY.md.
