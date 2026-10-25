"""Pydantic schemas for fraud detection inference input validation and response serialization."""

from enum import Enum
import math
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class RiskCategory(str, Enum):
    """Standardized risk categories for transaction fraud classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TransactionInput(BaseModel):
    """Validates raw transaction payload containing numerical PCA and monetary features."""

    Amount: float = Field(
        ...,
        description="Transaction amount in currency units for raw input, or pre-scaled value",
        examples=[149.62],
    )
    Time: float = Field(
        ...,
        description="Seconds elapsed since start for raw input, or pre-scaled value",
        examples=[0.0],
    )
    V1: float = Field(..., description="PCA component 1", examples=[-1.359807])
    V2: float = Field(..., description="PCA component 2", examples=[-0.072781])
    V3: float = Field(..., description="PCA component 3", examples=[2.536347])
    V4: float = Field(..., description="PCA component 4", examples=[1.378155])
    V5: float = Field(..., description="PCA component 5", examples=[-0.338321])
    V6: float = Field(..., description="PCA component 6", examples=[0.462388])
    V7: float = Field(..., description="PCA component 7", examples=[0.239599])
    V8: float = Field(..., description="PCA component 8", examples=[0.098698])
    V9: float = Field(..., description="PCA component 9", examples=[0.363787])
    V10: float = Field(..., description="PCA component 10", examples=[0.090794])
    V11: float = Field(..., description="PCA component 11", examples=[-0.551600])
    V12: float = Field(..., description="PCA component 12", examples=[-0.617801])
    V13: float = Field(..., description="PCA component 13", examples=[-0.991390])
    V14: float = Field(..., description="PCA component 14", examples=[-0.311169])
    V15: float = Field(..., description="PCA component 15", examples=[1.468177])
    V16: float = Field(..., description="PCA component 16", examples=[-0.470401])
    V17: float = Field(..., description="PCA component 17", examples=[0.207971])
    V18: float = Field(..., description="PCA component 18", examples=[0.025791])
    V19: float = Field(..., description="PCA component 19", examples=[0.403993])
    V20: float = Field(..., description="PCA component 20", examples=[0.251412])
    V21: float = Field(..., description="PCA component 21", examples=[-0.018307])
    V22: float = Field(..., description="PCA component 22", examples=[0.277838])
    V23: float = Field(..., description="PCA component 23", examples=[-0.110474])
    V24: float = Field(..., description="PCA component 24", examples=[0.066928])
    V25: float = Field(..., description="PCA component 25", examples=[0.128539])
    V26: float = Field(..., description="PCA component 26", examples=[-0.189115])
    V27: float = Field(..., description="PCA component 27", examples=[0.133558])
    V28: float = Field(..., description="PCA component 28", examples=[-0.021053])

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "Amount": 149.62,
                "Time": 406.0,
                "V1": -2.312227,
                "V2": 1.951992,
                "V3": -1.609851,
                "V4": 3.997906,
                "V5": -0.522188,
                "V6": -1.426545,
                "V7": -2.537387,
                "V8": 1.391657,
                "V9": -2.770089,
                "V10": -2.772272,
                "V11": 3.202033,
                "V12": -2.899907,
                "V13": -0.595222,
                "V14": -4.289254,
                "V15": 0.389724,
                "V16": -1.140747,
                "V17": -2.830056,
                "V18": -0.016822,
                "V19": 0.416956,
                "V20": 0.126911,
                "V21": 0.517232,
                "V22": -0.035049,
                "V23": -0.465211,
                "V24": 0.320198,
                "V25": 0.044519,
                "V26": 0.177840,
                "V27": 0.261145,
                "V28": -0.143276,
            }
        },
    }

    @field_validator("*")
    @classmethod
    def check_finite_numbers(cls, v: float, info) -> float:
        """Validate that all numerical fields are non-null and finite (no NaN or +/-Inf)."""
        if isinstance(v, (int, float)):
            if math.isnan(v):
                raise ValueError(f"Feature '{info.field_name}' must not be NaN")
            if math.isinf(v):
                raise ValueError(f"Feature '{info.field_name}' must not be infinite")
        return float(v)


class PredictionRequest(BaseModel):
    """Payload for single transaction inference request with runtime configurations."""

    transaction: TransactionInput
    threshold: Optional[Union[float, str]] = Field(
        default="default",
        description="Threshold value (float between 0.0 and 1.0) or named threshold: 'default', 'cost_optimal', 'f1_optimal', 'f2_optimal'",
        examples=["default"],
    )
    risk_levels: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional custom cutoffs for risk levels. E.g. {'low': 0.182, 'medium': 0.50, 'high': 0.80}",
        examples=[{"low": 0.182, "medium": 0.50, "high": 0.80}],
    )
    apply_preprocessing: bool = Field(
        default=True,
        description="Set to True if transaction contains raw Amount and Time needing RobustScaler scaling. Set False if already scaled.",
    )

    @field_validator("threshold")
    @classmethod
    def validate_threshold_param(cls, v: Optional[Union[float, str]]) -> Optional[Union[float, str]]:
        if v is None:
            return "default"
        if isinstance(v, (int, float)):
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"Numerical threshold must be between 0.0 and 1.0, got {v}")
            return float(v)
        if isinstance(v, str):
            allowed_names = {"default", "cost_optimal", "f1_optimal", "f2_optimal"}
            if v.lower() not in allowed_names:
                raise ValueError(f"Unknown threshold name '{v}'. Allowed names: {sorted(allowed_names)}")
            return v.lower()
        raise ValueError(f"Invalid threshold type: {type(v)}. Must be float or string.")

    @model_validator(mode="after")
    def validate_raw_bounds(self) -> "PredictionRequest":
        if self.apply_preprocessing:
            if self.transaction.Amount < 0.0:
                raise ValueError(f"Raw transaction Amount cannot be negative, got {self.transaction.Amount}")
            if self.transaction.Time < 0.0:
                raise ValueError(f"Raw transaction Time cannot be negative, got {self.transaction.Time}")
        return self


class BatchPredictionRequest(BaseModel):
    """Payload for batch transactions prediction."""

    transactions: List[TransactionInput] = Field(
        ...,
        min_length=1,
        description="List of transaction objects to evaluate",
    )
    threshold: Optional[Union[float, str]] = Field(
        default="default",
        description="Threshold value (float between 0.0 and 1.0) or named threshold",
    )
    risk_levels: Optional[Dict[str, float]] = Field(default=None)
    apply_preprocessing: bool = Field(
        default=True,
        description="Whether to apply preprocessing to Amount and Time",
    )

    @field_validator("threshold")
    @classmethod
    def validate_threshold_param(cls, v: Optional[Union[float, str]]) -> Optional[Union[float, str]]:
        if v is None:
            return "default"
        if isinstance(v, (int, float)):
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"Numerical threshold must be between 0.0 and 1.0, got {v}")
            return float(v)
        if isinstance(v, str):
            allowed_names = {"default", "cost_optimal", "f1_optimal", "f2_optimal"}
            if v.lower() not in allowed_names:
                raise ValueError(f"Unknown threshold name '{v}'. Allowed names: {sorted(allowed_names)}")
            return v.lower()
        raise ValueError(f"Invalid threshold type: {type(v)}")

    @model_validator(mode="after")
    def validate_raw_bounds(self) -> "BatchPredictionRequest":
        if self.apply_preprocessing:
            for idx, tx in enumerate(self.transactions):
                if tx.Amount < 0.0:
                    raise ValueError(f"Raw transaction at index {idx} has negative Amount: {tx.Amount}")
                if tx.Time < 0.0:
                    raise ValueError(f"Raw transaction at index {idx} has negative Time: {tx.Time}")
        return self


class PredictionResponse(BaseModel):
    """Output schema for transaction scoring prediction."""

    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability that transaction is fraudulent (0.0 to 1.0)",
    )
    is_fraud: bool = Field(
        ...,
        description="Binary decision: True if fraud_probability >= threshold_used, else False",
    )
    risk_category: str = Field(
        ...,
        description="Risk categorization tier: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'",
    )
    threshold_used: float = Field(
        ...,
        description="The operational decision threshold applied to classify this transaction",
    )
    threshold_name: Optional[str] = Field(
        default=None,
        description="The symbolic name of the threshold (e.g. 'cost_optimal', 'default', 'custom')",
    )
    model_version: str = Field(
        ...,
        description="Version string of the scoring model artifact",
    )
    model_name: str = Field(
        ...,
        description="Name of the champion model architecture",
    )


class BatchPredictionResponse(BaseModel):
    """Output schema for batch transaction scoring."""

    total_transactions: int = Field(..., description="Total count of transactions evaluated")
    fraud_count: int = Field(..., description="Count of transactions flagged as fraudulent")
    threshold_used: float = Field(..., description="Operational decision threshold applied")
    model_version: str = Field(..., description="Model version")
    predictions: List[PredictionResponse] = Field(..., description="Individual scoring results")


class ModelMetadataResponse(BaseModel):
    """Detailed model metadata and operational parameters."""

    model_name: str
    model_version: str
    feature_names: List[str]
    features_count: int
    evaluation_metrics: Dict[str, float]
    operational_thresholds: Dict[str, float]
    cost_parameters: Dict[str, float]
    top_5_predictive_features: List[str]


class HealthResponse(BaseModel):
    """API health and service status response."""

    status: str = "healthy"
    model_loaded: bool = True
    model_version: str
    model_name: str
    features_count: int
