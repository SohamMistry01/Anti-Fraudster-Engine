"""Reusable Fraud Predictor engine loading trained XGBoost model and executing inference."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.inference.preprocessing import (
    FEATURE_NAMES,
    load_preprocessor,
    preprocess_input,
    resolve_artifact_path,
    verify_features,
)
from src.inference.schemas import (
    PredictionResponse,
    RiskCategory,
    TransactionInput,
)

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Production-ready inference pipeline for credit card fraud scoring.

    Attributes:
        model: Trained XGBoost classifier.
        metadata: Model metadata dictionary.
        preprocessor: Fitted ColumnTransformer for feature preprocessing.
        model_name: Identifier name of the model.
        model_version: Version string of the model.
        feature_names: Canonical list of 30 feature names.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the predictor by loading artifacts and verifying consistency."""
        self.model_path = resolve_artifact_path(str(model_path) if model_path else "models/xgb_fraud_detector.joblib")
        self.metadata_path = resolve_artifact_path(str(metadata_path) if metadata_path else "models/model_metadata.json")
        self.preprocessor_path = resolve_artifact_path(str(preprocessor_path) if preprocessor_path else "models/preprocessor.joblib")

        self.metadata = self._load_metadata()
        self.model = self._load_model()
        self.preprocessor = self._load_preprocessor()

        self.model_name = self.metadata.get("model_name", "XGBoost Classifier")
        self.model_version = self.metadata.get("model_version", "1.0.0")
        self.feature_names = self.metadata.get("feature_names", FEATURE_NAMES)

        # Enforce strict verification between model artifact, metadata, and canonical features
        self.verify_model_consistency()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load and parse model metadata from JSON artifact."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Model metadata file not found at '{self.metadata_path}'")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta

    def _load_model(self) -> Any:
        """Load trained XGBoost model from joblib or native JSON booster."""
        if not self.model_path.exists():
            # Fallback to json if joblib not found
            alt_path = self.model_path.with_suffix(".json")
            if alt_path.exists():
                logger.info("Loading native XGBoost booster from %s", alt_path)
                booster = xgb.Booster()
                booster.load_model(str(alt_path))
                return booster
            raise FileNotFoundError(f"Trained model artifact not found at '{self.model_path}'")

        if self.model_path.suffix == ".json":
            logger.info("Loading native XGBoost booster from %s", self.model_path)
            booster = xgb.Booster()
            booster.load_model(str(self.model_path))
            return booster

        logger.info("Loading model from %s", self.model_path)
        try:
            return joblib.load(self.model_path)
        except Exception as exc:
            alt_path = self.model_path.with_suffix(".json")
            if alt_path.exists():
                logger.warning("Joblib loading failed (%s). Falling back to JSON booster %s", exc, alt_path)
                booster = xgb.Booster()
                booster.load_model(str(alt_path))
                return booster
            raise exc

    def _load_preprocessor(self) -> Any:
        """Load fitted ColumnTransformer preprocessor."""
        return load_preprocessor(self.preprocessor_path)

    def verify_model_consistency(self) -> None:
        """Verify feature ordering, count, and model version match training configuration.

        Raises:
            ValueError: If feature count or ordering does not match training specification.
        """
        # 1. Verify model feature names if available
        model_features: Optional[List[str]] = None
        if hasattr(self.model, "feature_names_in_"):
            model_features = list(self.model.feature_names_in_)
        elif hasattr(self.model, "get_booster"):
            model_features = self.model.get_booster().feature_names
        elif isinstance(self.model, xgb.Booster):
            model_features = self.model.feature_names

        if model_features is not None:
            if len(model_features) != len(self.feature_names):
                raise ValueError(
                    f"Model expects {len(model_features)} features, but metadata has {len(self.feature_names)}"
                )
            if list(model_features) != list(self.feature_names):
                raise ValueError(
                    f"Feature order mismatch! Model expects {model_features}, but metadata defines {self.feature_names}"
                )

        # 2. Verify canonical FEATURE_NAMES match metadata
        if list(self.feature_names) != list(FEATURE_NAMES):
            raise ValueError(
                f"Metadata feature list does not match canonical order: {self.feature_names} != {FEATURE_NAMES}"
            )

        # 3. Verify operational thresholds exist
        op_thresholds = self.metadata.get("operational_thresholds", {})
        if "default" not in op_thresholds:
            raise ValueError("Operational thresholds must contain 'default' threshold definition.")

        logger.info(
            "Model consistency verified: version=%s, features=%d, thresholds=%s",
            self.model_version,
            len(self.feature_names),
            list(op_thresholds.keys()),
        )

    def resolve_threshold(self, threshold: Optional[Union[float, str]]) -> Tuple[float, str]:
        """Resolve threshold specification to a concrete numeric cutoff and label.

        Args:
            threshold: Named threshold string ('default', 'cost_optimal', 'f1_optimal', 'f2_optimal')
                       or float between 0.0 and 1.0.

        Returns:
            Tuple of (cutoff_value: float, threshold_label: str)
        """
        thresholds = self.metadata.get("operational_thresholds", {
            "default": 0.5,
            "cost_optimal": 0.182,
            "f1_optimal": 0.798,
            "f2_optimal": 0.552,
        })

        if threshold is None or threshold == "default":
            val = float(thresholds.get("default", 0.5))
            return val, "default"

        if isinstance(threshold, str):
            key = threshold.lower().strip()
            if key not in thresholds:
                raise ValueError(
                    f"Unknown threshold name '{threshold}'. Valid operational options: {list(thresholds.keys())}"
                )
            return float(thresholds[key]), key

        if isinstance(threshold, (int, float)):
            f_val = float(threshold)
            if not (0.0 <= f_val <= 1.0):
                raise ValueError(f"Threshold value must be between 0.0 and 1.0, got {f_val}")
            # Check if matches any known named threshold
            for k, v in thresholds.items():
                if abs(f_val - float(v)) < 1e-4:
                    return f_val, k
            return f_val, "custom"

        raise ValueError(f"Invalid threshold type: {type(threshold)}. Must be float or str.")

    def calculate_risk_category(
        self,
        probability: float,
        threshold: float,
        custom_levels: Optional[Dict[str, float]] = None,
    ) -> str:
        """Categorize transaction risk into LOW, MEDIUM, HIGH, or CRITICAL.

        Business tiering logic:
        - If custom_levels provided, e.g. {'low': 0.182, 'medium': 0.50, 'high': 0.80}:
          prob < low -> LOW
          low <= prob < medium -> MEDIUM
          medium <= prob < high -> HIGH
          prob >= high -> CRITICAL
        - Default policy:
          prob < cost_optimal (0.182): LOW (Auto-approved)
          cost_optimal <= prob < threshold (e.g. 0.5): MEDIUM (Step-up 2FA challenge / Review)
          threshold <= prob < 0.80: HIGH (Decline / Block)
          prob >= 0.80: CRITICAL (Immediate high-priority fraud alert)
        """
        if custom_levels:
            low_cutoff = float(custom_levels.get("low", 0.182))
            med_cutoff = float(custom_levels.get("medium", threshold))
            high_cutoff = float(custom_levels.get("high", 0.80))
        else:
            cost_opt = float(self.metadata.get("operational_thresholds", {}).get("cost_optimal", 0.182))
            low_cutoff = min(cost_opt, threshold)
            med_cutoff = max(cost_opt, threshold)
            high_cutoff = 0.80

        if probability < low_cutoff:
            return RiskCategory.LOW.value
        elif probability < med_cutoff:
            return RiskCategory.MEDIUM.value
        elif probability < high_cutoff:
            return RiskCategory.HIGH.value
        else:
            return RiskCategory.CRITICAL.value

    def _predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        """Internal helper to predict positive class probabilities from 2D numpy array."""
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        elif isinstance(self.model, xgb.Booster):
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            probs = self.model.predict(dmatrix)
        else:
            raise TypeError(f"Unsupported model type for prediction: {type(self.model)}")
        return np.asarray(probs, dtype=float)

    def predict_transaction(
        self,
        transaction: Union[dict, TransactionInput, pd.Series],
        threshold: Union[float, str] = "default",
        risk_levels: Optional[Dict[str, float]] = None,
        apply_preprocessing: bool = True,
    ) -> Dict[str, Any]:
        """Reusable function to score a single transaction.

        Args:
            transaction: Transaction data as dict, Pydantic TransactionInput, or pandas Series.
            threshold: Operational threshold cutoff (str name or float 0.0-1.0).
            risk_levels: Optional custom cutoffs for risk categorization.
            apply_preprocessing: If True, applies fitted RobustScaler to Amount & Time.
                                Set False if passing already preprocessed values.

        Returns:
            Dictionary containing:
            - fraud_probability: float
            - is_fraud: bool
            - risk_category: str ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
            - threshold_used: float
            - threshold_name: str
            - model_version: str
            - model_name: str
        """
        # 1. Convert Pydantic model to dictionary if needed
        if isinstance(transaction, TransactionInput):
            data_dict = transaction.model_dump()
        elif isinstance(transaction, pd.Series):
            data_dict = transaction.to_dict()
        elif isinstance(transaction, dict):
            # Validate through Pydantic schema for strict input guarantees
            validated = TransactionInput(**transaction)
            data_dict = validated.model_dump()
        else:
            raise ValueError(f"Unsupported transaction type: {type(transaction)}")

        # 2. Enforce non-negative bounds on raw Amount and Time if preprocessing is requested
        if apply_preprocessing:
            if float(data_dict.get("Amount", 0.0)) < 0.0:
                raise ValueError(f"Raw transaction Amount cannot be negative: {data_dict.get('Amount')}")
            if float(data_dict.get("Time", 0.0)) < 0.0:
                raise ValueError(f"Raw transaction Time cannot be negative: {data_dict.get('Time')}")

        # 3. Preprocess input to enforce exact feature order and scaling
        X = preprocess_input(
            data=data_dict,
            preprocessor=self.preprocessor,
            apply_scaling=apply_preprocessing,
            expected_features=self.feature_names,
        )

        # 3. Model inference
        prob = float(self._predict_raw_proba(X)[0])

        # 4. Threshold & Decision
        cutoff_val, cutoff_name = self.resolve_threshold(threshold)
        is_fraud = bool(prob >= cutoff_val)
        risk_cat = self.calculate_risk_category(prob, cutoff_val, risk_levels)

        return {
            "fraud_probability": round(prob, 6),
            "is_fraud": is_fraud,
            "risk_category": risk_cat,
            "threshold_used": round(cutoff_val, 4),
            "threshold_name": cutoff_name,
            "model_version": self.model_version,
            "model_name": self.model_name,
        }

    def predict_batch(
        self,
        transactions: Union[List[dict], List[TransactionInput], pd.DataFrame],
        threshold: Union[float, str] = "default",
        risk_levels: Optional[Dict[str, float]] = None,
        apply_preprocessing: bool = True,
    ) -> Dict[str, Any]:
        """High-performance vectorized batch transaction scoring.

        Args:
            transactions: List of dicts, list of TransactionInputs, or pandas DataFrame.
            threshold: Operational threshold cutoff.
            risk_levels: Optional custom cutoffs for risk categorization.
            apply_preprocessing: Whether to apply RobustScaler preprocessing.

        Returns:
            Batch summary dict with list of individual prediction responses.
        """
        if isinstance(transactions, list):
            clean_list = []
            for item in transactions:
                if isinstance(item, TransactionInput):
                    clean_list.append(item.model_dump())
                elif isinstance(item, dict):
                    clean_list.append(TransactionInput(**item).model_dump())
                else:
                    raise ValueError(f"Invalid transaction in batch list: {type(item)}")
            payload = clean_list
        elif isinstance(transactions, pd.DataFrame):
            payload = transactions
        # 1. Enforce non-negative bounds on raw inputs if preprocessing is requested
        if apply_preprocessing:
            items_to_check = payload if isinstance(payload, list) else payload.to_dict(orient="records")
            for idx, item in enumerate(items_to_check):
                if float(item.get("Amount", 0.0)) < 0.0:
                    raise ValueError(f"Raw transaction at index {idx} has negative Amount: {item.get('Amount')}")
                if float(item.get("Time", 0.0)) < 0.0:
                    raise ValueError(f"Raw transaction at index {idx} has negative Time: {item.get('Time')}")

        # 2. Preprocess batch
        X = preprocess_input(
            data=payload,
            preprocessor=self.preprocessor,
            apply_scaling=apply_preprocessing,
            expected_features=self.feature_names,
        )

        # 2. Model inference
        probs = self._predict_raw_proba(X)

        # 3. Decisions & Categories
        cutoff_val, cutoff_name = self.resolve_threshold(threshold)

        results: List[Dict[str, Any]] = []
        fraud_count = 0

        for p in probs:
            p_float = float(p)
            is_fraud = bool(p_float >= cutoff_val)
            if is_fraud:
                fraud_count += 1
            risk_cat = self.calculate_risk_category(p_float, cutoff_val, risk_levels)

            results.append({
                "fraud_probability": round(p_float, 6),
                "is_fraud": is_fraud,
                "risk_category": risk_cat,
                "threshold_used": round(cutoff_val, 4),
                "threshold_name": cutoff_name,
                "model_version": self.model_version,
                "model_name": self.model_name,
            })

        return {
            "total_transactions": len(results),
            "fraud_count": fraud_count,
            "threshold_used": round(cutoff_val, 4),
            "threshold_name": cutoff_name,
            "model_version": self.model_version,
            "predictions": results,
        }


# Singleton pattern for fast reuse in web services and scripts
_PREDICTOR_INSTANCE: Optional[FraudPredictor] = None


def get_predictor() -> FraudPredictor:
    """Retrieve or initialize the cached singleton FraudPredictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = FraudPredictor()
    return _PREDICTOR_INSTANCE


def predict_transaction(
    transaction: Union[dict, TransactionInput, pd.Series],
    threshold: Union[float, str] = "default",
    risk_levels: Optional[Dict[str, float]] = None,
    apply_preprocessing: bool = True,
) -> Dict[str, Any]:
    """Module-level reusable function to score a transaction using the singleton predictor."""
    predictor = get_predictor()
    return predictor.predict_transaction(
        transaction=transaction,
        threshold=threshold,
        risk_levels=risk_levels,
        apply_preprocessing=apply_preprocessing,
    )
