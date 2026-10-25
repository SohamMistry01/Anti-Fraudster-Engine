"""Preprocessing and feature verification utilities for fraud inference pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# Canonical feature ordering matching model training artifacts
PCA_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]
SKEWED_FEATURES: List[str] = ["Amount", "Time"]
FEATURE_NAMES: List[str] = SKEWED_FEATURES + PCA_FEATURES

_CACHED_PREPROCESSOR: Optional[ColumnTransformer] = None


def create_preprocessor() -> ColumnTransformer:
    """Instantiate the feature-aware ColumnTransformer architecture matching Part 1.

    Architecture:
    - Skewed features ('Amount', 'Time'): Median Imputer + RobustScaler (IQR-based).
    - PCA features ('V1' through 'V28'): Mean Imputer + Passthrough.
    """
    skewed_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    pca_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
    ])

    return ColumnTransformer(
        transformers=[
            ("skewed", skewed_pipeline, SKEWED_FEATURES),
            ("pca", pca_pipeline, PCA_FEATURES),
        ],
        remainder="drop",
    )


def resolve_artifact_path(relative_path: str) -> Path:
    """Resolve file path relative to current working directory or repository root."""
    path = Path(relative_path)
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        Path.cwd() / relative_path,
        Path(__file__).resolve().parent.parent.parent / relative_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def fit_and_save_preprocessor(output_path: Optional[Union[str, Path]] = None) -> ColumnTransformer:
    """Fit ColumnTransformer on stratified training split of raw data and serialize."""
    from sklearn.model_selection import train_test_split

    raw_path = resolve_artifact_path("data/creditcard.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Cannot fit preprocessor: raw dataset not found at '{raw_path}'")

    logger.info("Fitting new preprocessor from %s...", raw_path)
    df = pd.read_csv(raw_path)
    df_cleaned = df.drop_duplicates(keep="first")
    X = df_cleaned.drop(columns=["Class"])
    y = df_cleaned["Class"]

    X_train, _, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)

    save_path = resolve_artifact_path(str(output_path) if output_path else "models/preprocessor.joblib")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_path)
    logger.info("Preprocessor successfully fitted and saved to %s", save_path)
    return preprocessor


def load_preprocessor(path: Optional[Union[str, Path]] = None) -> ColumnTransformer:
    """Load the pre-fitted ColumnTransformer from serialized joblib artifact."""
    global _CACHED_PREPROCESSOR
    if _CACHED_PREPROCESSOR is not None:
        return _CACHED_PREPROCESSOR

    target_path = resolve_artifact_path(str(path) if path else "models/preprocessor.joblib")
    if not target_path.exists():
        # Attempt automatic fallback fit if creditcard.csv exists
        raw_path = resolve_artifact_path("data/creditcard.csv")
        if raw_path.exists():
            logger.warning(
                "Preprocessor artifact '%s' missing. Auto-fitting from raw dataset...",
                target_path,
            )
            preprocessor = fit_and_save_preprocessor(target_path)
            _CACHED_PREPROCESSOR = preprocessor
            return preprocessor

        raise FileNotFoundError(
            f"Fitted preprocessor artifact not found at '{target_path}'. "
            "Please ensure 'models/preprocessor.joblib' is generated."
        )

    logger.info("Loading preprocessor from %s", target_path)
    preprocessor = joblib.load(target_path)
    _CACHED_PREPROCESSOR = preprocessor
    return preprocessor


def verify_features(
    data: Union[dict, List[dict], pd.DataFrame, pd.Series],
    expected_features: List[str] = FEATURE_NAMES,
    strict_extra: bool = False,
) -> pd.DataFrame:
    """Validate feature completeness, detect invalid values, and enforce exact column ordering.

    Args:
        data: Input payload as single dict, list of dicts, pandas Series, or DataFrame.
        expected_features: Canonical list of feature names in required order.
        strict_extra: If True, reject payloads that include unknown extraneous columns.

    Returns:
        pd.DataFrame strictly sorted with columns matching expected_features.

    Raises:
        ValueError: If required features are missing, extra features present (when strict),
                    or values are non-numeric / Infinite.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        df = pd.DataFrame([data.to_dict()])
    elif isinstance(data, list):
        if not data:
            raise ValueError("Input transaction list cannot be empty.")
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(f"Unsupported data type for verification: {type(data)}")

    # Detect missing features
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Input is missing {len(missing_features)} required feature(s): {missing_features}"
        )

    # Detect unexpected extra features if strict
    if strict_extra:
        extra_features = [c for c in df.columns if c not in expected_features]
        if extra_features:
            raise ValueError(
                f"Unexpected extra feature(s) detected: {extra_features}. "
                f"Expected only the 30 features: {expected_features}"
            )

    # Reorder columns to guarantee exact match with expected_features
    df_ordered = df[expected_features].copy()

    # Enforce numeric float types
    for col in expected_features:
        try:
            df_ordered[col] = df_ordered[col].astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Feature '{col}' contains non-numeric values: {exc}") from exc

    # Check for Infinite values
    numeric_vals = df_ordered.to_numpy(dtype=float)
    if np.isinf(numeric_vals).any():
        raise ValueError("Input data contains Infinite values, which are invalid for inference.")

    return df_ordered


def preprocess_input(
    data: Union[dict, List[dict], pd.DataFrame, pd.Series],
    preprocessor: Optional[ColumnTransformer] = None,
    apply_scaling: bool = True,
    expected_features: List[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Verify feature alignment and apply feature-aware transformations.

    Args:
        data: Raw transaction or batch of transactions.
        preprocessor: Fitted ColumnTransformer instance (if None, loaded from cache/disk).
        apply_scaling: If True, scales 'Amount' and 'Time' via RobustScaler.
                       If False, assumes inputs are already scaled (e.g. from test_data.csv).
        expected_features: Canonical list of feature names.

    Returns:
        np.ndarray of shape (N, 30) with float values ready for model inference.
    """
    # 1. Enforce feature ordering and schema validation
    df_ordered = verify_features(data, expected_features=expected_features, strict_extra=False)

    # 2. Apply preprocessing transformation
    if apply_scaling:
        proc = preprocessor or load_preprocessor()
        transformed = proc.transform(df_ordered)
        return np.asarray(transformed, dtype=np.float32)

    return np.asarray(df_ordered.values, dtype=np.float32)
