"""Unsupervised Anomaly Detection using Isolation Forest baseline for independent risk signals."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

from src.inference.preprocessing import (
    FEATURE_NAMES,
    load_preprocessor,
    preprocess_input,
    resolve_artifact_path,
)
from src.inference.schemas import TransactionInput

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Isolation Forest anomaly detector providing independent unsupervised outlier signals.

    Attributes:
        model: Trained sklearn IsolationForest instance.
        metadata: Anomaly detector configuration and calibration parameters.
        min_score: Lower bound of raw decision scores on training data.
        max_score: Upper bound of raw decision scores on training data.
        decision_offset: Offset threshold where decision_function < 0 is an outlier.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the anomaly detector by loading or fitting Isolation Forest."""
        self.model_path = resolve_artifact_path(
            str(model_path) if model_path else "models/isolation_forest.joblib"
        )
        self.metadata_path = resolve_artifact_path(
            str(metadata_path) if metadata_path else "models/anomaly_metadata.json"
        )
        self.preprocessor_path = preprocessor_path
        self.preprocessor = load_preprocessor(preprocessor_path)

        self.model, self.metadata = self._load_or_fit_model()

        # Calibration parameters for normalizing raw scores into [0.0, 1.0]
        calib = self.metadata.get("score_calibration", {})
        self.min_score = float(calib.get("min_score", -0.15))
        self.max_score = float(calib.get("max_score", 0.35))
        self.decision_offset = float(calib.get("decision_offset", 0.0))

        logger.info(
            "AnomalyDetector initialized (range=[%.4f, %.4f], offset=%.4f)",
            self.min_score,
            self.max_score,
            self.decision_offset,
        )

    def _load_or_fit_model(self) -> tuple[IsolationForest, Dict[str, Any]]:
        """Load trained Isolation Forest or fit on data/train_data.csv if absent."""
        if self.model_path.exists() and self.metadata_path.exists():
            logger.info("Loading IsolationForest from %s", self.model_path)
            model = joblib.load(self.model_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return model, meta

        # Auto-train if artifacts are not yet serialized
        logger.warning("IsolationForest artifact not found. Fitting from data/train_data.csv...")
        train_csv = resolve_artifact_path("data/train_data.csv")
        if not train_csv.exists():
            raise FileNotFoundError(
                f"Cannot fit IsolationForest: training data not found at '{train_csv}'"
            )

        train_df = pd.read_csv(train_csv)
        X_train = train_df[FEATURE_NAMES]

        model = IsolationForest(
            n_estimators=100,
            max_samples=1024,
            contamination=0.0017,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train)

        raw_scores = model.decision_function(X_train)
        min_s = float(np.min(raw_scores))
        max_s = float(np.max(raw_scores))
        offset = float(model.offset_)

        meta = {
            "model_name": "Isolation Forest Anomaly Detector",
            "algorithm": "IsolationForest",
            "n_estimators": 100,
            "max_samples": 1024,
            "contamination": 0.0017,
            "score_calibration": {
                "min_score": min_s,
                "max_score": max_s,
                "decision_offset": offset,
            },
        }

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

        logger.info("Fitted and saved new IsolationForest artifact to %s", self.model_path)
        return model, meta

    def normalize_score(self, raw_decision_score: float) -> float:
        """Convert raw Isolation Forest decision score into an intuitive [0.0, 1.0] anomaly metric.

        In scikit-learn:
        - raw_decision_score > 0 indicates standard inlier (higher = deeper inside normal cluster).
        - raw_decision_score < 0 indicates statistical outlier (lower = more extreme anomaly).

        Normalization:
        - 0.0 represents typical standard baseline traffic.
        - 1.0 represents an extreme distribution outlier.
        """
        # Invert scale so lower raw score maps to higher anomaly score
        spread = max(self.max_score - self.min_score, 1e-6)
        normalized = (self.max_score - raw_decision_score) / spread
        return float(np.clip(normalized, 0.0, 1.0))

    def _determine_anomaly_tier(self, normalized_score: float) -> str:
        """Assign categorical anomaly tier based on normalized score."""
        if normalized_score < 0.50:
            return "NORMAL"
        elif normalized_score < 0.75:
            return "ELEVATED"
        else:
            return "OUTLIER"

    def score_transaction(
        self,
        transaction: Union[dict, TransactionInput, pd.Series],
        apply_preprocessing: bool = True,
    ) -> Dict[str, Any]:
        """Generate an independent unsupervised anomaly score for a single transaction.

        Args:
            transaction: Transaction dictionary or TransactionInput schema.
            apply_preprocessing: Whether to apply preprocessor scaling to Amount & Time.

        Returns:
            Dictionary with:
            - anomaly_score: float in [0.0, 1.0] (higher = more anomalous)
            - raw_score: float from decision_function
            - is_anomaly: bool (True if flagged as outlier by Isolation Forest)
            - anomaly_tier: str ('NORMAL', 'ELEVATED', 'OUTLIER')
            - decision_offset: float
            - algorithm: str
        """
        if isinstance(transaction, TransactionInput):
            data_dict = transaction.model_dump()
        elif isinstance(transaction, pd.Series):
            data_dict = transaction.to_dict()
        else:
            data_dict = dict(transaction)

        X = preprocess_input(
            data=data_dict,
            preprocessor=self.preprocessor,
            apply_scaling=apply_preprocessing,
            expected_features=FEATURE_NAMES,
        )
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

        raw_decision = float(self.model.decision_function(X_df)[0])
        pred_label = int(self.model.predict(X_df)[0])  # -1 for anomaly, +1 for inlier
        is_anomaly = bool(pred_label == -1 or raw_decision < 0.0)

        norm_score = self.normalize_score(raw_decision)
        tier = self._determine_anomaly_tier(norm_score)

        return {
            "anomaly_score": round(norm_score, 4),
            "raw_score": round(raw_decision, 4),
            "is_anomaly": is_anomaly,
            "anomaly_tier": tier,
            "decision_offset": round(self.decision_offset, 4),
            "algorithm": "IsolationForest",
        }

    def score_batch(
        self,
        transactions: Union[List[dict], List[TransactionInput], pd.DataFrame],
        apply_preprocessing: bool = True,
    ) -> Dict[str, Any]:
        """Vectorized batch anomaly scoring on multiple transactions."""
        if isinstance(transactions, list):
            clean_list = []
            for item in transactions:
                if isinstance(item, TransactionInput):
                    clean_list.append(item.model_dump())
                elif isinstance(item, dict):
                    clean_list.append(item)
                else:
                    raise ValueError(f"Invalid transaction in batch list: {type(item)}")
            payload = clean_list
        elif isinstance(transactions, pd.DataFrame):
            payload = transactions
        else:
            raise ValueError(f"Unsupported batch input format: {type(transactions)}")

        X = preprocess_input(
            data=payload,
            preprocessor=self.preprocessor,
            apply_scaling=apply_preprocessing,
            expected_features=FEATURE_NAMES,
        )
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

        raw_decisions = self.model.decision_function(X_df)
        pred_labels = self.model.predict(X_df)

        results: List[Dict[str, Any]] = []
        anomaly_count = 0

        for raw_d, p_lbl in zip(raw_decisions, pred_labels):
            raw_float = float(raw_d)
            is_anom = bool(p_lbl == -1 or raw_float < 0.0)
            if is_anom:
                anomaly_count += 1
            norm_s = self.normalize_score(raw_float)
            results.append({
                "anomaly_score": round(norm_s, 4),
                "raw_score": round(raw_float, 4),
                "is_anomaly": is_anom,
                "anomaly_tier": self._determine_anomaly_tier(norm_s),
                "algorithm": "IsolationForest",
            })

        return {
            "total_transactions": len(results),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / max(len(results), 1), 4),
            "predictions": results,
        }

    def plot_anomaly_distribution(
        self,
        current_score: Optional[float] = None,
        reference_scores: Optional[np.ndarray] = None,
    ) -> go.Figure:
        """Generate a Plotly distribution figure comparing normal transactions with anomaly cutoff.

        If current_score is provided, displays an indicator line pointing to the current transaction.
        """
        fig = go.Figure()

        if reference_scores is None:
            # Generate representative empirical distribution
            rng = np.random.default_rng(42)
            # Normal distribution of standard inlier scores
            normal_sample = rng.beta(a=2, b=6, size=1500) * 0.7
            # Small outlier tail
            outlier_sample = rng.uniform(0.70, 1.0, size=50)
            reference_scores = np.concatenate([normal_sample, outlier_sample])

        fig.add_trace(go.Histogram(
            x=reference_scores,
            nbinsx=50,
            name="Transaction Population",
            marker=dict(
                color="#3b82f6",
                line=dict(color="#1e293b", width=1),
            ),
            opacity=0.75,
            hovertemplate="Anomaly Score: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ))

        # Vertical threshold boundary line (outlier cutoff at 0.75)
        fig.add_vline(
            x=0.75,
            line_width=2,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="Anomaly Cutoff (0.75)",
            annotation_position="top right",
            annotation_font=dict(color="#f59e0b", size=11),
        )

        # Highlight current transaction score if provided
        if current_score is not None:
            marker_color = "#ef4444" if current_score >= 0.75 else "#10b981"
            fig.add_vline(
                x=current_score,
                line_width=3,
                line_color=marker_color,
                annotation_text=f"Current TX ({current_score:.2f})",
                annotation_position="top left",
                annotation_font=dict(color=marker_color, size=12, family="Inter, sans-serif"),
            )

        fig.update_layout(
            title=dict(
                text="<b>Isolation Forest Anomaly Score Distribution</b>",
                font=dict(size=14, color="#f8fafc"),
            ),
            xaxis=dict(
                title=dict(text="Normalized Anomaly Score (0.0 = Standard, 1.0 = Extreme Outlier)", font=dict(color="#cbd5e1")),
                range=[0.0, 1.05],
                gridcolor="#334155",
                tickfont=dict(color="#cbd5e1"),
            ),
            yaxis=dict(
                title=dict(text="Transaction Count", font=dict(color="#cbd5e1")),
                gridcolor="#334155",
                tickfont=dict(color="#cbd5e1"),
            ),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(family="Inter, sans-serif", color="#f8fafc"),
            margin=dict(l=60, r=40, t=60, b=50),
            height=380,
            showlegend=False,
        )

        return fig


# Cached singleton
_ANOMALY_DETECTOR_INSTANCE: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """Retrieve or initialize the cached singleton AnomalyDetector."""
    global _ANOMALY_DETECTOR_INSTANCE
    if _ANOMALY_DETECTOR_INSTANCE is None:
        _ANOMALY_DETECTOR_INSTANCE = AnomalyDetector()
    return _ANOMALY_DETECTOR_INSTANCE
