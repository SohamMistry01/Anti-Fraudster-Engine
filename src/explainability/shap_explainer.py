"""SHAP explainability engine for individual XGBoost fraud predictions with Plotly visualisations.

Provides mathematical and statistical evidence for model decisions without inventing
missing cardholder or merchant information.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from src.inference.predictor import FraudPredictor, get_predictor
from src.inference.preprocessing import FEATURE_NAMES, preprocess_input
from src.inference.schemas import TransactionInput

logger = logging.getLogger(__name__)


class FraudSHAPExplainer:
    """Explains XGBoost fraud predictions using TreeSHAP and interactive Plotly figures.

    Attributes:
        predictor: Initialized FraudPredictor instance.
        explainer: Fitted shap.TreeExplainer.
        expected_value: Base value (model expected log-odds or probability margin).
    """

    def __init__(self, predictor: Optional[FraudPredictor] = None) -> None:
        """Initialize SHAP explainer with the trained XGBoost model."""
        self.predictor = predictor or get_predictor()
        logger.info("Initializing shap.TreeExplainer for %s...", self.predictor.model_name)

        # In XGBoost, sklearn wrapper or raw booster can be passed to TreeExplainer
        self.explainer = shap.TreeExplainer(self.predictor.model)

        # Base value represents the prior baseline margin before feature contributions
        raw_base = self.explainer.expected_value
        if isinstance(raw_base, (list, np.ndarray)):
            self.base_value = float(raw_base[0]) if len(raw_base) > 0 else 0.0
        else:
            self.base_value = float(raw_base)

        logger.info("TreeExplainer initialized with base_value=%.4f", self.base_value)

    def _generate_statistical_evidence(
        self,
        feature_name: str,
        feature_value: float,
        shap_value: float,
        raw_transaction: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format purely factual, statistical evidence for a feature's contribution.

        CRITICAL INTEGRITY RULE:
        V1 to V28 are anonymized PCA components. Under no circumstances should domain stories
        such as 'card velocity', 'merchant country mismatch', or 'ATM location' be fabricated.
        The evidence describes the statistical magnitude, deviation, and directional impact.
        """
        impact_dir = "increases fraud risk" if shap_value > 0 else "decreases fraud risk"
        abs_shap = abs(shap_value)

        if feature_name == "Amount":
            raw_amt = raw_transaction.get("Amount", feature_value) if raw_transaction else feature_value
            return (
                f"Transaction Amount (${raw_amt:.2f}) has SHAP impact {shap_value:+.3f}, "
                f"which {impact_dir} (magnitude: {abs_shap:.3f})."
            )
        elif feature_name == "Time":
            raw_time = raw_transaction.get("Time", feature_value) if raw_transaction else feature_value
            return (
                f"Transaction Time offset ({raw_time:.1f}s) has SHAP impact {shap_value:+.3f}, "
                f"which {impact_dir}."
            )
        else:
            # PCA Feature: Describe mathematical relationship strictly
            direction_desc = "elevated positive value" if feature_value > 0 else "depressed negative value"
            return (
                f"PCA component {feature_name} = {feature_value:+.3f} ({direction_desc}) shifts "
                f"decision log-odds by {shap_value:+.3f}, which {impact_dir}."
            )

    def explain_transaction(
        self,
        transaction: Union[dict, TransactionInput, pd.Series],
        top_n: int = 10,
        apply_preprocessing: bool = True,
        generate_plot: bool = True,
        threshold: Union[float, str] = "default",
    ) -> Dict[str, Any]:
        """Compute SHAP feature attribution for an individual transaction.

        Args:
            transaction: Transaction dictionary or TransactionInput schema.
            top_n: Number of leading features to highlight in summary.
            apply_preprocessing: Whether to scale Amount and Time.
            generate_plot: Whether to construct an interactive Plotly figure.
            threshold: Operational decision threshold.

        Returns:
            Dictionary containing:
            - fraud_probability: float
            - is_fraud: bool
            - risk_category: str
            - base_value: float
            - top_positive_contributors: List[dict]
            - top_negative_contributors: List[dict]
            - all_contributions: List[dict]
            - evidence_summary: str
            - plot_dict: Optional[dict] (Plotly figure JSON representation)
            - plot_html: Optional[str] (Plotly standalone interactive HTML snippet)
        """
        # 1. Score transaction with the predictor
        prediction_result = self.predictor.predict_transaction(
            transaction=transaction,
            threshold=threshold,
            apply_preprocessing=apply_preprocessing,
        )

        # 2. Extract transaction dictionary for feature lookup
        if isinstance(transaction, TransactionInput):
            data_dict = transaction.model_dump()
        elif isinstance(transaction, pd.Series):
            data_dict = transaction.to_dict()
        else:
            data_dict = dict(transaction)

        # 3. Format preprocessed input matrix for SHAP explainer
        X = preprocess_input(
            data=data_dict,
            preprocessor=self.predictor.preprocessor,
            apply_scaling=apply_preprocessing,
            expected_features=self.predictor.feature_names,
        )

        # 4. Compute TreeSHAP values
        raw_shap = self.explainer.shap_values(X)
        if isinstance(raw_shap, list):
            # Binary classification list: pick class 1 (fraud)
            shap_vector = np.asarray(raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0], dtype=float)
        elif raw_shap.ndim == 2:
            shap_vector = np.asarray(raw_shap[0], dtype=float)
        else:
            shap_vector = np.asarray(raw_shap, dtype=float).flatten()

        feature_names = self.predictor.feature_names
        feature_vals = X[0]

        # 5. Build structured per-feature contributions
        contributions: List[Dict[str, Any]] = []
        for feat_name, feat_val, s_val in zip(feature_names, feature_vals, shap_vector):
            s_float = float(s_val)
            f_float = float(feat_val)
            evidence_desc = self._generate_statistical_evidence(
                feature_name=feat_name,
                feature_value=f_float,
                shap_value=s_float,
                raw_transaction=data_dict,
            )
            contributions.append({
                "feature": feat_name,
                "feature_value": round(f_float, 4),
                "shap_value": round(s_float, 4),
                "abs_shap": round(abs(s_float), 4),
                "direction": "INCREASES_FRAUD_RISK" if s_float > 0 else "DECREASES_FRAUD_RISK",
                "evidence": evidence_desc,
            })

        # Rank by absolute SHAP impact
        contributions_sorted = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)

        top_positive = [c for c in contributions_sorted if c["shap_value"] > 0][:top_n]
        top_negative = [c for c in contributions_sorted if c["shap_value"] < 0][:top_n]
        top_overall = contributions_sorted[:top_n]

        # 6. Synthesize factual evidence summary
        pos_names = [f"{c['feature']} ({c['shap_value']:+.2f})" for c in top_positive[:3]]
        neg_names = [f"{c['feature']} ({c['shap_value']:+.2f})" for c in top_negative[:2]]

        summary_parts = [
            f"Predicted fraud probability is {prediction_result['fraud_probability']:.4f} "
            f"(Classified: {'FRAUD' if prediction_result['is_fraud'] else 'LEGITIMATE'} at "
            f"threshold {prediction_result['threshold_used']}).",
        ]
        if top_positive:
            summary_parts.append(f"Primary features pushing towards fraud: {', '.join(pos_names)}.")
        if top_negative:
            summary_parts.append(f"Primary features mitigating fraud risk: {', '.join(neg_names)}.")
        summary_parts.append(
            "Note: Features V1-V28 are anonymized PCA components. Evidence represents statistical "
            "log-odds impact without unverified merchant or cardholder assumptions."
        )
        evidence_summary = " ".join(summary_parts)

        # 7. Generate Plotly figure if requested
        plot_dict = None
        plot_html = None
        fig = None
        if generate_plot:
            fig = self.create_shap_waterfall_plot(
                top_contributions=top_overall,
                base_value=self.base_value,
                fraud_prob=prediction_result["fraud_probability"],
                is_fraud=prediction_result["is_fraud"],
            )
            plot_dict = fig.to_dict()
            plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

        return {
            "fraud_probability": prediction_result["fraud_probability"],
            "is_fraud": prediction_result["is_fraud"],
            "risk_category": prediction_result["risk_category"],
            "threshold_used": prediction_result["threshold_used"],
            "threshold_name": prediction_result["threshold_name"],
            "base_value": round(self.base_value, 4),
            "top_features": top_overall,
            "top_positive_contributors": top_positive,
            "top_negative_contributors": top_negative,
            "all_contributions": contributions_sorted,
            "evidence_summary": evidence_summary,
            "figure": fig,
            "plot_dict": plot_dict,
            "plot_html": plot_html,
        }

    def create_shap_waterfall_plot(
        self,
        top_contributions: List[Dict[str, Any]],
        base_value: float,
        fraud_prob: float,
        is_fraud: bool,
    ) -> go.Figure:
        """Create an interactive horizontal bar chart of top SHAP feature contributions using Plotly.

        Positive contributions (increasing fraud risk) are colored crimson/red,
        while negative contributions (decreasing fraud risk) are colored cobalt/green.
        """
        # Reverse list so top contributor appears at the top of the horizontal bar chart
        ordered = list(reversed(top_contributions))
        feature_labels = [f"{c['feature']} ({c['feature_value']:+.2f})" for c in ordered]
        shap_values = [c["shap_value"] for c in ordered]
        colors = ["#ef4444" if val > 0 else "#10b981" for val in shap_values]

        hover_texts = [
            f"<b>Feature:</b> {c['feature']}<br>"
            f"<b>Value:</b> {c['feature_value']:+.4f}<br>"
            f"<b>SHAP Impact:</b> {c['shap_value']:+.4f}<br>"
            f"<b>Effect:</b> {c['direction']}"
            for c in ordered
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=feature_labels,
            x=shap_values,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            name="SHAP Value",
        ))

        status_text = "FRAUD" if is_fraud else "LEGITIMATE"
        status_color = "#ef4444" if is_fraud else "#10b981"

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>SHAP Prediction Explanation</b> | "
                    f"Fraud Probability: <b>{fraud_prob:.4f}</b> "
                    f"(<span style='color:{status_color}'>{status_text}</span>)"
                ),
                font=dict(size=14, color="#f8fafc"),
            ),
            xaxis=dict(
                title=dict(
                    text="SHAP Value (Impact on Model Log-Odds)",
                    font=dict(color="#cbd5e1", size=12),
                ),
                zeroline=True,
                zerolinecolor="#94a3b8",
                zerolinewidth=1.5,
                gridcolor="#334155",
                tickfont=dict(color="#cbd5e1"),
            ),
            yaxis=dict(
                title=dict(
                    text="Top Predictive Features",
                    font=dict(color="#cbd5e1", size=12),
                ),
                tickfont=dict(color="#cbd5e1"),
            ),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(family="Inter, sans-serif", color="#f8fafc"),
            margin=dict(l=140, r=40, t=60, b=50),
            height=420,
            showlegend=False,
        )

        return fig


# Cached singleton
_EXPLAINER_INSTANCE: Optional[FraudSHAPExplainer] = None


def get_explainer() -> FraudSHAPExplainer:
    """Retrieve or initialize the cached singleton FraudSHAPExplainer."""
    global _EXPLAINER_INSTANCE
    if _EXPLAINER_INSTANCE is None:
        _EXPLAINER_INSTANCE = FraudSHAPExplainer()
    return _EXPLAINER_INSTANCE
