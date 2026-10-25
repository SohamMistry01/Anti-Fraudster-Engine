"""Integrated Interpretable Risk Assessor combining Supervised XGBoost, Unsupervised Isolation Forest,

and SHAP feature evidence without fabricating missing cardholder or merchant information.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.anomaly.detector import AnomalyDetector, get_anomaly_detector
from src.explainability.shap_explainer import FraudSHAPExplainer, get_explainer
from src.inference.predictor import FraudPredictor, get_predictor
from src.inference.schemas import TransactionInput

logger = logging.getLogger(__name__)


class IntegratedRiskAssessor:
    """Evaluates supervised and unsupervised risk signals separately before combining them.

    Provides multi-signal decision triage, evidence aggregation, and action recommendations.
    """

    def __init__(
        self,
        predictor: Optional[FraudPredictor] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        explainer: Optional[FraudSHAPExplainer] = None,
    ) -> None:
        """Initialize the integrated assessor with scoring, anomaly, and SHAP components."""
        self.predictor = predictor or get_predictor()
        self.anomaly_detector = anomaly_detector or get_anomaly_detector()
        self.explainer = explainer or get_explainer()

    def assess_transaction(
        self,
        transaction: Union[dict, TransactionInput, pd.Series],
        threshold: Union[float, str] = "default",
        apply_preprocessing: bool = True,
        top_n_features: int = 5,
        generate_plots: bool = False,
    ) -> Dict[str, Any]:
        """Perform comprehensive, interpretable multi-signal assessment for a transaction.

        1. Evaluates supervised classifier (XGBoost).
        2. Evaluates unsupervised anomaly detector (Isolation Forest).
        3. Extracts local SHAP evidence.
        4. Synthesizes a unified risk diagnosis and action recommendation.

        Args:
            transaction: Input transaction data.
            threshold: Operational threshold for XGBoost.
            apply_preprocessing: Whether to scale raw Amount and Time.
            top_n_features: Number of SHAP feature drivers to include.
            generate_plots: Whether to generate Plotly visualisations.

        Returns:
            Dictionary containing individual and combined risk evidence.
        """
        # Step 1: Supervised Model Evaluation
        supervised_res = self.predictor.predict_transaction(
            transaction=transaction,
            threshold=threshold,
            apply_preprocessing=apply_preprocessing,
        )

        # Step 2: Unsupervised Anomaly Detection
        anomaly_res = self.anomaly_detector.score_transaction(
            transaction=transaction,
            apply_preprocessing=apply_preprocessing,
        )

        # Step 3: SHAP Feature Attribution
        shap_res = self.explainer.explain_transaction(
            transaction=transaction,
            top_n=top_n_features,
            apply_preprocessing=apply_preprocessing,
            generate_plot=generate_plots,
            threshold=threshold,
        )

        # Step 4: Multi-Signal Synthesis & Operational Triage
        p_fraud = supervised_res["fraud_probability"]
        is_sup_fraud = supervised_res["is_fraud"]
        anom_score = anomaly_res["anomaly_score"]
        is_anom = anomaly_res["is_anomaly"]

        # Determine Joint Risk Quadrant
        if is_sup_fraud and is_anom:
            quadrant = "CONFIRMED_HIGH_RISK"
            action = "DECLINE_TRANSACTION"
            assessment_text = (
                f"HIGH-RISK CONSENSUS: Supervised model assigned high fraud probability ({p_fraud:.4f}) "
                f"and unsupervised Isolation Forest detected an extreme population anomaly "
                f"(score {anom_score:.2f}). Both independent signals agree on severe risk."
            )
        elif is_sup_fraud and not is_anom:
            quadrant = "SUBTLE_IN_DISTRIBUTION_FRAUD"
            action = "CHALLENGE_OR_REVIEW"
            assessment_text = (
                f"SUBTLE FRAUD RISK: Supervised model flagged transaction as fraud ({p_fraud:.4f}), "
                f"though Isolation Forest classifies it within normal distribution boundaries "
                f"(score {anom_score:.2f}). Characteristic of card-testing or stealthy fraud "
                f"matching multivariate fraud correlations without extreme outlier values."
            )
        elif not is_sup_fraud and is_anom:
            quadrant = "NOVEL_ANOMALY_OUTLIER"
            action = "STEP_UP_2FA"
            assessment_text = (
                f"UNUSUAL ANOMALY: Isolation Forest flagged this transaction as a rare statistical "
                f"outlier (score {anom_score:.2f}), but supervised XGBoost model indicates low match "
                f"to known historical fraud templates ({p_fraud:.4f}). May indicate novel fraud "
                f"mechanisms or rare legitimate activity."
            )
        else:
            quadrant = "CONFIRMED_LEGITIMATE"
            action = "APPROVE"
            assessment_text = (
                f"STANDARD LEGITIMATE: Low fraud probability ({p_fraud:.4f}) and normal population "
                f"distribution (anomaly score {anom_score:.2f}). Both independent models agree."
            )

        # Format Top Statistical Drivers
        top_pos = [
            f"{c['feature']} (value: {c['feature_value']:+.2f}, SHAP impact: {c['shap_value']:+.2f})"
            for c in shap_res["top_positive_contributors"][:3]
        ]
        top_neg = [
            f"{c['feature']} (value: {c['feature_value']:+.2f}, SHAP impact: {c['shap_value']:+.2f})"
            for c in shap_res["top_negative_contributors"][:3]
        ]

        evidence_breakdown = {
            "fraud_driving_factors": top_pos,
            "legitimate_mitigating_factors": top_neg,
            "integrity_notice": (
                "Features V1-V28 are anonymized PCA components. Evidence represents purely statistical "
                "deviations and log-odds contributions without fabricated cardholder or merchant attributes."
            ),
        }

        return {
            "assessment_summary": assessment_text,
            "operational_quadrant": quadrant,
            "recommended_action": action,
            "supervised_signal": {
                "fraud_probability": supervised_res["fraud_probability"],
                "is_fraud": supervised_res["is_fraud"],
                "risk_category": supervised_res["risk_category"],
                "threshold_used": supervised_res["threshold_used"],
                "model_version": supervised_res["model_version"],
            },
            "unsupervised_anomaly_signal": {
                "anomaly_score": anomaly_res["anomaly_score"],
                "raw_decision_score": anomaly_res["raw_score"],
                "is_anomaly": anomaly_res["is_anomaly"],
                "anomaly_tier": anomaly_res["anomaly_tier"],
                "algorithm": anomaly_res["algorithm"],
            },
            "shap_explanation": {
                "base_value": shap_res["base_value"],
                "top_positive_contributors": shap_res["top_positive_contributors"],
                "top_negative_contributors": shap_res["top_negative_contributors"],
                "evidence_breakdown": evidence_breakdown,
                "evidence_summary": shap_res.get("evidence_summary"),
            },
            "plots": {
                "shap_plot_fig": shap_res.get("figure"),
                "shap_plot_dict": shap_res.get("plot_dict"),
                "shap_plot_html": shap_res.get("plot_html"),
            },
        }


# Singleton
_ASSESSOR_INSTANCE: Optional[IntegratedRiskAssessor] = None


def get_risk_assessor() -> IntegratedRiskAssessor:
    """Retrieve or initialize the cached singleton IntegratedRiskAssessor."""
    global _ASSESSOR_INSTANCE
    if _ASSESSOR_INSTANCE is None:
        _ASSESSOR_INSTANCE = IntegratedRiskAssessor()
    return _ASSESSOR_INSTANCE
