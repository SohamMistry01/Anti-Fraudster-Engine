"""Anomaly detection package featuring Isolation Forest baseline and supervised model comparison."""

from src.anomaly.detector import AnomalyDetector, get_anomaly_detector
from src.anomaly.comparison import compare_models, generate_comparison_report

__all__ = [
    "AnomalyDetector",
    "get_anomaly_detector",
    "compare_models",
    "generate_comparison_report",
]
