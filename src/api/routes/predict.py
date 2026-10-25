"""FastAPI routing endpoints for fraud prediction, batch scoring, and model metadata."""

import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, Body, HTTPException, Query, status

from src.inference.predictor import get_predictor
from src.inference.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
    TransactionInput,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Fraud Detection & Scoring"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns the status of the inference service and loaded model metadata.",
)
def health_check() -> HealthResponse:
    """Verify inference engine readiness and model presence."""
    predictor = get_predictor()
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None,
        model_version=predictor.model_version,
        model_name=predictor.model_name,
        features_count=len(predictor.feature_names),
    )


@router.get(
    "/model/metadata",
    response_model=ModelMetadataResponse,
    summary="Model Metadata & Deployment Thresholds",
    description="Retrieve model performance metrics, calibrated operational thresholds, and top predictive features.",
)
def get_model_metadata() -> ModelMetadataResponse:
    """Return model architecture metadata, evaluation benchmarks, and operational parameters."""
    predictor = get_predictor()
    meta = predictor.metadata
    return ModelMetadataResponse(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        feature_names=predictor.feature_names,
        features_count=len(predictor.feature_names),
        evaluation_metrics=meta.get("evaluation_metrics", {}),
        operational_thresholds=meta.get("operational_thresholds", {}),
        cost_parameters=meta.get("cost_parameters", {}),
        top_5_predictive_features=meta.get("top_5_predictive_features", []),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score Single Transaction",
    description=(
        "Score a single credit card transaction. Accepts either a full PredictionRequest payload "
        "or a direct TransactionInput object."
    ),
)
def predict_single(
    payload: Union[PredictionRequest, TransactionInput] = Body(...),
    threshold: Optional[str] = Query(
        default=None,
        description="Override threshold ('default', 'cost_optimal', 'f1_optimal', 'f2_optimal', or float 0.0-1.0)",
    ),
    apply_preprocessing: Optional[bool] = Query(
        default=None,
        description="Override whether to apply RobustScaler preprocessing to Amount & Time",
    ),
) -> PredictionResponse:
    """Execute real-time fraud scoring for an individual transaction."""
    predictor = get_predictor()

    try:
        if isinstance(payload, PredictionRequest):
            tx_data = payload.transaction
            req_thresh = threshold if threshold is not None else payload.threshold
            req_prep = apply_preprocessing if apply_preprocessing is not None else payload.apply_preprocessing
            risk_levels = payload.risk_levels
        else:
            tx_data = payload
            req_thresh = threshold if threshold is not None else "default"
            req_prep = apply_preprocessing if apply_preprocessing is not None else True
            risk_levels = None

        result = predictor.predict_transaction(
            transaction=tx_data,
            threshold=req_thresh,
            risk_levels=risk_levels,
            apply_preprocessing=req_prep,
        )
        return PredictionResponse(**result)

    except ValueError as exc:
        logger.warning("Prediction validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Internal prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score Multiple Transactions (Batch)",
    description="Vectorized batch scoring for a list of transactions with configurable threshold.",
)
def predict_batch_endpoint(
    payload: Union[BatchPredictionRequest, List[TransactionInput]] = Body(...),
    threshold: Optional[str] = Query(
        default=None,
        description="Override threshold ('default', 'cost_optimal', 'f1_optimal', 'f2_optimal', or float 0.0-1.0)",
    ),
    apply_preprocessing: Optional[bool] = Query(
        default=None,
        description="Override whether to apply RobustScaler preprocessing to Amount & Time",
    ),
) -> BatchPredictionResponse:
    """Execute high-throughput batch scoring on multiple transactions."""
    predictor = get_predictor()

    try:
        if isinstance(payload, BatchPredictionRequest):
            tx_list = payload.transactions
            req_thresh = threshold if threshold is not None else payload.threshold
            req_prep = apply_preprocessing if apply_preprocessing is not None else payload.apply_preprocessing
            risk_levels = payload.risk_levels
        else:
            tx_list = payload
            req_thresh = threshold if threshold is not None else "default"
            req_prep = apply_preprocessing if apply_preprocessing is not None else True
            risk_levels = None

        result = predictor.predict_batch(
            transactions=tx_list,
            threshold=req_thresh,
            risk_levels=risk_levels,
            apply_preprocessing=req_prep,
        )
        return BatchPredictionResponse(**result)

    except ValueError as exc:
        logger.warning("Batch prediction validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Internal batch prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {exc}",
        ) from exc
