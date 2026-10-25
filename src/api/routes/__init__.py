"""API routes package."""

from src.api.routes.alerts import router as alerts_router
from src.api.routes.cases import router as cases_router
from src.api.routes.predict import router as predict_router

__all__ = ["alerts_router", "cases_router", "predict_router"]
