"""FastAPI Application entry point for Credit Card Fraud Detection Inference Service."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.agent import router as agent_router
from src.api.routes.alerts import router as alerts_router
from src.api.routes.cases import router as cases_router
from src.api.routes.predict import router as predict_router
from src.cases.database import init_db
from src.inference.predictor import get_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up predictor and initialize database tables during application startup."""
    logger.info("Initializing database tables...")
    init_db()

    logger.info("Initializing FraudPredictor singleton during application startup...")
    predictor = get_predictor()
    logger.info(
        "Predictor warmed up successfully: Model='%s', Version='%s', Features=%d",
        predictor.model_name,
        predictor.model_version,
        len(predictor.feature_names),
    )
    yield
    logger.info("Shutting down Fraud Detection API service.")


app = FastAPI(
    title="Anti-Fraudster Inference Engine",
    description=(
        "Production-ready real-time XGBoost inference API for European cardholder transactions. "
        "Provides single/batch scoring, calibrated operational thresholds, configurable risk tiers, "
        "and strict schema validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web dashboards and external microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(predict_router)
app.include_router(alerts_router)
app.include_router(cases_router)
app.include_router(agent_router)


@app.get("/", tags=["General"])
def root():
    """Service root endpoint providing API status and documentation links."""
    return {
        "service": "Anti-Fraudster Inference Engine",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metadata": "/model/metadata",
        "alerts_queue": "GET /alerts",
        "cases_management": "GET /cases",
        "agent_investigate": "POST /cases/{case_id}/investigate",
        "agent_decision": "POST /cases/{case_id}/agent-decision",
        "agent_evaluate": "POST /agent/evaluate",
        "agent_tools_check": "GET /agent/tools/check",
        "predict_single": "POST /predict",
        "predict_batch": "POST /predict/batch",
        "predict_explain": "POST /predict/explain",
        "anomaly_score": "POST /anomaly/score",
        "evaluate_combined": "POST /evaluate/combined",
        "anomaly_comparison": "GET /anomaly/comparison",
    }
