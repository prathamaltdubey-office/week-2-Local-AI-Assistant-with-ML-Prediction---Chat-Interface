import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.logger import logger
from backend.model_loader import get_model
from backend.schemas import ChatRequest, Customer, PredictionResponse
from chatbot.chains import ChurnChatbot

app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI service for customer churn prediction and AI chatbot.",
    version="1.0.0",
)


# ============================================================
# REQUEST VALIDATION ERROR HANDLER
# ============================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle Pydantic/FastAPI request validation errors.
    """

    logger.error(
        "Request validation failed | " "method=%s | path=%s | errors=%s",
        request.method,
        request.url.path,
        exc.errors(),
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
        },
    )


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CHATBOT
# ============================================================

chatbot = ChurnChatbot()


# ============================================================
# ROOT ENDPOINT
# ============================================================


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    logger.info("Root endpoint requested")

    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/health")
def health() -> dict[str, str]:
    """Check whether the API is running."""

    logger.info("Health check requested")

    return {
        "status": "healthy",
    }


# ============================================================
# AVAILABLE MODELS
# ============================================================


@app.get("/models")
def available_models() -> dict[str, list[str]]:
    """Return the ML models available for prediction."""

    logger.info("Available models requested")

    return {
        "models": [
            "logistic_regression",
            "random_forest",
            "xgboost",
        ]
    }


# ============================================================
# CUSTOMER CHURN PREDICTION
# ============================================================


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: Customer,
    model_name: str = "random_forest",
) -> PredictionResponse:
    """
    Predict customer churn.

    Args:
        customer:
            Customer information validated by Pydantic.

        model_name:
            ML model to use for prediction.

    Returns:
        Prediction result containing churn probability and risk level.
    """

    logger.info(
        "Prediction request received | model=%s",
        model_name,
    )

    try:
        # ----------------------------------------------------
        # Load requested model
        # ----------------------------------------------------

        model = get_model(model_name)

        # ----------------------------------------------------
        # Convert Pydantic model to DataFrame
        # ----------------------------------------------------

        customer_data = pd.DataFrame([customer.model_dump()])

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = int(model.predict(customer_data)[0])

        probability = float(model.predict_proba(customer_data)[0][1])

        # ----------------------------------------------------
        # Determine risk level
        # ----------------------------------------------------

        if probability >= 0.7:
            risk_level = "High"

        elif probability >= 0.4:
            risk_level = "Medium"

        else:
            risk_level = "Low"

        logger.info(
            "Prediction completed | " "model=%s | prediction=%s | risk=%s",
            model_name,
            prediction,
            risk_level,
        )

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return PredictionResponse(
            prediction=prediction,
            churn_probability=probability,
            risk_level=risk_level,
            model=model_name,
        )

    except ValueError as exc:
        logger.error(
            "Invalid model requested | model=%s | error=%s",
            model_name,
            str(exc),
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Prediction failed | model=%s | error=%s",
            model_name,
            str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc


# ============================================================
# AI CHATBOT
# ============================================================


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """
    Chat with the local Llama 3 AI assistant.

    The chatbot uses Ollama + Llama 3 + LangChain.
    """

    logger.info("Chat request received")

    try:
        response = chatbot.chat(request.message)

        logger.info("Chat response generated")

        return response.model_dump()

    except Exception as exc:
        logger.exception(
            "Chat request failed | error=%s",
            str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="Chatbot request failed.",
        ) from exc
