from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from chatbot.customer_extractor import extract_customer_data
from chatbot.customer_normalizer import normalize_customer_data
from chatbot.error_handler import (
    LLMError,
    PredictionError,
    handle_error,
)
from chatbot.llm import get_llm
from chatbot.memory import ConversationMemory
from chatbot.parser import ChatResponse
from chatbot.prediction_client import predict_customer
from chatbot.prediction_intent import is_prediction_request
from chatbot.prompts import SYSTEM_PROMPT

REQUIRED_CUSTOMER_FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class ChurnChatbot:
    """Customer churn chatbot with memory and ML prediction support."""

    def __init__(self) -> None:
        """Initialize the chatbot, LLM, memory, and customer data."""

        self.llm = get_llm()

        self.structured_llm = self.llm.with_structured_output(ChatResponse)

        self.memory = ConversationMemory()

        # Stores customer information collected across messages.
        self.customer_data: dict[str, Any] = {}

    # ============================================================
    # MESSAGE BUILDING
    # ============================================================

    def _build_messages(
        self,
        user_message: str,
    ) -> list[BaseMessage]:
        """Build the complete message history for the LLM."""

        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        messages.extend(self.memory.get_messages())

        messages.append(HumanMessage(content=user_message))

        return messages

    # ============================================================
    # CUSTOMER DATA
    # ============================================================

    def _update_customer_data(
        self,
        user_message: str,
    ) -> dict[str, Any]:
        """
        Extract and normalize customer information from a message.

        Only non-null information explicitly provided by the user
        is stored. Existing valid information is never overwritten
        by None.
        """

        extracted = extract_customer_data(user_message)

        if not extracted:
            return self.customer_data

        normalized = normalize_customer_data(extracted)

        # Only update fields that contain actual information.
        # This prevents the LLM from accidentally replacing
        # previously collected values with None.
        for key, value in normalized.items():
            if value is not None:
                self.customer_data[key] = value

        return self.customer_data

    # ============================================================
    # MISSING FIELDS
    # ============================================================

    def _missing_customer_fields(self) -> list[str]:
        """Return customer fields that have not been collected yet."""

        return [
            field
            for field in REQUIRED_CUSTOMER_FIELDS
            if self.customer_data.get(field) is None
        ]

    # ============================================================
    # PREDICTION
    # ============================================================

    def _predict(self) -> ChatResponse:
        """Run ML prediction or request missing customer information."""

        missing_fields = self._missing_customer_fields()

        if missing_fields:
            field_questions = {
                "gender": "What is your gender (Male/Female)?",
                "SeniorCitizen": "Are you a senior citizen (Yes/No)?",
                "Partner": "Do you have a partner (Yes/No)?",
                "Dependents": "Do you have dependents (Yes/No)?",
                "tenure": "How many months have you been with the company?",
                "PhoneService": "Do you have phone service (Yes/No)?",
                "MultipleLines": (
                    "Do you have multiple lines " "(Yes/No/No phone service)?"
                ),
                "InternetService": (
                    "Which internet service do you use " "(DSL/Fiber optic/No)?"
                ),
                "OnlineSecurity": (
                    "Do you have online security " "(Yes/No/No internet service)?"
                ),
                "OnlineBackup": (
                    "Do you have online backup " "(Yes/No/No internet service)?"
                ),
                "DeviceProtection": (
                    "Do you have device protection " "(Yes/No/No internet service)?"
                ),
                "TechSupport": (
                    "Do you have tech support " "(Yes/No/No internet service)?"
                ),
                "StreamingTV": (
                    "Do you have streaming TV " "(Yes/No/No internet service)?"
                ),
                "StreamingMovies": (
                    "Do you have streaming movies " "(Yes/No/No internet service)?"
                ),
                "Contract": (
                    "What type of contract do you have "
                    "(Month-to-month/One year/Two year)?"
                ),
                "PaperlessBilling": ("Do you use paperless billing (Yes/No)?"),
                "PaymentMethod": ("What payment method do you use?"),
                "MonthlyCharges": ("What is your monthly charge?"),
                "TotalCharges": ("What are your total charges?"),
            }

            questions = [
                field_questions[field]
                for field in missing_fields
                if field in field_questions
            ]

            # Avoid making the response unnecessarily huge.
            questions_to_show = questions[:5]

            answer = (
                "I can predict the customer's churn risk, but I still "
                "need a few more details.\n\n"
                + "\n".join(f"- {question}" for question in questions_to_show)
            )

            if len(questions) > 5:
                answer += (
                    "\n\nYou can provide these details in multiple "
                    "messages, or provide several of them together."
                )

            return ChatResponse(
                answer=answer,
                topic="customer churn prediction",
                confidence=1.0,
            )

        try:
            result = predict_customer(
                self.customer_data,
                model_name="random_forest",
            )

            prediction = result["prediction"]
            probability = result["churn_probability"]
            risk_level = result["risk_level"]
            model = result["model"]

            if prediction == 1:
                prediction_text = "likely to churn"
            else:
                prediction_text = "unlikely to churn"

            answer = (
                f"The {model.replace('_', ' ').title()} model predicts "
                f"that this customer is {prediction_text}. "
                f"The estimated churn probability is "
                f"{probability:.2%}, giving a {risk_level} risk level."
            )

            return ChatResponse(
                answer=answer,
                topic="churn prediction",
                confidence=1.0,
            )

        except Exception as exc:
            print(f"Prediction error: {exc}")

            error_response = handle_error(PredictionError(str(exc)))

            return ChatResponse.model_validate(error_response)

    # ============================================================
    # CHAT
    # ============================================================

    def chat(
        self,
        user_message: str,
    ) -> ChatResponse:
        """Generate a structured chatbot response."""

        # --------------------------------------------------------
        # First collect customer information from this message.
        # --------------------------------------------------------

        self._update_customer_data(user_message)

        # --------------------------------------------------------
        # Prediction request
        # --------------------------------------------------------

        if is_prediction_request(user_message):

            response = self._predict()

        else:

            # ----------------------------------------------------
            # Normal chatbot conversation
            # ----------------------------------------------------

            try:
                # Build system prompt + conversation memory
                # + current user message.
                messages = self._build_messages(user_message)

                response: Any = self.structured_llm.invoke(messages)

                if not isinstance(response, ChatResponse):
                    response = ChatResponse.model_validate(response)

            except Exception as exc:
                print(f"LLM error: {exc}")

                error_response = handle_error(LLMError(str(exc)))

                response = ChatResponse.model_validate(error_response)

        # --------------------------------------------------------
        # Store conversation memory.
        # --------------------------------------------------------

        self.memory.add_user_message(user_message)

        self.memory.add_ai_message(response.answer)

        return response


def stream_chat(self, user_message: str):
    """
    Stream a normal LLM response token by token.

    This method is intended for conversational responses.
    Structured prediction responses continue to use chat().
    """

    self._update_customer_data(user_message)

    if is_prediction_request(user_message):
        response = self._predict()

        yield response.answer

        self.memory.add_user_message(user_message)
        self.memory.add_ai_message(response.answer)

        return

    try:
        messages = self._build_messages(user_message)

        full_response = ""

        for chunk in self.llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        self.memory.add_user_message(user_message)
        self.memory.add_ai_message(full_response)

    except Exception as exc:
        print(f"LLM streaming error: {exc}")

        error_response = handle_error(LLMError(str(exc)))

        response = ChatResponse.model_validate(error_response)

        yield response.answer

        self.memory.add_user_message(user_message)
        self.memory.add_ai_message(response.answer)

    # ============================================================
    # MEMORY
    # ============================================================

    def clear_memory(self) -> None:
        """Clear conversation memory and customer information."""

        self.memory.clear()

        self.customer_data.clear()
