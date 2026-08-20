# Customer Churn AI

### Local AI Assistant with ML Prediction & Chat Interface

An end-to-end Machine Learning and GenAI application that combines:

* Machine Learning
* MLOps
* FastAPI
* Pydantic
* Structured Logging
* Podman containerization
* Ollama
* Llama 3
* LangChain
* React
* REST APIs

The application predicts customer churn using trained ML models and provides a local AI chatbot that can answer questions and assist with churn prediction.

---

# 1. Project Overview

The project contains two major capabilities.

## Machine Learning Prediction

A user provides customer information through the React frontend.

The data is sent to FastAPI.

FastAPI:

1. Validates the request using Pydantic.
2. Loads the selected ML model.
3. Converts the request into a DataFrame.
4. Runs the saved ML pipeline.
5. Calculates churn probability.
6. Determines the risk level.
7. Returns the prediction to React.

Flow:

```text
React
   |
   | POST /predict
   v
FastAPI
   |
   v
Pydantic Validation
   |
   v
Model Loader
   |
   v
Saved ML Pipeline
   |
   v
Prediction
   |
   v
Churn Probability
   |
   v
Risk Level
   |
   v
React
```

---

# 2. AI Chatbot

The application also provides a local AI chatbot.

The chatbot uses:

* Ollama
* Llama 3 8B
* LangChain
* Conversation memory
* Structured responses
* Customer information extraction
* ML prediction integration

Normal chatbot flow:

```text
React
   |
   | POST /chat
   v
FastAPI
   |
   v
ChurnChatbot
   |
   v
LangChain
   |
   v
Ollama
   |
   v
Llama 3
   |
   v
Structured Response
   |
   v
React
```

The chatbot can also collect customer information and call the ML prediction API.

```text
User
 |
 v
React
 |
 v
POST /chat
 |
 v
ChurnChatbot
 |
 v
Customer Data Extraction
 |
 v
Customer Data Normalization
 |
 v
Check Missing Fields
 |
 v
POST /predict
 |
 v
FastAPI
 |
 v
ML Model
 |
 v
Prediction
 |
 v
Chatbot Response
 |
 v
React
```

---

# 3. Technology Stack

## Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib

## MLOps / Backend

* FastAPI
* Uvicorn
* Pydantic
* Python Logging
* REST API
* Podman

## GenAI

* Ollama
* Llama 3 8B
* LangChain
* LangChain Ollama

## Frontend

* React
* Vite
* JavaScript
* Fetch API
* CSS

---

# 4. Project Structure

```text
customer-churn-prediction-main/
│
├── backend/
│   ├── api.py
│   ├── config.py
│   ├── logger.py
│   ├── model_loader.py
│   ├── schemas.py
│   └── utils.py
│
├── chatbot/
│   ├── chains.py
│   ├── customer_data.py
│   ├── customer_extractor.py
│   ├── customer_normalizer.py
│   ├── error_handler.py
│   ├── llm.py
│   ├── memory.py
│   ├── parser.py
│   ├── prediction_client.py
│   ├── prediction_intent.py
│   └── prompts.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── chat/
│   │   │   └── prediction/
│   │   │
│   │   ├── hooks/
│   │   │   ├── useChat.js
│   │   │   └── usePrediction.js
│   │   │
│   │   ├── services/
│   │   │   └── api.js
│   │   │
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   ├── best_logistic_model.pkl
│   ├── best_rf_model.pkl
│   └── best_xgb_model.pkl
│
├── logs/
│   ├── app.log
│   └── error.log
│
├── src/
│   ├── train.py
│   ├── tuning.py
│   └── evaluate.py
│
├── tests/
│
├── docs/
│   └── api-integration.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 5. ML Models

The application supports three trained models:

```text
Logistic Regression
Random Forest
XGBoost
```

Saved models:

```text
models/best_logistic_model.pkl
models/best_rf_model.pkl
models/best_xgb_model.pkl
```

The models contain the complete Scikit-learn pipeline.

This means preprocessing and prediction remain together.

---

# 6. FastAPI Backend

The backend application is:

```text
backend/api.py
```

The application is created using:

```python
app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI service for customer churn prediction and AI chatbot.",
    version="1.0.0",
)
```

---

# 7. Starting FastAPI

From the project root:

```powershell
python -m uvicorn backend.api:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI specification:

```text
http://127.0.0.1:8000/openapi.json
```

---

# 8. REST API Endpoints

The application exposes:

| Method | Endpoint   | Purpose                   |
| ------ | ---------- | ------------------------- |
| GET    | `/`        | API information           |
| GET    | `/health`  | Health check              |
| GET    | `/models`  | Available ML models       |
| POST   | `/predict` | Customer churn prediction |
| POST   | `/chat`    | AI chatbot                |

---

# 9. Root Endpoint

```http
GET /
```

Example response:

```json
{
  "message": "Customer Churn Prediction API",
  "version": "1.0.0",
  "status": "running"
}
```

---

# 10. Health Endpoint

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

This can be used to check whether the API is running.

---

# 11. Models Endpoint

```http
GET /models
```

Response:

```json
{
  "models": [
    "logistic_regression",
    "random_forest",
    "xgboost"
  ]
}
```

---

# 12. Prediction Endpoint

```http
POST /predict
```

The model can be selected using:

```text
model_name
```

Example:

```text
POST /predict?model_name=random_forest
```

Supported models:

```text
logistic_regression
random_forest
xgboost
```

Default:

```text
random_forest
```

---

# 13. Prediction Request

Example:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 80.5,
  "TotalCharges": 966.0
}
```

---

# 14. Pydantic Validation

Customer input is validated using:

```text
backend/schemas.py
```

Pydantic prevents invalid data from reaching the ML model.

Examples:

```python
gender: Literal["Male", "Female"]
```

```python
SeniorCitizen: Literal[0, 1]
```

```python
tenure: int = Field(ge=0, le=72)
```

```python
MonthlyCharges: float = Field(ge=0)
```

```python
TotalCharges: float = Field(ge=0)
```

If invalid data is sent, FastAPI automatically returns:

```text
422 Unprocessable Content
```

For example, an invalid value or missing required field can generate a 422 response.

---

# 15. Prediction Response

Example:

```json
{
  "prediction": 0,
  "churn_probability": 0.2938,
  "risk_level": "Low",
  "model": "random_forest"
}
```

Meaning:

```text
prediction = 0
```

means the customer is predicted not to churn.

```text
prediction = 1
```

means the customer is predicted to churn.

---

# 16. Risk Classification

The backend converts probability into a business-friendly risk level.

```text
Probability >= 0.70
        |
        v
High Risk
```

```text
0.40 <= Probability < 0.70
        |
        v
Medium Risk
```

```text
Probability < 0.40
        |
        v
Low Risk
```

Example:

```text
Churn probability = 0.2938

0.2938 < 0.40

Risk = Low
```

---

# 17. Model Loading

Model loading is implemented in:

```text
backend/model_loader.py
```

The mapping is:

```python
MODEL_FILES = {
    "logistic_regression": "best_logistic_model.pkl",
    "random_forest": "best_rf_model.pkl",
    "xgboost": "best_xgb_model.pkl",
}
```

The model loader:

1. Checks whether the requested model name is valid.
2. Checks whether the model file exists.
3. Loads the model using Joblib.
4. Verifies that the loaded object is a Scikit-learn Pipeline.

---

# 18. Prediction Processing

The complete processing flow is:

```text
JSON Request
     |
     v
Pydantic
     |
     v
Validated Customer
     |
     v
Pandas DataFrame
     |
     v
Saved ML Pipeline
     |
     v
model.predict()
     |
     v
model.predict_proba()
     |
     v
Risk Classification
     |
     v
PredictionResponse
```

---

# 19. Structured Logging

Logging is implemented in:

```text
backend/logger.py
```

The logger writes messages to:

```text
logs/app.log
```

and displays them in the terminal.

Example:

```text
2026-08-20 16:43:43 | INFO | customer_churn_api | Prediction request received
2026-08-20 16:43:44 | INFO | customer_churn_api | Prediction completed
```

The logging format contains:

```text
Timestamp
|
Log Level
|
Logger Name
|
Message
```

Example:

```text
2026-08-20 16:43:43 | INFO | customer_churn_api | Prediction request received
```

---

# 20. Error Logging

Application errors should be separated from normal application logs.

The recommended log structure is:

```text
logs/
├── app.log
└── error.log
```

`app.log` contains normal application events.

Example:

```text
Prediction request received
Prediction completed
Chat request received
Chat response generated
```

`error.log` contains errors and tracebacks.

Example:

```text
2026-08-20 16:50:10 | ERROR | customer_churn_api | Prediction failed
```

For unexpected exceptions, the application uses:

```python
logger.exception("Prediction failed")
```

`logger.exception()` is useful because it records the error together with the traceback.

---

# 21. Important Logging Design

The backend should have two handlers:

```text
Console Handler
      |
      v
Terminal

File Handler
      |
      v
logs/app.log
```

For errors:

```text
Error Handler
      |
      v
logs/error.log
```

Therefore:

```text
Application
     |
     +-----------> Terminal
     |
     +-----------> logs/app.log
     |
     +-----------> logs/error.log
```

---

# 22. Why Error Logs Are Important

Suppose the backend receives:

```text
POST /predict
```

but the model crashes.

The user should receive a safe API response such as:

```json
{
  "detail": "Prediction failed."
}
```

The actual technical error should be recorded in:

```text
logs/error.log
```

This prevents exposing internal implementation details to the frontend while still giving developers enough information to debug the problem.

---

# 23. FastAPI Error Handling

Prediction errors are handled using exceptions.

Invalid model:

```text
HTTP 400
```

Example:

```json
{
  "detail": "Unknown model: invalid_model"
}
```

Invalid Pydantic input:

```text
HTTP 422
```

Unexpected server error:

```text
HTTP 500
```

Example:

```json
{
  "detail": "Prediction failed."
}
```

---

# 24. Chatbot Error Handling

The chatbot also handles errors.

For example:

```text
React
 |
 v
POST /chat
 |
 v
FastAPI
 |
 v
Ollama
 |
 X
LLM error
```

The error is logged and the API returns a controlled response.

The backend uses:

```python
logger.exception("Chat request failed")
```

This records the traceback in the error log.

---

# 25. Why My 422 Error Was Not in error.log

A very important distinction:

A Pydantic validation error occurs **before the `/predict` function executes**.

Therefore this:

```python
logger.info("Prediction request received")
```

may never execute.

FastAPI validates:

```text
Request
   |
   v
Pydantic
   |
   X Invalid
```

before entering:

```python
def predict(...):
```

Therefore a simple endpoint-level `try/except` will not automatically capture every 422 validation error.

To log all validation failures, a FastAPI exception handler can be added for:

```text
RequestValidationError
```

This is the proper production-style approach if validation failures need to appear in `error.log`.

---

# 26. Ollama

The chatbot uses Ollama for local LLM inference.

Configured model:

```text
llama3:8b
```

Ollama runs locally at:

```text
http://localhost:11434
```

The application therefore does not need an external LLM API for chatbot inference.

---

# 27. Check Ollama

Check installed models:

```powershell
ollama list
```

The expected model should include:

```text
llama3:8b
```

If the model is not installed:

```powershell
ollama pull llama3:8b
```

Run the model:

```powershell
ollama run llama3:8b
```

---

# 28. LangChain + Ollama

The LLM configuration is located at:

```text
chatbot/llm.py
```

The application uses:

```python
from langchain_ollama import ChatOllama
```

The LLM is configured as:

```python
ChatOllama(
    model="llama3:8b",
    temperature=0.2,
    base_url="http://localhost:11434",
)
```

---

# 29. Prompt Templates

The chatbot uses a system prompt located in:

```text
chatbot/prompts.py
```

The prompt defines how the AI assistant should behave.

The application combines:

```text
System Prompt
+
Conversation Memory
+
Current User Message
```

before sending the request to the LLM.

---

# 30. Conversation Memory

Conversation memory is implemented in:

```text
chatbot/memory.py
```

The chatbot stores:

```text
User Message
AI Response
```

Example:

```text
User:
What is churn?

AI:
Churn means a customer leaves a service.
```

The next request can use previous messages to maintain context.

---

# 31. Structured JSON Responses

The chatbot uses a Pydantic model:

```text
ChatResponse
```

The response contains fields such as:

```text
answer
topic
confidence
prediction
churn_probability
risk_level
model
```

This makes the response predictable for React.

Example:

```json
{
  "answer": "The customer has a low churn risk.",
  "topic": "churn prediction",
  "confidence": 1.0,
  "prediction": 0,
  "churn_probability": 0.2938,
  "risk_level": "Low",
  "model": "random_forest"
}
```

---

# 32. Response Streaming

The project also contains LLM streaming support in:

```text
chatbot/llm.py
```

The streaming function uses:

```python
llm.stream(message)
```

Example:

```python
def stream_response(message: str):
    llm = get_llm()

    for chunk in llm.stream(message):
        if chunk.content:
            yield chunk.content
```

This means the LLM can produce output incrementally instead of waiting for the complete response.

Conceptually:

```text
Normal:

Request
   |
   v
Wait
   |
   v
Complete Response


Streaming:

Request
   |
   v
Token 1
   |
   v
Token 2
   |
   v
Token 3
   |
   v
...
```

## Important

The current React `/chat` integration uses:

```text
POST /chat
```

and waits for the complete JSON response.

Therefore, the existence of:

```python
llm.stream()
```

means **LLM streaming capability exists**, but it does not automatically mean the browser UI is currently displaying tokens one-by-one.

True end-to-end UI streaming would require a streaming FastAPI endpoint and frontend streaming handling.

---

# 33. Chatbot Architecture

Main chatbot:

```text
chatbot/chains.py
```

Important components:

```text
chains.py
customer_extractor.py
customer_normalizer.py
error_handler.py
llm.py
memory.py
parser.py
prediction_client.py
prediction_intent.py
prompts.py
```

The main class is:

```python
ChurnChatbot
```

It manages:

```text
LLM
Memory
Customer Data
Structured Output
Prediction
```

---

# 34. Customer Information Extraction

When a user wants a churn prediction, the chatbot collects customer information.

Example:

```text
User:
I am a senior citizen and I have been with the company for 12 months.
```

The chatbot extracts customer information.

The data is then normalized.

Files:

```text
chatbot/customer_extractor.py
chatbot/customer_normalizer.py
```

The collected information is stored in:

```python
self.customer_data
```

---

# 35. Missing Customer Information

The chatbot checks required fields.

If information is missing, it asks the user for additional information.

Example:

```text
I can predict the customer's churn risk, but I still need a few more details.

- What is your gender?
- Do you have a partner?
- How many months have you been with the company?
```

The chatbot can collect information across multiple messages.

---

# 36. Prediction Through Chat

Once all required fields are available:

```text
Chatbot
   |
   v
prediction_client.py
   |
   v
POST /predict
   |
   v
FastAPI
   |
   v
Random Forest
   |
   v
Prediction
   |
   v
Chatbot
   |
   v
Natural Language Response
```

The prediction client uses:

```text
chatbot/prediction_client.py
```

---

# 37. React Frontend

The frontend is implemented using:

```text
React
+
Vite
+
JavaScript
```

The frontend communicates with FastAPI using the browser Fetch API.

The API service is:

```text
frontend/src/services/api.js
```

Base URL:

```javascript
const API_BASE_URL = "http://localhost:8000";
```

---

# 38. React Chat Integration

The frontend sends:

```http
POST /chat
```

Example:

```json
{
  "message": "What is customer churn?"
}
```

The function is:

```javascript
sendChatMessage(message)
```

The Fetch API sends the request to FastAPI.

---

# 39. React Chat History

Chat history is maintained in:

```text
frontend/src/hooks/useChat.js
```

The application maintains:

```javascript
const [messages, setMessages] = useState([
  initialMessage,
]);
```

When the user sends a message:

```text
User message
   |
   v
messages
   |
   v
API request
   |
   v
AI response
   |
   v
messages
```

The chat interface therefore displays previous user and assistant messages.

---

# 40. React Loading Indicator

The chat hook maintains:

```javascript
const [loading, setLoading] = useState(false);
```

When an API request starts:

```javascript
setLoading(true);
```

After the request finishes:

```javascript
setLoading(false);
```

The UI uses this state to display a loading indicator and prevent duplicate submissions.

---

# 41. React Error Handling

The API helper checks:

```javascript
if (!response.ok)
```

If FastAPI returns an error, React attempts to read:

```text
detail
```

or:

```text
message
```

and converts it into a JavaScript error.

This allows the UI to show an appropriate error message.

---

# 42. CORS

FastAPI uses:

```text
CORSMiddleware
```

The development React application runs on:

```text
http://localhost:5173
```

FastAPI allows this origin so the browser can communicate with the backend.

---

# 43. Running React

Open another terminal.

From the project root:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm install
```

Start React:

```powershell
npm run dev
```

The frontend normally runs at:

```text
http://localhost:5173
```

---

# 44. Podman Containerization

The inference API is containerized using Podman.

Podman is used as a container engine.

The container packages the application and its runtime dependencies so the API can run consistently.

Conceptually:

```text
Python Application
+
Dependencies
+
Models
+
FastAPI
+
Uvicorn
        |
        v
Container Image
        |
        v
Podman Container
```

---

# 45. Why Podman

The goal of containerization is reproducibility.

Without a container:

```text
Machine
 |
 +-- Python version
 +-- Packages
 +-- Dependencies
 +-- Configuration
```

Different machines may behave differently.

With a container:

```text
Container
 |
 +-- Python
 +-- Dependencies
 +-- Application
 +-- Configuration
```

The environment becomes more consistent.

---

# 46. Podman Image

Example image:

```text
customer-churn-api:1.0
```

Build:

```powershell
podman build -t customer-churn-api:1.0 .
```

Check images:

```powershell
podman images
```

---

# 47. Run the Podman Container

Example:

```powershell
podman run -d `
  --name customer-churn-api `
  -p 8000:8000 `
  customer-churn-api:1.0
```

The API becomes available at:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# 48. Check Podman Container

Run:

```powershell
podman ps
```

Example:

```text
CONTAINER ID
IMAGE
COMMAND
STATUS
PORTS
NAMES
```

To see stopped containers:

```powershell
podman ps -a
```

---

# 49. Podman Logs

View container logs:

```powershell
podman logs customer-churn-api
```

Follow logs live:

```powershell
podman logs -f customer-churn-api
```

This is different from the application's own log files.

There are therefore two levels of logging:

```text
Podman
   |
   v
Container stdout/stderr
```

and:

```text
Application
   |
   +----> logs/app.log
   |
   +----> logs/error.log
```

---

# 50. Stop Podman Container

```powershell
podman stop customer-churn-api
```

Remove it:

```powershell
podman rm customer-churn-api
```

---

# 51. API Testing

The API can be tested using:

* Swagger
* Postman
* Bruno
* PowerShell

Swagger is automatically available through FastAPI.

```text
http://127.0.0.1:8000/docs
```

---

# 52. Test Health

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected:

```text
status
------
healthy
```

---

# 53. Test Models

```powershell
Invoke-RestMethod http://127.0.0.1:8000/models
```

Expected:

```json
{
  "models": [
    "logistic_regression",
    "random_forest",
    "xgboost"
  ]
}
```

---

# 54. Test Prediction

Use Swagger:

```text
http://127.0.0.1:8000/docs
```

Select:

```text
POST /predict
```

Click:

```text
Try it out
```

Select:

```text
random_forest
```

Provide customer JSON.

Click:

```text
Execute
```

The response should contain:

```text
prediction
churn_probability
risk_level
model
```

---

# 55. Test Chatbot

Start Ollama.

Then start FastAPI:

```powershell
python -m uvicorn backend.api:app --reload
```

Start React:

```powershell
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

Ask:

```text
What is customer churn?
```

The request travels through:

```text
React
 ↓
FastAPI
 ↓
ChurnChatbot
 ↓
LangChain
 ↓
Ollama
 ↓
Llama 3
 ↓
FastAPI
 ↓
React
```

---

# 56. Complete Development Startup

For local development, the main components are:

## Terminal 1 — Ollama

Ensure Ollama is running and the model exists:

```powershell
ollama list
```

If required:

```powershell
ollama run llama3:8b
```

## Terminal 2 — FastAPI

From project root:

```powershell
python -m uvicorn backend.api:app --reload
```

## Terminal 3 — React

```powershell
cd frontend
npm run dev
```

Then open:

```text
http://localhost:5173
```

---

# 57. Prediction Without Ollama

The ML prediction API does not require Ollama.

This works independently:

```text
React
 ↓
FastAPI
 ↓
Pydantic
 ↓
ML Model
 ↓
Prediction
```

Therefore, if Ollama is unavailable:

```text
Prediction API
```

can still work.

However, the chatbot requires:

```text
Ollama
+
Llama 3
```

---

# 58. Backend Dependency

If FastAPI is not running:

```text
React
   |
   X
FastAPI unavailable
```

The React application itself may still open, but API-based features will fail.

For example:

```text
Chat
Prediction
```

will show an API connection error.

Therefore:

```text
React
+
FastAPI
```

are both required for the complete application.

---

# 59. Logs Directory

The application creates:

```text
logs/
```

The recommended structure is:

```text
logs/
├── app.log
└── error.log
```

Example `app.log`:

```text
2026-08-20 16:43:43 | INFO | customer_churn_api | Prediction request received
2026-08-20 16:43:44 | INFO | customer_churn_api | Prediction completed
2026-08-20 16:45:10 | INFO | customer_churn_api | Chat request received
2026-08-20 16:45:15 | INFO | customer_churn_api | Chat response generated
```

Example `error.log`:

```text
2026-08-20 16:50:10 | ERROR | customer_churn_api | Prediction failed
```

Unexpected errors should also include their traceback.

---

# 60. What Should Be Ignored by Git

Generated logs should normally not be committed to Git.

Add to `.gitignore`:

```text
logs/
*.log
__pycache__/
*.pyc
.venv/
node_modules/
dist/
.env
```

This prevents runtime-generated files from being tracked unnecessarily.

---

# 61. API Integration Documentation

Detailed API integration documentation is stored in:

```text
docs/api-integration.md
```

It explains:

```text
React
 ↓
FastAPI
 ↓
Pydantic
 ↓
ML Model
```

and:

```text
React
 ↓
FastAPI
 ↓
LangChain
 ↓
Ollama
 ↓
Llama 3
```

---

# 62. Error Flow

The complete error-handling architecture is:

```text
User Request
     |
     v
React
     |
     v
FastAPI
     |
     v
Pydantic Validation
     |
     +------ Invalid ------> 422
     |
     v
Business Logic
     |
     +------ Invalid Model ------> 400
     |
     v
ML / LLM Processing
     |
     +------ Unexpected Error ------> 500
     |
     v
Response
```

Errors are:

```text
Returned to frontend
+
Logged by backend
```

---

# 63. Logging Flow

```text
API Request
     |
     v
FastAPI
     |
     v
Logger
     |
     +-------------> Terminal
     |
     +-------------> logs/app.log
     |
     +-------------> logs/error.log
```

Normal events:

```text
INFO
```

Problems:

```text
ERROR
```

Unexpected exceptions:

```text
ERROR + traceback
```

---

# 64. End-to-End Architecture

```text
                         CUSTOMER CHURN AI
                                |
                +---------------+---------------+
                |                               |
                v                               v
         React Prediction                  React Chat
                |                               |
                | POST /predict                 | POST /chat
                |                               |
                +---------------+---------------+
                                |
                                v
                         FastAPI Backend
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
       Prediction API                         Chatbot
              |                                   |
              v                         +---------+---------+
       Pydantic Validation              |                   |
              |                         v                   v
              v                     LangChain          Prediction
        Model Loader                     |                   |
              |                         v                   |
              v                      Ollama                |
      Saved ML Pipeline                 |                   |
              |                      Llama 3                |
              v                                             |
       ML Prediction                                         |
              |                                             |
              +--------------------+------------------------+
                                   |
                                   v
                             JSON Response
                                   |
                                   v
                             React Frontend
```

Logging exists around the backend:

```text
                         FastAPI
                            |
                            v
                         Logger
                       /    |    \
                      /     |     \
                     v      v      v
                Terminal  app.log  error.log
```

---

# 65. Project Responsibilities

## React

Responsible for:

* User interface
* Prediction form
* Chat interface
* Chat history
* Loading indicators
* API requests
* Displaying responses
* Displaying errors

## FastAPI

Responsible for:

* REST APIs
* Request validation
* Model loading
* ML inference
* Risk classification
* Chatbot integration
* Error handling
* Logging

## ML Pipeline

Responsible for:

* Preprocessing
* Feature transformation
* Model inference
* Churn prediction
* Probability calculation

## Chatbot

Responsible for:

* Natural language interaction
* Prompt handling
* Conversation memory
* Customer extraction
* Customer normalization
* Prediction intent detection
* Llama 3 interaction
* Structured responses

## Ollama

Responsible for:

* Running the local LLM

## Podman

Responsible for:

* Packaging the FastAPI service
* Running the backend in a container
* Providing a reproducible runtime environment

---

# 66. Week 2 Task Completion

## MLOps

| Task                                  | Status                |
| ------------------------------------- | --------------------- |
| FastAPI inference service             | Completed             |
| REST prediction endpoint              | Completed             |
| Model version/model selection loading | Completed             |
| Pydantic validation                   | Completed             |
| Structured logging                    | Completed             |
| Error logging                         | Completed             |
| Containerization                      | Completed with Podman |
| API testing                           | Completed             |

## GenAI

| Task                               | Status                    |
| ---------------------------------- | ------------------------- |
| Ollama                             | Completed                 |
| Local open-source LLM              | Completed                 |
| Llama 3                            | Completed                 |
| LangChain chatbot                  | Completed                 |
| Prompt templates                   | Completed                 |
| Conversation memory                | Completed                 |
| Structured JSON responses          | Completed                 |
| Error handling                     | Completed                 |
| LLM streaming capability           | Implemented               |
| End-to-end browser token streaming | Not currently implemented |

## React

| Task                  | Status    |
| --------------------- | --------- |
| React chat interface  | Completed |
| Chat history          | Completed |
| Loading indicator     | Completed |
| Fetch API integration | Completed |
| FastAPI integration   | Completed |
| Error display         | Completed |

---

# 67. Important Interview Explanation

If an interviewer asks:

## "How does your application work?"

Answer:

> My application is an end-to-end customer churn AI system. The React frontend communicates with a FastAPI backend through REST APIs. For ML prediction, FastAPI validates customer data using Pydantic, loads a saved Scikit-learn pipeline, performs inference, calculates churn probability, and returns a risk level. For GenAI, the backend uses LangChain with a locally running Llama 3 model through Ollama. The chatbot maintains conversation memory and can also collect customer information and call the ML prediction API. I added structured logging and error handling for observability, and I containerized the FastAPI service using Podman.

---

# 68. Why FastAPI?

FastAPI provides:

* High-performance REST APIs
* Automatic OpenAPI documentation
* Pydantic validation
* Easy integration with Python ML models
* Async support
* Simple deployment

It is suitable for serving ML models because the model and preprocessing pipeline are already available in Python.

---

# 69. Why Pydantic?

Pydantic validates incoming API data before it reaches the ML model.

For example:

```text
MonthlyCharges = -50
```

can be rejected before inference.

This improves:

* Data quality
* Reliability
* Error handling
* API safety

---

# 70. Why Structured Logging?

Instead of using:

```python
print("something happened")
```

the application uses:

```python
logger.info("Prediction completed")
```

and:

```python
logger.exception("Prediction failed")
```

This provides:

* Timestamp
* Severity
* Logger name
* Message
* Traceback for exceptions

It is much better for debugging production services.

---

# 71. Why Containerization?

Containerization makes the application environment reproducible.

The container includes:

```text
Application
+
Python
+
Dependencies
+
Configuration
```

Podman can then run the same image consistently across environments.

---

# 72. Why Ollama?

Ollama allows the application to run an open-source LLM locally.

Advantages:

* No external LLM API required
* Local inference
* Better control over data
* Easy local development
* Supports open-source models

---

# 73. Why LangChain?

LangChain provides application-level abstractions around the LLM.

In this project it helps with:

```text
Prompt
+
Messages
+
Memory
+
Structured Output
+
Ollama
```

The LLM itself generates language, while LangChain helps organize the application around it.

---

# 74. Why React?

React provides the interactive frontend.

It manages:

```text
Chat
Prediction Form
Chat History
Loading State
API Responses
Errors
```

FastAPI handles backend logic while React handles presentation.

---

# 75. Final Project Flow

The complete application can be summarized as:

```text
                    USER
                      |
                      v
                    REACT
                 /         \
                /           \
               v             v
          /predict          /chat
              |               |
              v               v
           FASTAPI         CHATBOT
              |               |
              v          LANGCHAIN
          PYDANTIC             |
              |              OLLAMA
              v                |
        MODEL LOADER         LLAMA 3
              |                |
              v                |
        ML PIPELINE            |
              |                |
              v                |
         PREDICTION <----------+
              |
              v
        RISK CLASSIFICATION
              |
              v
          JSON RESPONSE
              |
              v
             REACT
              |
              v
             USER
```

Observability:

```text
FastAPI
   |
   v
Logger
   |
   +----> Terminal
   |
   +----> logs/app.log
   |
   +----> logs/error.log
```

Deployment:

```text
FastAPI Application
        |
        v
   Container Image
        |
        v
      Podman
        |
        v
Running API Container
```

---

# 76. Final Expected Outputs

The project delivers:

```text
✓ FastAPI prediction service

✓ REST prediction APIs

✓ Pydantic request validation

✓ ML model loading

✓ Structured logging

✓ Error logging

✓ Local Ollama AI chatbot

✓ Llama 3 integration

✓ LangChain chatbot

✓ Prompt templates

✓ Conversation memory

✓ Structured JSON responses

✓ LLM streaming capability

✓ React chat interface

✓ Chat history

✓ Loading indicators

✓ React-FastAPI integration

✓ Podman containerized inference API

✓ Swagger API documentation

✓ API integration documentation
```

---

# 77. Final Architecture Summary

The project combines:

```text
Machine Learning
        +
MLOps
        +
FastAPI
        +
Pydantic
        +
Structured Logging
        +
Error Handling
        +
Podman
        +
Ollama
        +
Llama 3
        +
LangChain
        +
React
        =
Customer Churn AI
```

The system demonstrates an end-to-end AI application rather than only an ML model.

It covers:

```text
Training
   ↓
Saved Model
   ↓
Model Serving
   ↓
REST API
   ↓
Validation
   ↓
Logging
   ↓
Containerization
   ↓
Local LLM
   ↓
Chatbot
   ↓
React UI
```