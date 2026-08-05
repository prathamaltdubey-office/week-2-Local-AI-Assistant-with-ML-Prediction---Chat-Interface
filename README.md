# 🚀 Production-Ready Customer Churn Prediction Pipeline

> A complete Machine Learning Engineering project demonstrating data preprocessing, model training, hyperparameter tuning, experiment tracking, testing, and deployment using Streamlit.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses such as telecom companies.

This project builds a **production-ready Machine Learning pipeline** that predicts whether a customer is likely to churn based on demographic information, subscribed services, billing information, and customer tenure.

The project follows **ML Engineering** and **MLOps best practices**, including:

- Modular project structure
- Scikit-learn Pipelines
- Hyperparameter tuning
- MLflow experiment tracking
- Unit testing with Pytest
- Code formatting using Black, Ruff and isort
- Streamlit deployment
- Docker support
- Reproducible workflows

---

# 📷 Application Preview

## Streamlit Home

> *(Insert Screenshot Here)*

![Streamlit Home](docs/images/streamlit-home.png)

---

## Prediction Result

> *(Insert Screenshot Here)*

![Prediction](docs/images/prediction-result.png)

---

## Model Insights

> *(Insert Screenshot Here)*

![Insights](docs/images/model-insights.png)

---

# 📁 Project Structure

```
customer-churn-prediction-main
│
├── app
│   └── app.py
│
├── data
│   └── Telco_Customer_churn.csv
│
├── models
│   ├── logistic_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── best_logistic_model.pkl
│   ├── best_rf_model.pkl
│   └── best_xgb_model.pkl
│
├── reports
│   └── model_metrics.csv
│
├── src
│   ├── train.py
│   ├── tuning.py
│   └── evaluate.py
│
├── tests
│   ├── conftest.py
│   └── test_pipeline.py
│
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📊 Dataset

Dataset Used:

**IBM Telco Customer Churn Dataset**

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

Target Variable

```
Churn
```

- Yes
- No

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-learn | ML Pipeline |
| XGBoost | Gradient Boosting |
| MLflow | Experiment Tracking |
| Joblib | Model Serialization |
| Streamlit | Web Application |
| Matplotlib | Visualization |
| Pytest | Unit Testing |
| Ruff | Linting |
| Black | Formatting |
| isort | Import Sorting |
| Docker | Containerization |

---

# ⚙ Machine Learning Workflow

```
Dataset

        │

        ▼

Data Cleaning

        │

        ▼

Feature Engineering

        │

        ▼

Train Test Split

        │

        ▼

Preprocessing Pipeline

        │

        ▼

Train 3 ML Models

        │

        ▼

Hyperparameter Tuning

        │

        ▼

Model Evaluation

        │

        ▼

MLflow Logging

        │

        ▼

Save Best Models

        │

        ▼

Streamlit Deployment
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline performs:

- Removal of customerID
- Conversion of TotalCharges to numeric
- Missing value handling
- Label Encoding of target
- One Hot Encoding for categorical variables
- Standard Scaling for numerical variables

Implemented using

```
ColumnTransformer
```

and

```
Pipeline
```

from Scikit-learn.

---

# 🤖 Models Trained

Three supervised learning models were trained.

## Logistic Regression

Advantages

- Simple
- Fast
- Interpretable

---

## Random Forest

Advantages

- Ensemble learning
- Handles nonlinear relationships
- Robust against overfitting

---

## XGBoost

Advantages

- Gradient Boosting
- High predictive performance
- Efficient training

---

# 🔍 Hyperparameter Tuning

RandomizedSearchCV was used for tuning.

Optimized Parameters included:

### Logistic Regression

- C
- Solver
- Penalty

### Random Forest

- Number of Trees
- Maximum Depth
- Minimum Samples Split
- Minimum Samples Leaf

### XGBoost

- Learning Rate
- Number of Trees
- Maximum Depth
- Subsample

---

# 📈 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Example Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|-----------|--------|-----|---------|
| Logistic Regression | *(Your Result)* | | | | |
| Random Forest | *(Your Result)* | | | | |
| XGBoost | *(Your Result)* | | | | |

---

# 📊 MLflow Experiment Tracking

Each experiment logs:

- Parameters
- Metrics
- Best Model
- Artifacts

### MLflow Dashboard

> *(Insert Screenshot Here)*

![MLflow](docs/images/mlflow-dashboard.png)

---

### MLflow Run Details

> *(Insert Screenshot Here)*

![MLflow Run](docs/images/mlflow-run.png)

---

# 💻 Streamlit Application

The application allows users to:

- Choose a model
- Enter customer information
- Predict customer churn
- View churn probability
- View risk level
- Compare model performance
- Explore feature importance

---

# 📈 Feature Importance

Random Forest feature importance is visualized inside Streamlit.

Top influential features include:

- Tenure
- Contract Type
- Monthly Charges
- Total Charges
- Internet Service

> *(Insert Screenshot Here)*

![Feature Importance](docs/images/feature-importance.png)

---

# 🧪 Testing

Unit tests were written using **Pytest**.

Covered:

- Data Loading
- Data Cleaning
- Data Splitting
- Preprocessing
- Training
- Evaluation
- Hyperparameter Tuning
- Model Saving

Current Results

```
25 Passed
```

Coverage

```
89%
```

---

### Test Coverage

> *(Insert Screenshot Here)*

![Coverage](docs/images/test-coverage.png)

---

# ✅ Code Quality

The project follows Python best practices.

Formatting

```
black .
```

Import Sorting

```
isort .
```

Linting

```
ruff check .
```

Testing

```
pytest --cov=src --cov-report=term-missing
```

---

# 🐳 Docker

A Dockerfile has been included for reproducible model training.

Build

```bash
docker build -t customer-churn .
```

Run

```bash
docker run customer-churn
```

> **Note:** Docker image execution could not be verified locally due to corporate security restrictions on the development machine. The Dockerfile has been prepared according to standard Python project practices.

---

# ▶ Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Models

```bash
python src/train.py
```

---

## Hyperparameter Tuning

```bash
python src/tuning.py
```

---

## Evaluate Models

```bash
python src/evaluate.py
```

---

## Launch Streamlit

```bash
streamlit run app/app.py
```

---

# 📦 Saved Models

The following trained models are saved using Joblib.

```
best_logistic_model.pkl

best_rf_model.pkl

best_xgb_model.pkl
```

---

# 🎯 Business Insights

The trained models indicate that churn is strongly associated with:

- Short customer tenure
- Month-to-month contracts
- High monthly charges
- Fiber optic internet users
- Lack of value-added services

Recommendations:

- Offer long-term contract discounts.
- Improve onboarding for new customers.
- Bundle security and support services.
- Launch targeted retention campaigns.

---

# 🚀 Future Improvements

Possible future enhancements include:

- SHAP Explainability
- LightGBM Model
- FastAPI Deployment
- CI/CD using GitHub Actions
- Model Monitoring
- Automated Retraining
- Kubernetes Deployment
- Cloud Deployment (AWS/Azure/GCP)

---

# 📚 References

- IBM Telco Customer Churn Dataset
- Scikit-learn Documentation
- XGBoost Documentation
- MLflow Documentation
- Streamlit Documentation

---

# 👨‍💻 Author

**Pratham Dubey**

Production-Ready Customer Churn Prediction Pipeline

Machine Learning Engineering & MLOps Project

---

# ⭐ Acknowledgements

This project was developed as part of an **ML Engineering & MLOps learning program**, focusing on building reproducible, modular, and production-ready machine learning workflows using modern Python tools and best practices.