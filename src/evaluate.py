"""
Evaluate trained machine learning models for customer churn prediction.

This script loads the trained models, evaluates their
performance on the test dataset, and saves the evaluation
metrics to a CSV report.
"""

import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv("./data/Telco_Customer_churn.csv")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.

    Args:
        df: Raw customer churn dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    df = df.drop(
        "customerID",
        axis=1,
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = df["Churn"].map(
        {
            "Yes": 1,
            "No": 0,
        }
    )

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split dataset into training and testing sets.

    Args:
        df: Cleaned dataset.

    Returns:
        Tuple containing training and testing data.
    """

    X = df.drop(
        "Churn",
        axis=1,
    )

    y = df["Churn"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def load_models() -> dict[str, object]:
    """
    Load all trained machine learning models.

    Returns:
        Dictionary containing trained models.
    """

    return {
        "Tuned Logistic Regression": load("models/best_logistic_model.pkl"),
        "Tuned Random Forest": load("models/best_rf_model.pkl"),
        "Tuned XGBoost": load("models/best_xgb_model.pkl"),
    }


def evaluate_models(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate trained models.

    Args:
        models: Dictionary of trained models.
        X_test: Testing features.
        y_test: Testing labels.

    Returns:
        pd.DataFrame: Evaluation metrics.
    """

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "Precision": precision_score(
                    y_test,
                    predictions,
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                ),
                "F1 Score": f1_score(
                    y_test,
                    predictions,
                ),
                "ROC-AUC": roc_auc_score(
                    y_test,
                    probabilities,
                ),
            }
        )

    return pd.DataFrame(results)


def save_metrics(
    metrics: pd.DataFrame,
) -> None:
    """
    Save evaluation metrics.

    Args:
        metrics: DataFrame containing evaluation metrics.
    """

    metrics.to_csv(
        "reports/model_metrics.csv",
        index=False,
    )

    print("\nEvaluation metrics saved successfully.")


def main() -> None:
    """
    Execute the complete model evaluation workflow.
    """

    df = load_data()

    df = clean_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(df)

    models = load_models()

    metrics = evaluate_models(
        models,
        X_test,
        y_test,
    )

    print("\nModel Performance\n")
    print(metrics)

    save_metrics(metrics)


if __name__ == "__main__":
    main()
