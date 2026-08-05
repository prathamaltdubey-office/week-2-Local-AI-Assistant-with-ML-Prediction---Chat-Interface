"""
Train machine learning models for customer churn prediction.

This script loads the dataset, preprocesses the data,
trains multiple machine learning models using a
Scikit-learn pipeline, and saves the trained models.
"""

from typing import Dict

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


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
        df: Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = df.drop("customerID", axis=1)

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
    Split the dataset into train and test sets.

    Args:
        df: Cleaned dataset.

    Returns:
        Tuple containing training and testing datasets.
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def create_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create the preprocessing pipeline.

    Args:
        X_train: Training features.

    Returns:
        ColumnTransformer: Configured preprocessing pipeline.
    """
    categorical_columns = X_train.select_dtypes(include="object").columns

    numerical_columns = X_train.select_dtypes(exclude="object").columns

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_columns,
            ),
            (
                "num",
                StandardScaler(),
                numerical_columns,
            ),
        ]
    )


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> None:
    """
    Train machine learning models and save them.

    Args:
        X_train: Training features.
        y_train: Training labels.
        preprocessor: Data preprocessing pipeline.
    """

    models: Dict[str, object] = {
        "logistic": LogisticRegression(
            random_state=42,
            max_iter=1000,
        ),
        "rf": RandomForestClassifier(
            random_state=42,
        ),
        "xgb": XGBClassifier(
            random_state=42,
        ),
    }

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        dump(
            pipeline,
            f"models/{name}_model.pkl",
        )

        print(f"{name} model saved successfully.")


def main() -> None:
    """
    Execute the complete training workflow.
    """

    df = load_data()

    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = create_preprocessor(X_train)

    train_models(
        X_train,
        y_train,
        preprocessor,
    )

    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
