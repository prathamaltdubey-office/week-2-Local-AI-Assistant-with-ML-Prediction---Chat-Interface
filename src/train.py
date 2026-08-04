import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from joblib import dump


# Load dataset
df = pd.read_csv(
    "./data/Telco_Customer_churn.csv"
)

df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# Convert target column

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# Split features and target

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]


# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Identify columns

categorical_columns = X.select_dtypes(
    include="object"
).columns


numerical_columns = X.select_dtypes(
    exclude="object"
).columns



# Preprocessing

preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        ),

        (
            "num",
            StandardScaler(),
            numerical_columns
        )

    ]

)



# Models

models = {

    "logistic":
    LogisticRegression(),

    "rf":
    RandomForestClassifier(
        random_state=42
    ),

    "xgb":
    XGBClassifier(
        random_state=42
    )

}



# Train and save models

for name, model in models.items():

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )


    pipeline.fit(
        X_train,
        y_train
    )


    dump(
        pipeline,
        f"models/{name}_model.pkl"
    )


print("Models trained successfully")