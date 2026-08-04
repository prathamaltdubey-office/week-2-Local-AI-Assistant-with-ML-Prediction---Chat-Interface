import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.sklearn import SERIALIZATION_FORMAT_CLOUDPICKLE

from joblib import dump

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

mlflow.set_experiment("Customer Churn Prediction")

df = pd.read_csv(
    "./data/Telco_Customer_churn.csv"
)


df = df.drop(
    "customerID",
    axis=1
)


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


df["TotalCharges"] = df["TotalCharges"].fillna(0)


df["Churn"] = df["Churn"].map(
    {
        "Yes":1,
        "No":0
    }
)



X = df.drop(
    "Churn",
    axis=1
)


y = df["Churn"]



X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)



categorical_columns = X.select_dtypes(
    include="object"
).columns


numerical_columns = X.select_dtypes(
    exclude="object"
).columns


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

logistic_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )

    ]

)

logistic_params = {

    "model__C":

    [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ],


    "model__solver":

    [
        "liblinear",
        "lbfgs"
    ],


    "model__penalty":

    [
        "l2"
    ]

}

logistic_search = RandomizedSearchCV(

    estimator=logistic_pipeline,

    param_distributions=logistic_params,

    n_iter=10,

    cv=5,

    scoring="roc_auc",

    random_state=42,

    n_jobs=-1

)


with mlflow.start_run(run_name="Logistic Regression"):

    logistic_search.fit(
        X_train,
        y_train
    )

    print("Best Logistic Regression Parameters:")
    print(logistic_search.best_params_)

    mlflow.log_params(logistic_search.best_params_)

    mlflow.log_metric(
        "best_cv_score",
        logistic_search.best_score_
    )

    mlflow.sklearn.log_model(
        sk_model=logistic_search.best_estimator_,
        name="model",
        serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE
    )

    dump(
        logistic_search.best_estimator_,
        "models/best_logistic_model.pkl"
    )


rf_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),


        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        )

    ]

)





rf_params = {


    "model__n_estimators":

    [100,200,300],



    "model__max_depth":

    [5,10,20,None],



    "model__min_samples_split":

    [2,5,10],



    "model__min_samples_leaf":

    [1,2,4]


}



rf_search = RandomizedSearchCV(

    estimator=rf_pipeline,

    param_distributions=rf_params,

    n_iter=10,

    cv=5,

    scoring="roc_auc",

    random_state=42,

    n_jobs=-1

)



with mlflow.start_run(run_name="Random Forest"):

    rf_search.fit(
        X_train,
        y_train
    )

    print("Best Random Forest Parameters:")
    print(rf_search.best_params_)

    mlflow.log_params(
        rf_search.best_params_
    )

    mlflow.log_metric(
        "best_cv_score",
        rf_search.best_score_
    )

    mlflow.sklearn.log_model(
        rf_search.best_estimator_,
        name="model",
        serialization_format="cloudpickle"
    )

    dump(
        rf_search.best_estimator_,
        "models/best_rf_model.pkl"
    )



xgb_pipeline = Pipeline(

steps=[

(
"preprocessor",
preprocessor
),


(
"model",
XGBClassifier(
random_state=42
)
)

]

)


xgb_params = {


"model__n_estimators":

[100,200,300],



"model__learning_rate":

[0.01,0.05,0.1],



"model__max_depth":

[3,5,7],



"model__subsample":

[0.7,0.8,1.0]


}



xgb_search = RandomizedSearchCV(

    estimator=xgb_pipeline,

    param_distributions=xgb_params,

    n_iter=10,

    cv=5,

    scoring="roc_auc",

    random_state=42,

    n_jobs=-1

)



with mlflow.start_run(run_name="XGBoost"):

    xgb_search.fit(
        X_train,
        y_train
    )

    print("Best XGBoost Parameters:")
    print(xgb_search.best_params_)

    mlflow.log_params(
        xgb_search.best_params_
    )

    mlflow.log_metric(
        "best_cv_score",
        xgb_search.best_score_
    )

    mlflow.sklearn.log_model(
        xgb_search.best_estimator_,
        name="model",
        serialization_format="cloudpickle"
    )

    dump(
        xgb_search.best_estimator_,
        "models/best_xgb_model.pkl"
    )
