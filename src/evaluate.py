import pandas as pd

from joblib import load

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)



# Load data

df = pd.read_csv(
   "./data/Telco_Customer_churn.csv"
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


df = df.drop(
    "customerID",
    axis=1
)


X = df.drop(
    "Churn",
    axis=1
)


y = df["Churn"]



# Same split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)



# Load models


models = {

    "Tuned Logistic Regression":
    load(
        "models/best_logistic_model.pkl"
    ),


    "Tuned Random Forest":
    load(
        "models/best_rf_model.pkl"
    ),


    "Tuned XGBoost":
    load(
        "models/best_xgb_model.pkl"
    )

}

results = []



for name, model in models.items():

    prediction = model.predict(
        X_test
    )


    probability = model.predict_proba(
        X_test
    )[:,1]


    results.append({

        "Model": name,

        "Accuracy":
        accuracy_score(
            y_test,
            prediction
        ),


        "Precision":
        precision_score(
            y_test,
            prediction
        ),


        "Recall":
        recall_score(
            y_test,
            prediction
        ),


        "F1 Score":
        f1_score(
            y_test,
            prediction
        ),


        "ROC-AUC":
        roc_auc_score(
            y_test,
            probability
        )

    })



metrics = pd.DataFrame(results)


print(metrics)



metrics.to_csv(
    "reports/model_metrics.csv",
    index=False
)


print(
    "Metrics saved"
)