# Random Forest and XGBoost Model

## Random Forest Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/sales_lead_rfp_dataset.csv")

target = "rfp_win"

categorical = ["industry", "company_size", "region", "lead_source"]
numerical = [
    "technical_complexity",
    "response_time_hours",
    "solution_fit_score",
    "estimated_deal_value_usd",
    "past_vendor_experience"
]

X = df[categorical + numerical]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ]
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    class_weight="balanced"
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", rf)
    ]
)

rf_pipeline.fit(X_train, y_train)

rf_probs = rf_pipeline.predict_proba(X_test)[:,1]

print("ROC-AUC:", roc_auc_score(y_test, rf_probs))
print(classification_report(y_test, rf_pipeline.predict(X_test)))

feature_names = (
    rf_pipeline.named_steps["preprocess"]
    .get_feature_names_out()
)

importances = rf_pipeline.named_steps["model"].feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

fi.head(10)

## XGBOOST Model

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42
)

xgb_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", xgb)
    ]
)

xgb_pipeline.fit(X_train, y_train)

xgb_probs = xgb_pipeline.predict_proba(X_test)[:,1]

print("XGBoost ROC-AUC:", roc_auc_score(y_test, xgb_probs))

df["win_probability"] = xgb_pipeline.predict_proba(X)[:,1]
df["expected_revenue"] = (
    df["win_probability"] * df["estimated_deal_value_usd"]
)

df.sort_values("expected_revenue", ascending=False).head(10)
