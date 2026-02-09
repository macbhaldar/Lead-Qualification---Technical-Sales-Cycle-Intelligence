# SHAP-based ROI explanation

!pip install shap

import shap
import numpy as np

# Extract trained components
xgb_model = xgb_pipeline.named_steps["model"]
preprocessor = xgb_pipeline.named_steps["preprocess"]

X_transformed = preprocessor.transform(X)

explainer = shap.LinearExplainer(xgb_model, X_transformed)
shap_values = explainer.shap_values(X_transformed)

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=preprocessor.get_feature_names_out()
)

## ROI-Aware SHAP

df["shap_win_lift"] = shap_values.sum(axis=1)
df["shap_expected_revenue"] = (
    df["shap_win_lift"] * df["estimated_deal_value_usd"]
)

df["shap_expected_profit"] = (
    df["shap_expected_revenue"] - df["total_rfp_cost"]
)

deal_id = 42
idx = df.index[df["lead_id"] == deal_id][0]

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_transformed[idx],
        feature_names=preprocessor.get_feature_names_out()
    )
)

## Counterfactual Insight

df.loc[idx, [
    "solution_fit_score",
    "response_time_hours",
    "technical_complexity",
    "roi"
]]

## SHAP × ROI Segmentation

df["roi_bucket"] = pd.cut(
    df["roi"],
    bins=[-10,0,1,10],
    labels=["Value Destroying","Marginal","High ROI"]
)

df.groupby("roi_bucket")[
    ["solution_fit_score","response_time_hours","technical_complexity"]
].mean()
