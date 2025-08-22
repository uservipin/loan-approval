"""
loan_approval_deploy_pipeline.py
--------------------------------

This script implements a simplified, deployment‑oriented pipeline for a
loan‑approval prediction system.  It handles class imbalance via
class weights, uses stratified k‑fold cross‑validation with grid
search to tune a logistic regression model, excludes sensitive
fairness attributes from the predictors, and provides basic
transparency by reporting feature coefficients.  It also contains
placeholders and guidelines for integrating human‑in‑the‑loop review
and drift monitoring.

Key features implemented:

* **Class imbalance handling** – Logistic regression is fit with
  `class_weight='balanced'` so the minority class (approved loans)
  receives higher importance during training.
* **Cross‑validation & hyper‑parameter tuning** – Uses
  `StratifiedKFold` within `GridSearchCV` to select the best
  regularisation strength (`C` value).  Cross‑validation reduces
  over‑fitting by evaluating on multiple train/test splits【712762840479629†L127-L135】.
* **Excluding fairness features** – Protected attributes (e.g.
  gender, race) are removed from the predictor set to ensure the
  model does not directly learn from sensitive information.
* **Transparency via coefficients** – After fitting, the script
  extracts the logistic regression coefficients and corresponding
  feature names from the one‑hot encoder, allowing users to see
  which features have the strongest impact on loan decisions.
* **Human‑in‑the‑loop & drift monitoring (guidelines)** – Comments
  describe how one might route borderline predictions to a human
  reviewer and monitor data drift post‑deployment.  Implementing
  these systems typically requires integration with production
  services and is beyond the scope of this standalone script.

Running this script requires the synthetic dataset and metadata CSV
files to be available in the same directory.  Sampling the dataset
to 200 000 rows (via the `sample_size` argument) keeps the
computation time reasonable.  For production use, remove sampling.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


def load_data_and_metadata(
    dataset_path: str,
    metadata_path: str,
    sample_size: int | None = 200_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the synthetic dataset and feature metadata.

    Optionally sample rows to keep training manageable.

    Returns a tuple of (dataframe, metadata).
    """
    metadata = pd.read_csv(metadata_path)
    df = pd.read_csv(dataset_path, parse_dates=["application_date"], dayfirst=True)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df, metadata


def derive_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional risk‑related features from existing columns.

    This mirrors the derived features described in the metadata,
    including debt‑to‑income ratio, loan‑to‑income ratio, loan‑to‑value
    ratio, credit score buckets, age categories, employment/credit
    interaction and a rudimentary payment behaviour score.  Date
    columns are decomposed into year, month and weekday and then
    dropped.
    """
    df = df.copy()
    # Debt‑to‑income ratio: (existing debt + new loan payment) / income
    df["debt_to_income_ratio"] = (
        df["existing_debt_obligations"] + df["loan_amnt"] * (df["loan_int_rate"] / 100)
    ) / df["person_income"].replace(0, np.nan)
    # Loan‑to‑income ratio
    df["loan_to_income_ratio"] = df["loan_amnt"] / df["person_income"].replace(0, np.nan)
    # Loan‑to‑value ratio (approximate, using credit limits as proxy for collateral value)
    df["loan_to_value_ratio"] = df["loan_amnt"] / (
        df["loan_amnt"] + df["total_credit_limits"].replace(0, np.nan)
    )
    # Credit score bins
    def bin_score(s):
        if pd.isna(s):
            return "Unknown"
        return (
            "Low" if s < 600 else
            "Medium" if s < 650 else
            "Good" if s < 700 else
            "Excellent"
        )
    df["credit_score_bin"] = df["credit_score"].apply(bin_score)
    # Age categories
    def bucket_age(a):
        if pd.isna(a):
            return "Unknown"
        return (
            "Under25" if a < 25 else
            "25_34" if a < 35 else
            "35_44" if a < 45 else
            "45_54" if a < 55 else
            "55plus"
        )
    df["age_category"] = df["person_age"].apply(bucket_age)
    # Interaction between employment tenure and credit history length
    df["employment_credit_interaction"] = df["person_emp_exp"] * df["cb_person_cred_hist_length"]
    # Payment behaviour score: credit utilisation and hard inquiries
    df["payment_behavior_score"] = (
        0.5 * df["credit_utilisation_ratio"].fillna(0) +
        0.3 * df["number_of_hard_inquiries"].fillna(0) -
        0.1 * df["months_since_last_delinquency"].replace(999, np.nan).fillna(0) / 12
    )
    # Extract temporal features from application_date
    if "application_date" in df.columns and np.issubdtype(df["application_date"].dtype, np.datetime64):
        df["application_year"] = df["application_date"].dt.year
        df["application_month"] = df["application_date"].dt.month
        df["application_dayofweek"] = df["application_date"].dt.dayofweek
        df = df.drop(columns=["application_date"])
    return df


def build_and_train_model(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    categories: List[str],
    target_col: str = "loan_status",
) -> Tuple[Pipeline, Dict[str, float]]:
    """Prepare the data, build the pipeline, run grid search and return the model and metrics.

    - Selects predictor columns based on metadata categories, excluding fairness features.
    - Encodes categorical variables and scales numeric ones.
    - Performs stratified train/test split.
    - Executes grid search over C values using F1 score for evaluation.
    - Computes and returns basic performance metrics.
    """
    # Derive risk features
    df = derive_risk_features(df)
    # Identify columns to use as predictors
    fairness_cols = metadata.loc[metadata["Category"] == "FairnessGovernance", "FeatureName"]
    predictor_cols = metadata.loc[metadata["Category"].isin(categories), "FeatureName"].tolist()
    # Include all derived features explicitly listed in metadata
    derived_cols = metadata.loc[metadata["Category"] == "DerivedFeatures", "FeatureName"].tolist()
    predictor_cols = [c for c in predictor_cols + derived_cols if c in df.columns and c not in fairness_cols.values]
    # Split features and target
    X = df[predictor_cols].copy()
    y = df[target_col].copy()
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    # Identify categorical/numeric columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
    # Build preprocessing and model pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", clf)
    ])
    # Grid search over regularisation strengths
    param_grid = {"model__C": [0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "F1": f1_score(y_test, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "PR_AUC": average_precision_score(y_test, y_prob),
        "BrierScore": brier_score_loss(y_test, y_prob),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred),
    }
    return best_model, metrics


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Extract the logistic regression coefficients and corresponding feature names.

    This function maps coefficients back to the preprocessed feature names
    (after one‑hot encoding) so that domain experts can see which
    variables have the greatest positive or negative influence on the
    prediction.  Only applicable to linear models such as logistic
    regression.
    """
    # Retrieve names from the ColumnTransformer
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    # The order of transformed columns is numeric + one‑hot encoded categories
    num_features = preprocessor.transformers_[0][2]
    cat_encoder: OneHotEncoder = preprocessor.transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
    feature_names = list(num_features) + list(cat_features)
    # Coefficients from logistic regression (flat array)
    coef = model.named_steps["model"].coef_.ravel()
    importance = pd.DataFrame({"feature": feature_names, "coefficient": coef})
    importance = importance.sort_values(by="coefficient", key=lambda x: x.abs(), ascending=False)
    return importance


def main(
    dataset_path: str = "/home/oai/share/loan_approval_synthetic_dataset.csv",
    metadata_path: str = "/home/oai/share/feature_metadata.csv",
    sample_size: int | None = 200_000,
):
    """Execute training and evaluation on the synthetic dataset.

    After fitting the model, report predictive metrics and the
    top‑weighted features for transparency.  Guidelines for human
    oversight and drift monitoring are printed at the end.
    """
    df, metadata = load_data_and_metadata(dataset_path, metadata_path, sample_size=sample_size)
    categories = ["PrimaryCSV", "PrimaryCreditBureau"]
    model, metrics = build_and_train_model(df, metadata, categories)
    print("\nPredictive metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    # Transparency: show top 10 coefficients
    importance = extract_feature_importance(model)
    print("\nTop 10 features influencing approval decisions:")
    print(importance.head(10))
    # Human‑in‑the‑loop guidelines (conceptual)
    print("\n--- Human‑in‑the‑loop and drift monitoring guidelines ---")
    print(
        "1. Define probability thresholds that flag borderline cases (e.g. 0.45–0.55). "
        "Send those applications to a human underwriter for review and record whether "
        "they override the model.\n"
        "2. Track override rates and reasons; incorporate this feedback when retraining "
        "to refine thresholds.\n"
        "3. Monitor covariate drift by comparing incoming feature distributions to the "
        "training data (e.g. using statistical tests). Significant shifts should trigger "
        "model retraining or recalibration.\n"
        "4. Monitor concept drift by periodically evaluating model performance (F1, ROC‑AUC) "
        "on recent data. Declining performance signals a need for updates.\n"
        "5. Maintain a model card documenting data sources, preprocessing steps, "
        "hyper‑parameters and known limitations."
    )


if __name__ == "__main__":
    main()