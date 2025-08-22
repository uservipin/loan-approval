"""
loan_approval_full_pipeline.py
================================

This module implements a full end‑to‑end workflow for training and
evaluating a loan‑approval prediction model.  It relies on the
`feature_metadata.csv` file to identify which columns in the
synthetic dataset are primary features, credit‑bureau features and
fairness/governance attributes.  It then derives additional risk
features, performs preprocessing (standardisation and one‑hot
encoding), tunes a logistic regression model using cross‑validation,
evaluates predictive performance, computes fairness metrics and
outlines steps for governance and drift monitoring.

The code is organised into discrete functions so that each step can
easily be tested or modified independently.  The `main` function
orchestrates the workflow.  For demonstration purposes, the
training data are subsampled to 200 000 rows to keep computation
manageable; in a production setting you can remove this sampling
step to use the full dataset.

References:
  * Statistical parity difference and impact ratio definitions
    illustrate how to quantify fairness【699740487529563†L106-L112】【699740487529563†L119-L124】.
  * Cross‑validation helps avoid over‑fitting by testing on
    independent folds【712762840479629†L127-L135】.
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


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load the feature metadata file.

    Args:
        metadata_path: Path to the metadata CSV file.

    Returns:
        DataFrame with columns [FeatureName, Category, Description, Notes].
    """
    metadata = pd.read_csv(metadata_path)
    required_cols = {"FeatureName", "Category"}
    if not required_cols.issubset(metadata.columns):
        raise ValueError(
            f"Metadata file must contain columns {required_cols}, but has {metadata.columns.tolist()}"
        )
    return metadata


def load_dataset(dataset_path: str, sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Load the synthetic loan dataset.  Optionally sample a subset of rows.

    Sampling is useful to keep computation tractable during
    experimentation.  When `sample_size` is None, the full dataset is
    returned.

    Args:
        dataset_path: Path to the dataset CSV file.
        sample_size: Optional number of rows to sample from the
            dataset.  If None, all rows are loaded.
        random_state: Random seed used for reproducible sampling.

    Returns:
        DataFrame containing the dataset (or sampled subset).
    """
    df = pd.read_csv(dataset_path, parse_dates=["application_date"], dayfirst=True, infer_datetime_format=True)
    # If sampling is requested, sample without replacement
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df


def select_features(metadata: pd.DataFrame, categories: List[str]) -> List[str]:
    """Select feature names belonging to specified categories.

    Args:
        metadata: DataFrame loaded from feature_metadata.csv.
        categories: List of category names to include (e.g.
            ['PrimaryCSV', 'PrimaryCreditBureau']).

    Returns:
        List of feature names that belong to the given categories.
    """
    mask = metadata["Category"].isin(categories)
    return metadata.loc[mask, "FeatureName"].tolist()


def select_fairness_features(metadata: pd.DataFrame) -> List[str]:
    """Return the list of fairness/governance attributes for fairness testing.

    These features should not be used as predictors, but they are
    required to compute group‑level fairness metrics.
    """
    mask = metadata["Category"] == "FairnessGovernance"
    return metadata.loc[mask, "FeatureName"].tolist()


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute additional derived features on the dataframe in‑place.

    Derived features help capture non‑linear relationships and
    interactions described in the metadata.  Some derived features
    require creative approximations because the synthetic dataset may
    not contain all underlying data (e.g. collateral values).

    Args:
        df: DataFrame on which to compute derived features.  This
            DataFrame is modified in place and also returned.

    Returns:
        The DataFrame with new derived feature columns added.
    """
    # Debt‑to‑income ratio: (existing debt + payment for new loan) / income
    df["debt_to_income_ratio"] = (
        df["existing_debt_obligations"] + df["loan_amnt"] * (df["loan_int_rate"] / 100.0)
    ) / df["person_income"].replace(0, np.nan)
    # Loan‑to‑income ratio: loan amount divided by income
    df["loan_to_income_ratio"] = df["loan_amnt"] / df["person_income"].replace(0, np.nan)
    # Loan‑to‑value ratio: approximated as loan / (loan + credit limits) if
    # collateral values are unavailable.  This ensures the ratio is
    # bounded by [0, 1).  For applicants with no credit limits, set to 1.
    df["loan_to_value_ratio"] = df["loan_amnt"] / (
        df["loan_amnt"] + df["total_credit_limits"].replace(0, np.nan)
    )
    # Credit score bins: group numeric credit score into categories
    def credit_bin(score: float) -> str:
        if pd.isna(score):
            return "Unknown"
        if score < 600:
            return "Low"
        elif score < 650:
            return "Medium"
        elif score < 700:
            return "Good"
        else:
            return "Excellent"
    df["credit_score_bin"] = df["credit_score"].apply(credit_bin)
    # Age categories: bucket age into ranges
    def age_bucket(age: float) -> str:
        if pd.isna(age):
            return "Unknown"
        if age < 25:
            return "Under25"
        elif age < 35:
            return "25_34"
        elif age < 45:
            return "35_44"
        elif age < 55:
            return "45_54"
        else:
            return "55plus"
    df["age_category"] = df["person_age"].apply(age_bucket)
    # Employment‑credit interaction: product of employment exp and credit history length
    df["employment_credit_interaction"] = df["person_emp_exp"] * df["cb_person_cred_hist_length"]
    # Payment behaviour score: simple heuristic combining credit utilisation, hard inquiries and recent delinquency
    # Higher utilisation and more inquiries increase risk; longer time since delinquency reduces risk.
    df["payment_behavior_score"] = (
        0.5 * df["credit_utilisation_ratio"].fillna(0)
        + 0.3 * df["number_of_hard_inquiries"].fillna(0)
        - 0.1 * df["months_since_last_delinquency"].replace(999, np.nan).fillna(0) / 12.0
    )
    # Extract useful temporal features from application_date if present
    if "application_date" in df.columns and np.issubdtype(df["application_date"].dtype, np.datetime64):
        df["application_year"] = df["application_date"].dt.year
        df["application_month"] = df["application_date"].dt.month
        df["application_dayofweek"] = df["application_date"].dt.dayofweek
        # Drop original datetime column to avoid type errors in preprocessing
        df.drop(columns=["application_date"], inplace=True)
    return df


def prepare_data(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    categories: List[str],
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], List[str], List[str]]:
    """Prepare the dataset for modelling.

    This function derives features, selects predictor columns and
    separates the data into train and test splits.  It returns the
    training/testing features and labels as well as lists of
    categorical, numerical and fairness columns for later processing.

    Args:
        df: Raw DataFrame loaded from the CSV (sampled).  Must
            include a `loan_status` column as target.
        metadata: Feature metadata with categories and notes.
        categories: Which categories of features to use as predictors
            (e.g. ['PrimaryCSV', 'PrimaryCreditBureau']).  Derived
            features are added automatically.
        test_size: Fraction of data to use for testing.
        random_state: Seed for reproducible splits.

    Returns:
        Tuple containing X_train, y_train, X_test, y_test, list of
        categorical column names, numeric column names, and fairness
        attribute names.
    """
    # Derive additional features
    df = derive_features(df)
    # Select predictor features based on metadata categories
    predictor_cols = select_features(metadata, categories) + list(
        metadata[metadata["Category"] == "DerivedFeatures"]["FeatureName"]
    )
    # Ensure predictor_cols exist in the dataframe
    predictor_cols = [col for col in predictor_cols if col in df.columns]
    # Identify fairness attributes (for later fairness tests)
    fairness_cols = select_fairness_features(metadata)
    fairness_cols = [col for col in fairness_cols if col in df.columns]
    # Exclude fairness columns from predictors to prevent leakage of protected information
    predictor_cols = [col for col in predictor_cols if col not in fairness_cols]
    # Subset predictors and target
    X = df[predictor_cols].copy()
    y = df["loan_status"].copy()
    # Determine categorical and numeric feature names for preprocessing
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    # Split into train and test sets (stratified to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, y_train, X_test, y_test, categorical_cols, numeric_cols, fairness_cols


def build_pipeline(categorical_cols: List[str], numeric_cols: List[str]) -> Pipeline:
    """Construct the preprocessing and model pipeline.

    The pipeline standardises numerical features and one‑hot encodes
    categorical features, then fits a logistic regression classifier
    with class weights to handle imbalance.

    Args:
        categorical_cols: List of names of categorical predictor features.
        numeric_cols: List of names of numeric predictor features.

    Returns:
        A scikit‑learn Pipeline object.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    logit = LogisticRegression(max_iter=200, class_weight="balanced")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", logit)])
    return pipeline


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List],
    cv_splits: int = 3,
    scoring: str = "f1",
) -> GridSearchCV:
    """Tune hyper‑parameters and fit the model using cross‑validation.

    Args:
        pipeline: A scikit‑learn Pipeline with preprocessing and
            classifier steps defined.
        X_train: Training features.
        y_train: Training labels.
        param_grid: Dictionary mapping parameter names to lists of
            values for grid search (e.g., {'model__C': [0.1, 1, 10]}).
        cv_splits: Number of stratified k‑folds for cross‑validation.
        scoring: Metric to optimise during grid search.

    Returns:
        Fitted GridSearchCV object containing the best estimator.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_performance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    print_results: bool = True,
) -> Dict[str, float | np.ndarray]:
    """Compute predictive performance metrics on the test set.

    Args:
        model: Fitted pipeline or estimator with `predict` and
            `predict_proba` methods.
        X_test: Test features.
        y_test: True labels for the test set.
        print_results: If True, print the metrics to stdout.

    Returns:
        Dictionary containing metrics (F1, balanced accuracy, precision,
        recall, ROC‑AUC, PR‑AUC, Brier score and confusion matrix).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics: Dict[str, float | np.ndarray] = {
        "F1 Score": f1_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC‑AUC": roc_auc_score(y_test, y_prob),
        "PR‑AUC": average_precision_score(y_test, y_prob),
        "Brier Score": brier_score_loss(y_test, y_prob),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
    }
    if print_results:
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    return metrics


def compute_fairness_metrics(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    fairness_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    """Calculate fairness metrics across protected groups.

    Args:
        model: Fitted pipeline with predict and predict_proba.
        X_test: Processed features for test set (not used here but
            retained for interface consistency).
        y_test: True labels for test set.
        df_test: Original test DataFrame (including fairness columns).
        fairness_cols: List of protected attribute names (e.g.,
            ['person_gender', 'race']).

    Returns:
        A dictionary mapping each protected attribute to another
        dictionary of fairness metrics (SPD, impact ratio, equal
        opportunity difference, predictive equality difference).
    """
    fairness_results: Dict[str, Dict[str, float]] = {}
    # Predictions
    y_pred = model.predict(X_test)
    for attr in fairness_cols:
        groups = df_test[attr].dropna().unique()
        # Compute overall positive rate, true positives, etc.
        overall_positive_rate = y_test.mean()
        approval_rates = {}
        tpr = {}
        fpr = {}
        # For each group compute metrics
        for g in groups:
            mask = df_test[attr] == g
            group_positive_rate = y_test[mask].mean() if y_test[mask].size > 0 else np.nan
            approval_rates[g] = group_positive_rate
            # True positives and false positives for equal opportunity/predictive equality
            tp = ((y_pred == 1) & (y_test == 1) & mask).sum()
            fp = ((y_pred == 1) & (y_test == 0) & mask).sum()
            positives = (y_test == 1 & mask).sum() if y_test[mask].size > 0 else 0
            negatives = (y_test == 0 & mask).sum() if y_test[mask].size > 0 else 0
            tpr[g] = tp / positives if positives > 0 else 0
            fpr[g] = fp / negatives if negatives > 0 else 0
        # Statistical Parity Difference (difference between group approval rates and overall)
        spd = {g: (approval_rates[g] - overall_positive_rate) for g in groups}
        # Impact Ratio: ratio of group approval rate to overall rate (avoid div by 0)
        impact_ratio = {g: (approval_rates[g] / overall_positive_rate) if overall_positive_rate > 0 else np.nan for g in groups}
        # Equal Opportunity Difference: difference in TPR between highest and lowest group
        if len(groups) >= 2:
            max_tpr = max(tpr.values())
            min_tpr = min(tpr.values())
            eq_opp_diff = max_tpr - min_tpr
        else:
            eq_opp_diff = 0.0
        # Predictive Equality Difference: difference in FPR between highest and lowest group
        if len(groups) >= 2:
            max_fpr = max(fpr.values())
            min_fpr = min(fpr.values())
            pred_eq_diff = max_fpr - min_fpr
        else:
            pred_eq_diff = 0.0
        fairness_results[attr] = {
            "SPD": spd,
            "Impact Ratio": impact_ratio,
            "Equal Opportunity Diff": eq_opp_diff,
            "Predictive Equality Diff": pred_eq_diff,
        }
    return fairness_results


def main(
    dataset_path: str = "/home/oai/share/loan_approval_synthetic_dataset.csv",
    metadata_path: str = "/home/oai/share/feature_metadata.csv",
    sample_size: int = 200_000,
) -> None:
    """Run the full pipeline on the synthetic dataset.

    This function orchestrates loading the metadata and dataset, selecting
    features, training the model, evaluating predictive performance and
    computing fairness metrics.  It also prints out the metrics and
    describes next steps for governance and drift monitoring.

    Args:
        dataset_path: Path to the synthetic data file.
        metadata_path: Path to the feature metadata file.
        sample_size: Number of rows to sample from the dataset for
            modelling.  Set to None to use all rows.  Sampling helps
            keep runtime reasonable on large datasets.
    """
    # 1. Load metadata and dataset
    metadata = load_metadata(metadata_path)
    df = load_dataset(dataset_path, sample_size=sample_size)
    # 2. Prepare data (select predictors and derive features)
    categories = ["PrimaryCSV", "PrimaryCreditBureau"]
    X_train, y_train, X_test, y_test, categorical_cols, numeric_cols, fairness_cols = prepare_data(
        df, metadata, categories
    )
    # 3. Build pipeline and train model with hyper‑parameter tuning
    pipeline = build_pipeline(categorical_cols, numeric_cols)
    param_grid = {"model__C": [0.1, 1, 10], "model__penalty": ["l2"], "model__solver": ["liblinear"]}
    grid_search = train_model(pipeline, X_train, y_train, param_grid)
    best_model = grid_search.best_estimator_
    print("\nBest hyper‑parameters:", grid_search.best_params_)
    # 4. Evaluate predictive performance
    print("\n--- Predictive performance metrics ---")
    performance_results = evaluate_performance(best_model, X_test, y_test)
    # 5. Compute fairness metrics
    print("\n--- Fairness metrics ---")
    fairness_results = compute_fairness_metrics(
        best_model, X_test, y_test, df.loc[X_test.index], fairness_cols
    )
    for attr, metrics in fairness_results.items():
        print(f"\nFairness metrics for {attr}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value}")
    # 6. Governance & monitoring recommendations
    print("\n--- Governance, transparency and drift monitoring ---")
    print(
        "Document model details including training data, feature definitions, hyper‑parameters, "
        "performance and fairness metrics. Use SHAP or similar methods to explain individual "
        "predictions. Incorporate human reviewers for borderline cases and log overrides. Schedule "
        "out‑of‑time validation using the most recent data and monitor covariate and concept drift "
        "over time."
    )


if __name__ == "__main__":
    main()