#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loan Approval Model - L2 Balanced Fast Grid + Threshold Constraints

- Only L2 penalty
- class_weight='balanced' (fixed)
- Narrow, fast grid (lbfgs/saga × C in {0.01, 0.1, 1.0, 10.0})
- Higher max_iter for convergence
- Threshold tuning to require precision >= MIN_PRECISION and recall >= MIN_RECALL (if achievable)
"""

import json, pickle, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, classification_report, precision_recall_curve)

# ---------------- Config ----------------
DEFAULT_DATA = "Data/Loan_dataset_india_110000_final_update copy(in).csv"
DEFAULT_TARGET = "credit_approved"
FEATURE_CLASSIFICATION: Optional[str] = "data/feature_classification.csv"
OUT_DIR = "Data"

# speed/quality knobs
CV_FOLDS = 5  # faster
FAST_MODE = False        # if True -> only lbfgs (4 combos). If False -> lbfgs & saga (8 combos)
MAX_ITER = 1000 # higher convergence budget
C_GRID = [0.1, 1, 10]

# threshold constraints
MIN_PRECISION = 0.75
MIN_RECALL = 0.5

RANDOM_STATE = 45
TEST_SIZE = 0.3
SMOKE = False
SAMPLE_NROWS: Optional[int] = None

# --------------- Utils ------------------
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.05)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def read_feature_classification(csv_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not csv_path: return None
    p = Path(csv_path)
    if not p.exists(): return None
    df = pd.read_csv(p)
    colmap = {c.lower(): c for c in df.columns}
    feat = colmap.get("feature", list(df.columns)[0])
    cls  = colmap.get("classification", list(df.columns)[1])
    df = df.rename(columns={feat:"Feature", cls:"Classification"})
    df["Feature"] = df["Feature"].astype(str)
    df["Classification"] = (df["Classification"].astype(str).str.strip().str.lower()
                            .map({"protected":"protected","derived":"derived","given":"given"}).fillna("given"))
    return df


# Tag features: Protected, Derived and Given 
def tag_features_from_classification(all_cols: List[str], fc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    tag_map = {} if fc_df is None else dict(zip(fc_df["Feature"].astype(str), fc_df["Classification"]))
    rows = []
    for c in all_cols:
        t = tag_map.get(c, "given")
        tag = t if t in {"given","derived","protected"} else "given"
        rows.append({"feature": c, "tag": tag, "raw_tag": t})
    return pd.DataFrame(rows)


def apply_feature_tags_and_filter(df: pd.DataFrame, target: str, fc_path: Optional[str]):
    cols = [c for c in df.columns if c != target]
    tags = tag_features_from_classification(cols, read_feature_classification(fc_path))
    allowed = tags[tags["tag"].isin(["given","derived"])]["feature"].tolist()
    protected = tags[tags["tag"]=="protected"]["feature"].tolist()
    out = pd.concat([df[allowed], df[target]], axis=1)
    tags["used_in_training"] = tags["feature"].isin(allowed)
    return out, tags, allowed, protected



# Handle Outliers
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self): self.bounds_ = []
    def fit(self, X, y=None):
        A = self._arr(X); self.bounds_ = []
        for i in range(A.shape[1]):
            col = A[:, i]; mask = ~np.isnan(col); v = col[mask]
            if v.size == 0: self.bounds_.append((None,None)); continue
            skew = pd.Series(v).skew()
            if abs(skew) < 1:
                m, s = float(np.mean(v)), float(np.std(v, ddof=0))
                low, high = (m, m) if (s==0 or np.isnan(s)) else (m-3*s, m+3*s)
            else:
                q1, q3 = np.percentile(v,25), np.percentile(v,76); iqr = q3-q1
                low, high = (q1,q3) if iqr==0 else (q1-1.5*iqr, q3+1.5*iqr)
            self.bounds_.append((low, high))
        return self
    def transform(self, X):
        A = self._arr(X); B = A.copy()
        for i,(lo,hi) in enumerate(self.bounds_):
            if lo is not None and hi is not None: B[:,i] = np.clip(A[:,i], lo, hi)
        return B
    @staticmethod
    def _arr(X): return X.values.astype(float) if isinstance(X,pd.DataFrame) else np.asarray(X, dtype=float)

# Get Feature type from datatype
def infer_feature_types(df: pd.DataFrame, target: str):
    cols = [c for c in df.columns if c != target]
    num = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in cols if c not in num]
    return num, cat


# Create pipeline for Preprocessing Steps 
def build_preprocessor(num_cols, cat_cols):
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("cap", OutlierCapper()),
                       ("scale", RobustScaler())])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("ohe", make_ohe())])
    return ColumnTransformer([("num", num_tf, num_cols),
                              ("cat", cat_tf, cat_cols)], remainder="drop")

# Apply Pipeline/ Integrate pipeline with model
def build_pipeline(prep):
    clf = LogisticRegression(penalty="l2",
                             class_weight="balanced",
                             max_iter=MAX_ITER,
                             random_state=RANDOM_STATE)
    return Pipeline([("prep", prep), ("clf", clf)])

# Get Parameters 
def get_param_grid():
    solvers = ["lbfgs"] if FAST_MODE else ["lbfgs","saga"]
    return [{
        "clf__solver": solvers,
        "clf__penalty": ["l2", "l1"],
        "clf__C": C_GRID,
        "clf__class_weight": ["balanced"],
        "clf__max_iter": [MAX_ITER]
    }]

def choose_threshold(y_true, y_proba, min_p, min_r):
    p, r, thr = precision_recall_curve(y_true, y_proba)
    cand = []
    for pi, ri, ti in zip(p[:-1], r[:-1], thr):
        if pi >= min_p and ri >= min_r:
            f1 = 2*pi*ri/(pi+ri+1e-12); cand.append((ti,pi,ri,f1))
    if cand:
        ti,pi,ri,f1 = max(cand, key=lambda x:x[3])
        return {"threshold": float(ti), "precision": float(pi), "recall": float(ri),
                "f1": float(f1), "constraints_met": True}
    f1s = [2*pi*ri/(pi+ri+1e-12) for pi,ri in zip(p[:-1], r[:-1])]
    bi = int(np.argmax(f1s)) if f1s else 0
    bt = float(thr[bi]) if len(thr)>0 else 0.5
    return {"threshold": bt, "precision": float(p[bi]), "recall": float(r[bi]),
            "f1": float(f1s[bi]) if f1s else 0.0, "constraints_met": False}

# Evalualtion metric 
def evaluate(y_true, y_pred, y_proba=None):
    m = {"f1": f1_score(y_true,y_pred,zero_division=0),
         "precision": precision_score(y_true,y_pred,zero_division=0),
         "recall": recall_score(y_true,y_pred,zero_division=0)}
    if y_proba is not None:
        try: m["roc_auc"] = roc_auc_score(y_true,y_proba)
        except: m["roc_auc"] = None
        try: m["pr_auc"] = average_precision_score(y_true,y_proba)
        except: m["pr_auc"] = None
    return m


# Main function to run model, here calling supporing functions in main function
def main(data_path=DEFAULT_DATA, target=DEFAULT_TARGET, feature_classification=FEATURE_CLASSIFICATION):
    data_path = Path(data_path); assert data_path.exists(), f"Missing data: {data_path}"
    read_kwargs = {}
    # Read data 
    df = pd.read_csv(data_path)
    print("Shape of DF",df.shape)
    df.columns = [str(c).strip() for c in df.columns]
    if target not in df.columns: raise ValueError(f"Target '{target}' not in data.")

    # Target cleaning
    y_raw = df[target]
    if not pd.api.types.is_numeric_dtype(y_raw):
        mapping = {"y":1,"yes":1,"true":1,"t":1,"approved":1,"1":1,"n":0,"no":0,"false":0,"f":0,"rejected":0,"0":0}
        y_series = y_raw.astype(str).str.strip().str.lower().map(mapping)
    else:
        y_series = pd.to_numeric(y_raw, errors="coerce")
    mask = y_series.notna()
    if mask.sum()<len(y_series): print(f"[warn] dropped {len(y_series)-mask.sum()} rows with invalid target")
    df = df.loc[mask].copy(); y = y_series.loc[mask].astype(int)
    print("Dataframe after cleaning:", df.shape)

    # Get dataframe, allowed protected features by callinig apply_feature_tags_and_filter
    df_f, tags, allowed, protected = apply_feature_tags_and_filter(df, target, feature_classification)


    X = df_f.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE, stratify=y)
    num, cat = infer_feature_types(X_train, target="")

    # below 3 functions are to create preprocesing, and fet parameters 
    prep = build_preprocessor(num, cat)
    pipe = build_pipeline(prep)
    grid = get_param_grid()

    min_class = int(pd.Series(y_train).value_counts().min())
    n_splits = max(4, min(CV_FOLDS, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    combos = len(grid[0]["clf__solver"]) * len(grid[0]["clf__C"])
    print(f"[info] Effective grid combos: {combos}  |  CV folds: {n_splits}  |  total fits: {combos*n_splits}")

    # Grid search CV to get best estimator
    gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    #  best estimator
    best = gs.best_estimator_


    y_proba = best.predict_proba(X_test)[:,1] if hasattr(best,"predict_proba") else None
    y_pred = best.predict(X_test)
    tinfo = None
    if y_proba is not None:
        tinfo = choose_threshold(y_test, y_proba, MIN_PRECISION, MIN_RECALL)
        y_pred = (y_proba >= tinfo["threshold"]).astype(int)

    mets = evaluate(y_test, y_pred, y_proba)
    print("\n[report] Post-threshold classification report")
    print(classification_report(y_test, y_pred, zero_division=0))

    out = {
        "best_params": gs.best_params_,
        "best_score_mean_cv_f1": float(gs.best_score_),
        "threshold_constraints": {"min_precision": MIN_PRECISION, "min_recall": MIN_RECALL, "tuning": tinfo},
        "test_metrics": mets,
        "protected_features_excluded": protected,
        "allowed_features_count": len(allowed),
        "solvers": grid[0]["clf__solver"],
        "C_grid": grid[0]["clf__C"],
        "max_iter": MAX_ITER,
        "cv_folds": n_splits,
        "fast_mode": FAST_MODE
    }

    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"feature_tags_fast.csv").write_text(tags.to_csv(index=False))
    with open(out_dir/"run_artifacts_fast.json","w") as f: json.dump(out, f, indent=2)
    with open(out_dir/"best_model_fast.pkl","wb") as f: pickle.dump(best, f)

    print("\n=== Summary ===")
    print(json.dumps(out, indent=2))
    if tinfo and not tinfo.get("constraints_met", False):
        print(f"\n[warn] Could not meet precision>={MIN_PRECISION:.2f} & recall>={MIN_RECALL:.2f} at any threshold. "
              f"Best: P={tinfo['precision']:.3f}, R={tinfo['recall']:.2f}.")

if __name__ == "__main__":
    main()
