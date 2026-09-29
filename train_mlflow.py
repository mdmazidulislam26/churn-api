import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

MODEL_NAME = "telco-churn"
THRESHOLD = 0.30

mlflow.set_tracking_uri("sqlite:////content/churn_api/mlflow.db")
mlflow.set_experiment("telco-churn-experiments")

# Link every run to the exact data version tracked by DVC
with open("data/telco.csv.dvc") as f:
    data_md5 = yaml.safe_load(f)["outs"][0]["md5"]

df = pd.read_csv("data/telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn", "customerID"])
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat_cols = [c for c in X.columns if c not in num_cols]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def build_pipeline(max_depth, lr, n_estimators):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="constant", fill_value=0), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    clf = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        subsample=0.8, eval_metric="logloss", random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


configs = [(3, 0.05, 200), (4, 0.05, 300), (5, 0.03, 400)]
results = []

for depth, lr, n_est in configs:
    with mlflow.start_run(run_name=f"xgb_d{depth}_lr{lr}_n{n_est}") as run:
        pipe = build_pipeline(depth, lr, n_est)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_te, proba),
            "pr_auc": average_precision_score(y_te, proba),
            "recall": recall_score(y_te, pred),
            "precision": precision_score(y_te, pred),
        }
        mlflow.log_params({
            "max_depth": depth, "learning_rate": lr, "n_estimators": n_est,
            "threshold": THRESHOLD, "data_md5": data_md5,
        })
        mlflow.log_metrics(metrics)

        signature = infer_signature(
            X_tr.head(50), pipe.predict_proba(X_tr.head(50))[:, 1]
        )
        # cloudpickle avoids the skops "untrusted types" error with XGBoost
        mlflow.sklearn.log_model(
            pipe, name="model", signature=signature,
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
        )
        results.append((run.info.run_id, metrics["pr_auc"], metrics["roc_auc"]))
        print(
            f"depth={depth} lr={lr} n={n_est} -> "
            f"roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f} "
            f"recall={metrics['recall']:.4f} precision={metrics['precision']:.4f}"
        )

# Best run by PR-AUC (better for imbalanced data) becomes the champion
best_run_id, best_pr, _ = max(results, key=lambda r: r[1])
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# Tag every version so rollback to any version works
for v in versions:
    client.set_model_version_tag(MODEL_NAME, v.version, "threshold", str(THRESHOLD))

best_version = next(v.version for v in versions if v.run_id == best_run_id)
client.set_registered_model_alias(MODEL_NAME, "champion", best_version)
print(f"champion -> version {best_version} (pr_auc={best_pr:.4f})")
