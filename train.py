import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
THRESHOLD = 0.30  # decision threshold chosen in Milestone 2 (recall-oriented)

df = pd.read_csv(URL)
# Blank TotalCharges means tenure == 0, so we treat it as missing and impute 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn", "customerID"])

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="constant", fill_value=0), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipe = Pipeline([
    ("pre", pre),
    ("clf", XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, eval_metric="logloss", random_state=42,
    )),
])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipe.fit(X_tr, y_tr)

proba = pipe.predict_proba(X_te)[:, 1]
pred = (proba >= THRESHOLD).astype(int)
metrics = {
    "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
    "recall": round(float(recall_score(y_te, pred)), 4),
    "precision": round(float(precision_score(y_te, pred)), 4),
}

joblib.dump(pipe, "artifacts/model.joblib")
meta = {
    "model_name": "telco-churn-xgb",
    "model_version": "1.0.0",
    "threshold": THRESHOLD,
    "feature_columns": list(X.columns),
    "metrics": metrics,
}
with open("artifacts/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved model + metadata")
print(metrics)
