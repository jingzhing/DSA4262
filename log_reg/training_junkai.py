import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
from json_parse import parse

# --- Load features ---
keys, X = parse("dataset0.json.gz")

# --- Load labels ---
labels = pd.read_csv("data.info.labelled")
y_map = {(row.transcript_id, row.transcript_position): (row.label, row.gene_id)
         for row in labels.itertuples(index=False)}

y = []
gene_ids = []
for k in keys:
    label, gene = y_map.get(k, (0, "unknown"))
    y.append(label)
    gene_ids.append(gene)

X = np.array(X)
y = np.array(y)
gene_ids = np.array(gene_ids)

# --- Train/validation split by gene_id ---
unique_genes = np.unique(gene_ids)
np.random.seed(42)
np.random.shuffle(unique_genes)
n_train = int(0.8 * len(unique_genes))
train_genes = set(unique_genes[:n_train])
val_genes   = set(unique_genes[n_train:])

train_idx = [i for i, g in enumerate(gene_ids) if g in train_genes]
val_idx   = [i for i, g in enumerate(gene_ids) if g in val_genes]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print(f"Training sites: {len(X_train)}, Validation sites: {len(X_val)}")

# --- Scale features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)


clf = LogisticRegression(
    max_iter=1000,       # more iterations to ensure convergence
    class_weight='balanced',  # handle label imbalance
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
y_proba = clf.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_proba))

joblib.dump(clf, "logreg_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Model and scaler saved!")
