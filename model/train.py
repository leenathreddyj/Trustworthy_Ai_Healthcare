# model/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric

# Load dataset
df = pd.read_csv('data/heart_disease_uci.csv')

# Drop unnecessary columns
df.drop(['id', 'dataset'], axis=1, inplace=True)

# Map sex to binary: Male = 1, Female = 0
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# Rename 'num' column to 'target'
df.rename(columns={'num': 'target'}, inplace=True)

# Convert target to binary (presence of heart disease)
df['target'] = df['target'].apply(lambda x: 1 if int(x) > 0 else 0)

# Drop missing values
df.dropna(inplace=True)

# Encode categorical features using one-hot encoding
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
df = pd.get_dummies(df, columns=categorical_cols)

# Split features and target
X = df.drop('target', axis=1)
y = df['target']


# Protected attribute
protected_attr = 'sex'

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Baseline Logistic Regression ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("=== Baseline Model ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# === Helper to convert to AIF360 format ===
def create_aif360_dataset(X_df, y_series, protected_attr):
    df_combined = X_df.copy()
    df_combined['target'] = y_series
    return BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df_combined,
        label_names=['target'],
        protected_attribute_names=[protected_attr]
    )


# Convert to AIF360 datasets
train_bld = create_aif360_dataset(X_train, y_train, protected_attr)
test_bld = create_aif360_dataset(X_test, y_test, protected_attr)

# === Reweighing for Bias Mitigation ===
RW = Reweighing(
    unprivileged_groups=[{protected_attr: 0}],  # Female
    privileged_groups=[{protected_attr: 1}]     # Male
)
RW.fit(train_bld)
train_reweighted = RW.transform(train_bld)

# Train reweighed model
model_rw = LogisticRegression(max_iter=1000)
model_rw.fit(X_train_scaled, y_train, sample_weight=train_reweighted.instance_weights)
y_rw_pred = model_rw.predict(X_test_scaled)

print("\n=== Reweighed Model ===")
print("Accuracy:", accuracy_score(y_test, y_rw_pred))
print(classification_report(y_test, y_rw_pred))

# === Fairness Evaluation ===
# Baseline model fairness
metric_original = BinaryLabelDatasetMetric(
    test_bld,
    unprivileged_groups=[{protected_attr: 0}],
    privileged_groups=[{protected_attr: 1}]
)
print("\nDisparate Impact (original):", metric_original.disparate_impact())

# Fairness after reweighing
test_bld_pred = test_bld.copy()
test_bld_pred.labels = y_rw_pred.reshape(-1, 1)

metric_reweighed = BinaryLabelDatasetMetric(
    test_bld_pred,
    unprivileged_groups=[{protected_attr: 0}],
    privileged_groups=[{protected_attr: 1}]
)
print("Disparate Impact (reweighed):", metric_reweighed.disparate_impact())
