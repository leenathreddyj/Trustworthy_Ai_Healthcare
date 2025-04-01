# model/shap_explain.py

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('data/heart_disease_uci.csv')
df.drop(['id', 'dataset'], axis=1, inplace=True)
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
df.rename(columns={'num': 'target'}, inplace=True)
df['target'] = df['target'].apply(lambda x: 1 if int(x) > 0 else 0)
df.dropna(inplace=True)

# Encode categorical columns
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
df = pd.get_dummies(df, columns=categorical_cols)

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# SHAP LinearExplainer for linear models
explainer = shap.LinearExplainer(model, X_scaled, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_scaled)

# Plot SHAP summary
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("app/static/shap_summary.png")  # ✅ Save into app/static/
print("✅ SHAP summary plot saved to app/static/shap_summary.png")
