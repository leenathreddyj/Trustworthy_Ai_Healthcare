from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid macOS crash
import matplotlib.pyplot as plt
import csv
import datetime

app = Flask(__name__)

# Define columns expected by the model (in same order)
feature_columns = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
    'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'restecg_left ventricular hypertrophy', 'restecg_normal', 'restecg_ST-T wave abnormality',
    'slope_downsloping', 'slope_flat', 'slope_upsloping',
    'thal_fixed defect', 'thal_normal', 'thal_reversable defect'
]

# Load both baseline and fair models
def load_models():
    df = pd.read_csv('data/heart_disease_uci.csv')
    df.drop(['id', 'dataset'], axis=1, inplace=True)
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df.rename(columns={'num': 'target'}, inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if int(x) > 0 else 0)
    df.dropna(inplace=True)

    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    df = pd.get_dummies(df, columns=categorical_cols)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_columns]
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Baseline model
    model_baseline = LogisticRegression(max_iter=1000)
    model_baseline.fit(X_scaled, y)

    # Fair model (reweighted)
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing

    def to_aif360(X_df, y_series):
        df_temp = X_df.copy()
        df_temp['target'] = y_series
        return BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df_temp,
            label_names=['target'],
            protected_attribute_names=['sex']
        )

    bld_train = to_aif360(X, y)
    rw = Reweighing(unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
    rw.fit(bld_train)
    bld_rw = rw.transform(bld_train)

    model_fair = LogisticRegression(max_iter=1000)
    model_fair.fit(X_scaled, y, sample_weight=bld_rw.instance_weights)

    return model_baseline, model_fair, scaler

model_baseline, model_fair, scaler = load_models()

@app.route('/', methods=['GET', 'POST'])
def index():
    biased_prediction = None
    fair_prediction = None
    changed = False
    bias_prob = 0.0
    fair_prob = 0.0

    if request.method == 'POST':
        form = request.form

        input_data = {
            'age': float(form['age']),
            'sex': 1 if form['sex'] == 'Male' else 0,
            'trestbps': float(form['trestbps']),
            'chol': float(form['chol']),
            'fbs': float(form['fbs']),
            'thalach': float(form['thalach']),
            'exang': float(form['exang']),
            'oldpeak': float(form['oldpeak']),
            'ca': float(form['ca']),
        }

        cp = f"cp_{form['cp']}"
        restecg = f"restecg_{form['restecg']}"
        slope = f"slope_{form['slope']}"
        thal = f"thal_{form['thal']}"

        input_vector = {col: 0 for col in feature_columns}
        input_vector.update(input_data)
        input_vector[cp] = 1
        input_vector[restecg] = 1
        input_vector[slope] = 1
        input_vector[thal] = 1

        input_df = pd.DataFrame([input_vector])
        input_scaled = scaler.transform(input_df)

        prob_biased = model_baseline.predict_proba(input_scaled)[0][1]
        prob_fair = model_fair.predict_proba(input_scaled)[0][1]

        result_biased = model_baseline.predict(input_scaled)[0]
        result_fair = model_fair.predict(input_scaled)[0]

        biased_prediction = f"{'High' if result_biased else 'Low'} Risk of Heart Disease ({prob_biased:.2f})"
        fair_prediction = f"{'High' if result_fair else 'Low'} Risk of Heart Disease ({prob_fair:.2f})"
        changed = result_biased != result_fair

        # Log to CSV
        timestamp = datetime.datetime.now().isoformat()
        log_data = {
            "timestamp": timestamp,
            "input": input_data,
            "biased_prediction": biased_prediction,
            "fair_prediction": fair_prediction,
            "changed": changed
        }
        with open("app/static/predictions_log.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp] + list(input_data.values()) + [biased_prediction, fair_prediction, changed])

        # SHAP explanation (summary plot for this input)
        explainer = shap.LinearExplainer(model_fair, input_scaled)
        shap_values = explainer.shap_values(input_scaled)

        plt.clf()
        shap.summary_plot(shap_values, features=input_df, feature_names=feature_columns, show=False)
        plt.tight_layout()
        plt.savefig("app/static/shap_force.png")

    return render_template("index.html", biased_prediction=biased_prediction, fair_prediction=fair_prediction, changed=changed)

if __name__ == '__main__':
    app.run(debug=True)
