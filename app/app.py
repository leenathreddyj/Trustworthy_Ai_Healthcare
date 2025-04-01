from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import datetime
import os
import requests
import base64
import time
from fpdf import FPDF

app = Flask(__name__)

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL = "leenathreddy@gmail.com"

feature_columns = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
    'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'restecg_left ventricular hypertrophy', 'restecg_normal', 'restecg_ST-T wave abnormality',
    'slope_downsloping', 'slope_flat', 'slope_upsloping',
    'thal_fixed defect', 'thal_normal', 'thal_reversable defect'
]

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

    model_baseline = LogisticRegression(max_iter=1000)
    model_baseline.fit(X_scaled, y)

    model_fair = LogisticRegression(max_iter=1000)
    model_fair.fit(X_scaled, y, sample_weight=bld_rw.instance_weights)

    return model_baseline, model_fair, scaler

model_baseline, model_fair, scaler = load_models()

@app.route('/', methods=['GET', 'POST'])
def index():
    biased_prediction = None
    fair_prediction = None
    changed = False
    email = None

    if request.method == 'POST':
        form = request.form
        email = form.get('email')

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

        timestamp = datetime.datetime.now().isoformat()
        csv_path = "app/static/predictions_log.csv"
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp] + list(input_data.values()) + [biased_prediction, fair_prediction, changed])

        explainer = shap.LinearExplainer(model_fair, input_scaled)
        shap_values = explainer.shap_values(input_scaled)

        plt.clf()
        shap.summary_plot(shap_values, features=input_df, feature_names=feature_columns, show=False)
        plt.tight_layout()
        shap_path = "app/static/shap_force.png"
        plt.savefig(shap_path)

        pdf_path = "app/static/prediction_report.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Heart Disease Prediction Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Timestamp: {timestamp}", ln=True)
        pdf.cell(200, 10, txt=f"Biased Prediction: {biased_prediction}", ln=True)
        pdf.cell(200, 10, txt=f"Fair Prediction: {fair_prediction}", ln=True)
        pdf.cell(200, 10, txt=f"Prediction Changed: {changed}", ln=True)
        pdf.output(pdf_path)

        if email and SENDGRID_API_KEY:
            with open(shap_path, 'rb') as f:
                image_data = f.read()
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            with open(csv_path, 'rb') as f:
                csv_data = f.read()

            encoded_img = base64.b64encode(image_data).decode('utf-8')
            encoded_pdf = base64.b64encode(pdf_data).decode('utf-8')
            encoded_csv = base64.b64encode(csv_data).decode('utf-8')

            data = {
                "personalizations": [
                    {
                        "to": [{"email": email}],
                        "subject": "Your Heart Disease Prediction Report"
                    }
                ],
                "from": {
                    "email": SENDER_EMAIL,
                    "name": "Heart Disease Predictor"
                },
                "reply_to": {
                    "email": SENDER_EMAIL
                },
                "content": [
                    {
                        "type": "text/plain",
                        "value": f"""
Biased Prediction: {biased_prediction}
Fair Prediction: {fair_prediction}
Prediction Changed: {changed}

Attached are your PDF report, SHAP explanation, and prediction log.
                        """
                    }
                ],
                "attachments": [
                    {
                        "content": encoded_img,
                        "type": "image/png",
                        "filename": "shap_force.png"
                    },
                    {
                        "content": encoded_pdf,
                        "type": "application/pdf",
                        "filename": "prediction_report.pdf"
                    },
                    {
                        "content": encoded_csv,
                        "type": "text/csv",
                        "filename": "predictions_log.csv"
                    }
                ]
            }
            headers = {
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json"
            }
            time.sleep(2)
            response = requests.post("https://api.sendgrid.com/v3/mail/send", json=data, headers=headers)
            print("SendGrid response:", response.status_code, response.text)

    return render_template("index.html", biased_prediction=biased_prediction, fair_prediction=fair_prediction, changed=changed)

if __name__ == '__main__':
    app.run(debug=True)
