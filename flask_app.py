from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ PostgreSQL DB config
DB_USER = "postgres"
DB_PASSWORD = "vinay"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "mydb1"

# ✅ Set MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

def get_latest_production_model_name(stage="Production", alias=None):
    """
    Finds the latest-registered model name in a given MLflow stage or alias.
    """
    client = MlflowClient()
    registered = client.search_registered_models()
    if not registered:
        raise RuntimeError("No models registered in MLflow!")

    candidates = []
    for m in registered:
        for lv in m.latest_versions:
            if alias:
                aliases = getattr(lv, 'aliases', [])
                if alias in aliases:
                    candidates.append((m.name, lv.version, lv.creation_timestamp))
            else:
                if lv.current_stage == stage:
                    candidates.append((m.name, lv.version, lv.creation_timestamp))

    if not candidates:
        raise ValueError(f"No model found in MLflow registry for stage='{stage}' alias='{alias}'")

    # Sort by latest version timestamp
    candidates.sort(key=lambda t: t[2], reverse=True)
    chosen_model = candidates[0][0]
    print(f"✅ Will load model: {chosen_model} (version {candidates[0][1]})")
    return chosen_model

def load_model_from_registry(stage="Production", alias=None):
    model_name = get_latest_production_model_name(stage=stage, alias=alias)
    model_uri = f"models:/{model_name}/{alias or stage}"
    print(f"📦 Loading model from URI: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "❌ No file part in request."

    file = request.files['file']
    if file.filename == '':
        return "❌ No selected file."

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 🧹 Clean the data
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        df.drop(columns=['prospect_id', 'lead_number'], inplace=True, errors='ignore')
        df.replace(["Select", "", "None"], np.nan, inplace=True)
    except Exception as e:
        return f"❌ Failed to read or clean CSV: {e}"

    # ✅ Binary columns mapping
    binary_cols = [
        "do_not_email", "do_not_call", "search", "magazine", "newspaper_article",
        "newspaper", "digital_advertisement", "through_recommendations",
        "receive_more_updates_about_our_courses", "get_updates_on_dm_content",
        "i_agree_to_pay_the_amount_through_cheque", "a_free_copy_of_mastering_the_interview"
    ]
    binary_map = {"Yes": 1, "No": 0}
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # ✅ Load model automatically from MLflow Registry
    try:
        model_pipeline = load_model_from_registry(stage="Production")
        print("✅ Model loaded successfully.")
    except Exception as e:
        return f"❌ Failed to load model from MLflow Registry: {e}"

    # 🔍 Predict
    try:
        preds = model_pipeline.predict(df)
        df['prediction'] = preds.astype(int)  # binary output
    except Exception as e:
        return f"❌ Prediction failed: {e}"

    # ✅ Store in PostgreSQL `lead_scoring` table
    try:
        db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        df.to_sql("lead_scoring", engine, if_exists="append", index=False)
        print("✅ Data saved to 'lead_scoring' table")
    except Exception as e:
        return f"❌ Failed to save to PostgreSQL: {e}"

    return render_template(
        'results.html',
        table_name="lead_scoring",
        tables=[df.head().to_html(classes='data', header=True, index=False)]
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
