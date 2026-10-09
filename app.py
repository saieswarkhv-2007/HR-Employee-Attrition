import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "attrition_model.joblib"
DATASET_PATH = BASE_DIR / "dataset" / "HR_Employee_Attrition_raw.csv"

DEFAULT_PREDICTION = {
    "Age": 34,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 6,
    "Education": 4,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EmployeeNumber": 100,
    "EnvironmentSatisfaction": 3,
    "Gender": "Female",
    "HourlyRate": 42,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 4,
    "MaritalStatus": "Married",
    "MonthlyIncome": 5500,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "Over18": "Y",
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StandardHours": 80,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 4,
    "YearsInCurrentRole": 2,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3,
}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _normalize_prediction_input(payload):
    data = {**DEFAULT_PREDICTION, **payload}
    normalized = {}
    for key, value in data.items():
        if value is None or value == "":
            normalized[key] = DEFAULT_PREDICTION.get(key, 0)
        else:
            normalized[key] = value
    return normalized


def ensure_model_exists():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return MODEL_PATH

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if "Attrition" not in df.columns:
        raise ValueError("Target column 'Attrition' not found in dataset.")

    X = df.drop(columns=["Attrition"])
    y = (df["Attrition"] == "Yes").astype(int)

    numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    return MODEL_PATH


def predict_attrition(payload):
    ensure_model_exists()
    model = joblib.load(MODEL_PATH)
    normalized = _normalize_prediction_input(payload)

    row = {}
    numeric_columns = {
        "Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeCount",
        "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
        "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
        "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
        "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    }

    for key, value in normalized.items():
        if key in numeric_columns:
            row[key] = _safe_int(value, 0)
        else:
            row[key] = str(value)

    frame = pd.DataFrame([row])
    probability = float(model.predict_proba(frame)[0, 1])
    label = "High risk" if probability >= 0.5 else "Low risk"

    if probability >= 0.7:
        recommendation = "Immediate review is recommended. Focus on workload balance, compensation, and retention planning."
    elif probability >= 0.5:
        recommendation = "This employee needs closer monitoring. Review overtime, satisfaction, and career growth actions."
    else:
        recommendation = "The employee appears stable and likely to remain. Maintain a strong growth and engagement strategy."

    result = {
        "risk_probability": round(probability, 4),
        "risk_percent": round(probability * 100, 2),
        "label": label,
        "recommendation": recommendation,
    }
    return result


def _collect_output_images():
    groups = []
    for folder in sorted(OUTPUTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        images = []
        for file_path in sorted(folder.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                relative = file_path.relative_to(BASE_DIR).as_posix()
                images.append({
                    "name": file_path.name,
                    "preview_url": f"/outputs/{relative.split('outputs/', 1)[1]}",
                    "download_url": f"/download-output/{relative.split('outputs/', 1)[1]}",
                })
        if images:
            groups.append({
                "name": folder.name.replace("_", " "),
                "images": images,
            })
    return groups


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/visualization")
def visualization():
    return render_template("visualization.html", image_groups=_collect_output_images())


@app.route("/outputs/<path:filename>")
def serve_output_file(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/download-output/<path:filename>")
def download_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=True)


@app.route("/models")
def models():
    return render_template("models.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    defaults = DEFAULT_PREDICTION.copy()
    result = None
    if request.method == "POST":
        data = request.form.to_dict()
        defaults.update(data)
        result = predict_attrition(data)
    return render_template("prediction.html", result=result, defaults=defaults)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    return jsonify(predict_attrition(payload))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/reports")
def reports():
    return render_template("reports.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    ensure_model_exists()
    app.run(debug=True)
