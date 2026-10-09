# ================================================================
# XGBOOST - HR EMPLOYEE ATTRITION PREDICTION
# PyCharm Program
#
# Original / Main Dataset is NOT modified
# All outputs are stored in ONE folder
# ================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from xgboost import XGBClassifier

# ================================================================
# 1. FILE PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "final_preprocess.csv"
)

if not os.path.exists(INPUT_FILE):
    INPUT_FILE = os.path.join(
        BASE_DIR,
        "dataset",
        "HR_Employee_Attrition_raw.csv"
    )

# ONE output folder for everything
OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "XGBoost_Classifier_Outputs"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================================================================
# 2. LOAD DATASET
# ================================================================

print("=" * 70)
print("XGBOOST HR EMPLOYEE ATTRITION PREDICTION")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n"
        "Please check the INPUT_FILE path."
    )

# Read-only operation
original_data = pd.read_csv(INPUT_FILE)

# Create an independent copy in memory
# The original CSV is never modified
data = original_data.copy()

print("\nDataset loaded successfully.")
print("Dataset shape:", data.shape)

# ================================================================
# 3. DISPLAY DATASET INFORMATION
# ================================================================

print("\nDataset columns:")
print(data.columns.tolist())
print("\nFirst 5 records:")
print(data.head())
print("\nMissing values:")
print(data.isnull().sum())

# ================================================================
# 4. IDENTIFY TARGET COLUMN
# ================================================================

target_candidates = [
    "Attrition",
    "attrition",
    "Target",
    "target",
    "Placement",
    "Placed",
    "placement",
    "placed",
    "Status",
    "status",
    "PlacementStatus",
    "Placement_Status"
]

target_column = None
for column in target_candidates:
    if column in data.columns:
        target_column = column
        break

# If target is not found automatically, the last column is considered the target.
if target_column is None:
    target_column = data.columns[-1]
    print(
        "\nTarget column was not automatically detected."
    )
    print(
        "Using last column as target:",
        target_column
    )

print("\nTarget column:", target_column)

# ================================================================
# 5. REMOVE ONLY ROWS WITH MISSING TARGET
# ================================================================

data = data.dropna(
    subset=[target_column]
).copy()

# ================================================================
# 6. SEPARATE FEATURES AND TARGET
# ================================================================

X = data.drop(
    columns=[target_column]
).copy()
y = data[target_column].copy()

# ================================================================
# 7. HANDLE CATEGORICAL FEATURES
# ================================================================

print("\nProcessing categorical features...")
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()
numeric_columns = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

print("\nNumerical columns:")
print(numeric_columns)
print("\nCategorical columns:")
print(categorical_columns)

# Convert categorical variables into numerical values using pandas factorize
for column in categorical_columns:
    X[column] = X[column].astype(str)
    X[column], _ = pd.factorize(
        X[column]
    )

# ================================================================
# 8. HANDLE MISSING VALUES IN FEATURES
# ================================================================

for column in X.columns:
    if X[column].isnull().any():
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].fillna(X[column].median())
        else:
            X[column] = X[column].fillna(X[column].mode()[0])

# ================================================================
# 9. ENCODE TARGET
# ================================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(
    y.astype(str)
)
class_names = label_encoder.classes_
number_of_classes = len(class_names)

print("\nTarget classes:")
for i, class_name in enumerate(class_names):
    print(f"{i} = {class_name}")

# ================================================================
# 10. SAVE TARGET ENCODING INFORMATION
# ================================================================

target_encoding = pd.DataFrame({
    "Encoded_Value": range(len(class_names)),
    "Original_Class": class_names
})
target_encoding.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "target_encoding.csv"
    ),
    index=False
)

# ================================================================
# 11. TRAIN TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ================================================================
# 12. CREATE XGBOOST MODEL
# ================================================================

print("\nCreating XGBoost model...")

if number_of_classes == 2:
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
else:
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=number_of_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

# ================================================================
# 13. TRAIN XGBOOST
# ================================================================

print("\nTraining XGBoost model...")
xgb_model.fit(
    X_train,
    y_train
)
print("Training completed successfully.")

# ================================================================
# 14. PREDICTIONS
# ================================================================

y_pred = xgb_model.predict(
    X_test
)
y_probability = xgb_model.predict_proba(
    X_test
)

# ================================================================
# 15. PERFORMANCE METRICS
# ================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n")
print("=" * 70)
print("XGBOOST PERFORMANCE")
print("=" * 70)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ================================================================
# 16. SAVE PERFORMANCE METRICS
# ================================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})
metrics.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "xgboost_metrics.csv"
    ),
    index=False
)

# ================================================================
# 17. CLASSIFICATION REPORT
# ================================================================

classification_report_text = classification_report(
    y_test,
    y_pred,
    target_names=[str(c) for c in class_names],
    zero_division=0
)
print("\nClassification Report:")
print(classification_report_text)

with open(
    os.path.join(
        OUTPUT_FOLDER,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as file:
    file.write("XGBOOST CLASSIFICATION REPORT\n")
    file.write("=" * 60 + "\n\n")
    file.write(classification_report_text)

# ================================================================
# 18. ACTUAL VS PREDICTED
# ================================================================

actual_labels = label_encoder.inverse_transform(y_test)
predicted_labels = label_encoder.inverse_transform(y_pred)

prediction_results = pd.DataFrame({
    "Actual": actual_labels,
    "Predicted": predicted_labels,
    "Correct": (actual_labels == predicted_labels)
})

# Add probabilities
for i, class_name in enumerate(class_names):
    prediction_results[
        "Probability_" + str(class_name)
    ] = y_probability[:, i]

prediction_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "xgboost_predictions.csv"
    ),
    index=False
)

# ================================================================
# 19. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)
cm_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.csv"
    )
)

fig, ax = plt.subplots(figsize=(8, 6))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)
plt.title("XGBoost - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ================================================================
# 20. FEATURE IMPORTANCE
# ================================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})
feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

feature_importance.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "xgboost_feature_importance.csv"
    ),
    index=False
)

# ================================================================
# 21. FEATURE IMPORTANCE GRAPH
# ================================================================

top_n = min(20, len(feature_importance))
top_features = (
    feature_importance
    .head(top_n)
    .sort_values(
        by="Importance"
    )
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("XGBoost - Top Feature Importance")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "xgboost_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ================================================================
# 22. ROC CURVE
# ================================================================

try:
    plt.figure(figsize=(8, 6))
    if number_of_classes == 2:
        fpr, tpr, _ = roc_curve(
            y_test,
            y_probability[:, 1]
        )
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.4f}"
        )
    else:
        for i in range(number_of_classes):
            binary_y = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(
                binary_y,
                y_probability[:, i]
            )
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                label=f"{class_names[i]} (AUC = {roc_auc:.4f})"
            )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("XGBoost - ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "xgboost_roc_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
except Exception as error:
    print("\nROC curve could not be generated:", error)

# ================================================================
# 23. MODEL PARAMETERS
# ================================================================

parameters = pd.DataFrame({
    "Parameter": [
        "Algorithm",
        "Estimators",
        "Maximum Depth",
        "Learning Rate",
        "Subsample",
        "Column Sample By Tree",
        "Test Size",
        "Random State"
    ],
    "Value": [
        "XGBoost Classifier",
        200,
        5,
        0.05,
        0.8,
        0.8,
        0.20,
        42
    ]
})
parameters.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "xgboost_parameters.csv"
    ),
    index=False
)

# ================================================================
# 24. DATASET INFORMATION
# ================================================================

dataset_information = pd.DataFrame({
    "Information": [
        "Original Rows",
        "Original Columns",
        "Rows Used",
        "Number of Features",
        "Training Samples",
        "Testing Samples",
        "Target Column",
        "Number of Classes"
    ],
    "Value": [
        original_data.shape[0],
        original_data.shape[1],
        data.shape[0],
        X.shape[1],
        X_train.shape[0],
        X_test.shape[0],
        target_column,
        number_of_classes
    ]
})
dataset_information.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "dataset_information.csv"
    ),
    index=False
)

# ================================================================
# 25. SAVE XGBOOST MODEL
# ================================================================

model_file = os.path.join(
    OUTPUT_FOLDER,
    "xgboost_attrition_model.json"
)
xgb_model.save_model(model_file)

# ================================================================
# 26. FINAL OUTPUT LIST
# ================================================================

print("\n" + "=" * 70)
print("ALL XGBOOST OUTPUTS SAVED")
print("=" * 70)
print("\nOutput folder:")
print(OUTPUT_FOLDER)
print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" -", filename)
print("\n" + "=" * 70)
print("ORIGINAL / MAIN DATASET WAS NOT MODIFIED.")
print("=" * 70)
