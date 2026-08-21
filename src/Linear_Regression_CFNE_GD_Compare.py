# ============================================================
# LINEAR REGRESSION
# Closed-Form Normal Equation vs Gradient Descent
#
# HR EMPLOYEE ATTRITION DATASET
# Images are stored in ONE separate folder
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = (
    "/Users/khvsaieswar/Desktop/"
    "HR_Employee_Attrition/dataset/"
    "final_preprocess.csv"
)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

data = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("LINEAR REGRESSION - HR EMPLOYEE ATTRITION")
print("CLOSED-FORM NORMAL EQUATION VS GRADIENT DESCENT")
print("=" * 60)

print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())

# ============================================================
# 2. DISPLAY COLUMNS
# ============================================================

print("\nDataset columns:")
for column in data.columns:
    print(column)

# ============================================================
# 3. CREATE IMAGE OUTPUT FOLDER
# ============================================================

IMAGE_FOLDER = (
    "/Users/khvsaieswar/Desktop/"
    "HR_Employee_Attrition/outputs/"
    "Linear_Regression_CFNE_GD_Compare"
)

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)

print("\nImage output folder:")
print(IMAGE_FOLDER)

# ============================================================
# 4. EXTRACT FEATURES AND TARGET
# ============================================================

# All columns except the last column = features
# Last column = target (Attrition)

X = data.iloc[:, :-1].copy()
y = data.iloc[:, -1].copy()

print("\nNumber of features:")
print(X.shape[1])

print("\nTarget column:")
print(data.columns[-1])

# ============================================================
# 5. CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPE CHECK")
print("=" * 60)
print(X.dtypes)

# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

print("\nMissing values in each feature:")
print(X.isna().sum())

print("\nTotal missing values in features:")
print(X.isna().sum().sum())

# Impute missing values if any
if X.isna().sum().sum() > 0:
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))

# ============================================================
# 8. FEATURE SCALING FOR GRADIENT DESCENT
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 9. CLOSED FORM SOLUTION (NORMAL EQUATION)
# ============================================================

# Add bias column
X_train_bias = np.c_[
    np.ones((X_train.shape[0], 1)),
    X_train.values
]

X_test_bias = np.c_[
    np.ones((X_test.shape[0], 1)),
    X_test.values
]

# Normal Equation: theta = (X^T X)^(-1) X^T y
theta = np.linalg.pinv(
    X_train_bias.T.dot(X_train_bias)
).dot(
    X_train_bias.T
).dot(
    y_train.values
)

# Prediction
pred_normal = X_test_bias.dot(theta)

# Metrics
mse_normal = mean_squared_error(y_test, pred_normal)
r2_normal = r2_score(y_test, pred_normal)

print("\n------ Closed Form Normal Equation ------")
print("Coefficients (including intercept):")
print(theta)
print("MSE      :", mse_normal)
print("R2 Score :", r2_normal)

# ============================================================
# 10. GRADIENT DESCENT
# ============================================================

X_train_gd = np.c_[
    np.ones((X_train_scaled.shape[0], 1)),
    X_train_scaled
]

X_test_gd = np.c_[
    np.ones((X_test_scaled.shape[0], 1)),
    X_test_scaled
]

m = len(y_train)
theta_gd = np.zeros(X_train_gd.shape[1])
learning_rate = 0.01
epochs = 1000

loss_history = []

for epoch in range(epochs):
    predictions = X_train_gd.dot(theta_gd)
    errors = predictions - y_train.values
    gradients = (2 / m) * X_train_gd.T.dot(errors)
    theta_gd -= learning_rate * gradients
    loss = np.mean(errors ** 2)
    loss_history.append(loss)

pred_gd = X_test_gd.dot(theta_gd)

mse_gd = mean_squared_error(y_test, pred_gd)
r2_gd = r2_score(y_test, pred_gd)

print("\n------ Gradient Descent ------")
print("Coefficients:")
print(theta_gd)
print("MSE      :", mse_gd)
print("R2 Score :", r2_gd)

# ============================================================
# 11. COMPARISON
# ============================================================

print("\n=========== Comparison ===========")
print("\nNormal Equation")
print("MSE =", mse_normal)
print("R2  =", r2_normal)
print("\nGradient Descent")
print("MSE =", mse_gd)
print("R2  =", r2_gd)

# ============================================================
# IMAGE 1: ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(figsize=(8, 6))
plt.scatter(
    y_test,
    pred_normal,
    alpha=0.5,
    label="Normal Equation"
)
plt.scatter(
    y_test,
    pred_gd,
    alpha=0.5,
    label="Gradient Descent"
)

minimum = min(y_test.min(), pred_normal.min(), pred_gd.min())
maximum = max(y_test.max(), pred_normal.max(), pred_gd.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values - HR Attrition")
plt.legend()
plt.grid(True)
plt.tight_layout()

image1 = os.path.join(IMAGE_FOLDER, "actual_vs_predicted.png")
plt.savefig(image1, dpi=300, bbox_inches="tight")
plt.close()

print("\nImage saved:", image1)

# ============================================================
# IMAGE 2: RESIDUAL COMPARISON
# ============================================================

normal_residuals = y_test.values - pred_normal
gd_residuals = y_test.values - pred_gd

plt.figure(figsize=(9, 6))
plt.scatter(
    pred_normal,
    normal_residuals,
    alpha=0.5,
    label="Normal Equation"
)
plt.scatter(
    pred_gd,
    gd_residuals,
    alpha=0.5,
    label="Gradient Descent"
)

plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

image2 = os.path.join(IMAGE_FOLDER, "residual_comparison.png")
plt.savefig(image2, dpi=300, bbox_inches="tight")
plt.close()

print("Image saved:", image2)

# ============================================================
# IMAGE 3: GRADIENT DESCENT LOSS CURVE
# ============================================================

plt.figure(figsize=(9, 6))
plt.plot(
    range(1, epochs + 1),
    loss_history
)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.tight_layout()

image3 = os.path.join(IMAGE_FOLDER, "gradient_descent_loss.png")
plt.savefig(image3, dpi=300, bbox_inches="tight")
plt.close()

print("Image saved:", image3)

# ============================================================
# SAVE IMAGE INFORMATION
# ============================================================

image_info = pd.DataFrame({
    "Image": [
        "actual_vs_predicted.png",
        "residual_comparison.png",
        "gradient_descent_loss.png"
    ],
    "Description": [
        "Actual values versus predictions from both methods",
        "Residual comparison between Normal Equation and Gradient Descent",
        "MSE loss across Gradient Descent epochs"
    ]
})

image_info.to_csv(
    os.path.join(IMAGE_FOLDER, "image_information.csv"),
    index=False
)

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("PROCESS COMPLETED SUCCESSFULLY")
print("==========================================")
print("\nAll images are stored in ONE folder:")
print(IMAGE_FOLDER)
print("\nGenerated images:")
print("1. actual_vs_predicted.png")
print("2. residual_comparison.png")
print("3. gradient_descent_loss.png")
print("\nOriginal dataset was NOT modified.")
