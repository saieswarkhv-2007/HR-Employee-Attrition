import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "/Users/khvsaieswar/Desktop/HR_Employee_Attrition/dataset/HR_Employee_Attrition_raw.csv"
OUTPUT_DIR = "/Users/khvsaieswar/Desktop/HR_Employee_Attrition/outputs/Boxplots_correlation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("HR Employee Attrition Dataset Loaded Successfully.")
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# Numerical Columns
# ============================================================

numerical_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nNumerical Columns:")
print(numerical_cols)


# ============================================================
# Correlation Matrix
# ============================================================

corr_matrix = df[numerical_cols].corr()

print("\n--- Correlation Matrix ---")
print(corr_matrix)


# ============================================================
# Correlation Heatmap
# ============================================================

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of HR Employee Attrition Features",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nExported heatmap to: {heatmap_path}")


# ============================================================
# Target Column
# ============================================================

target_col = "Attrition"

print("\nTarget Column:", target_col)


# ============================================================
# Boxplots: Numerical Features vs Attrition
# ============================================================

if target_col in df.columns:

    print("\nGenerating Boxplots...")

    for col in numerical_cols:

        # Don't create Attrition vs Attrition
        if col == target_col:
            continue

        plt.figure(figsize=(7, 5))

        sns.boxplot(
            x=target_col,
            y=col,
            data=df
        )

        plt.title(
            f"{col} vs {target_col}",
            fontsize=12,
            fontweight="bold"
        )

        plt.xlabel(target_col)
        plt.ylabel(col)

        plt.tight_layout()

        boxplot_filename = (
            f"boxplot_{col}_vs_{target_col}.png"
        )

        boxplot_path = os.path.join(
            OUTPUT_DIR,
            boxplot_filename
        )

        plt.savefig(
            boxplot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Exported boxplot: {boxplot_filename}"
        )

else:

    print(
        f"\nTarget column '{target_col}' "
        "not found in dataset."
    )


# ============================================================
# Final Message
# ============================================================

print("\n" + "=" * 60)
print("All HR Attrition EDA tasks completed successfully!")
print("Output directory:")
print(OUTPUT_DIR)
print("=" * 60)