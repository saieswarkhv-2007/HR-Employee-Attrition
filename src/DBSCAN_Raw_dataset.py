# ============================================================
# DBSCAN CLUSTERING ON RAW HR EMPLOYEE ATTRITION DATASET
# Features:
# 1. Age
# 2. TotalWorkingYears
# 3. MonthlyIncome
#
# Original Dataset is NOT Modified
# All Outputs and Performance Results -> ONE Folder
#
# Does NOT use:
# silhouette_score
# calinski_harabasz_score
# davies_bouldin_score
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# 1. INPUT AND OUTPUT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "HR_Employee_Attrition_raw.csv"
)

if not os.path.exists(INPUT_FILE):
    INPUT_FILE = os.path.join(
        BASE_DIR,
        "dataset",
        "final_preprocess.csv"
    )

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs",
    "DBSCAN_Raw_dataset_outputs"
)

# Create only the output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD RAW DATASET
# ============================================================

print("=" * 75)
print("DBSCAN CLUSTERING - RAW HR EMPLOYEE ATTRITION DATASET")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    print("\nERROR: Input file not found!")
    print("Check the following path:")
    print(INPUT_FILE)
    raise SystemExit

# Read only - original dataset will not be changed
data = pd.read_csv(INPUT_FILE)
print("\nOriginal Raw Dataset Shape:")
print(data.shape)
print("\nAvailable Columns:")
print(list(data.columns))

# ============================================================
# 3. SELECT REQUIRED ATTRIBUTES
# ============================================================

preferred_features = [
    "Age",
    "TotalWorkingYears",
    "MonthlyIncome"
]

available_features = [col for col in preferred_features if col in data.columns]

if len(available_features) >= 3:
    FEATURES = available_features[:3]
else:
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) >= 3:
        FEATURES = numeric_columns[:3]
    else:
        FEATURES = numeric_columns

print("\nSelected Features for DBSCAN:", FEATURES)

missing_columns = [
    column for column in FEATURES
    if column not in data.columns
]

if missing_columns:
    print("\nERROR: The following required columns are missing:")
    print(missing_columns)
    print("\nPlease check the column names in the dataset.")
    raise SystemExit

# ============================================================
# 4. CREATE A SEPARATE COPY
# ============================================================

# IMPORTANT:
# This creates a completely separate DataFrame.
# The original dataset remains untouched.
dbscan_data = data[FEATURES].copy()

# ============================================================
# 5. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in FEATURES:
    dbscan_data[column] = pd.to_numeric(
        dbscan_data[column],
        errors="coerce"
    )

# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(dbscan_data.isnull().sum())

# ============================================================
# 7. REMOVE INVALID ROWS FROM THE COPY ONLY
# ============================================================

before_cleaning = len(dbscan_data)
dbscan_data = dbscan_data.dropna().reset_index(drop=True)
after_cleaning = len(dbscan_data)

removed_rows = before_cleaning - after_cleaning
print("\nRows before cleaning:", before_cleaning)
print("Rows after cleaning :", after_cleaning)
print("Rows removed        :", removed_rows)

# ============================================================
# 8. SAVE SELECTED FEATURES
# ============================================================

selected_features_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_selected_features.csv"
)
dbscan_data.to_csv(
    selected_features_file,
    index=False
)

# ============================================================
# 9. STANDARDIZATION
# ============================================================

# DBSCAN is distance-based.
# Standardization prevents one feature from dominating the distance calculation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    dbscan_data[FEATURES]
)

# ============================================================
# 10. SAVE STANDARDIZED DATA
# ============================================================

standardized_data = pd.DataFrame(
    X_scaled,
    columns=FEATURES
)
standardized_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_standardized_features.csv"
)
standardized_data.to_csv(
    standardized_file,
    index=False
)

# ============================================================
# 11. DBSCAN PARAMETERS
# ============================================================

EPS = 0.80
MIN_SAMPLES = 10

print("\nDBSCAN Parameters")
print("-" * 40)
print("eps         :", EPS)
print("min_samples :", MIN_SAMPLES)

# ============================================================
# 12. APPLY DBSCAN
# ============================================================

dbscan_model = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES
)

cluster_labels = dbscan_model.fit_predict(
    X_scaled
)

# ============================================================
# 13. CREATE CLUSTERED DATASET
# ============================================================

clustered_data = dbscan_data.copy()
clustered_data["DBSCAN_Cluster"] = cluster_labels

# ============================================================
# 14. CLUSTER INFORMATION
# ============================================================

unique_labels = sorted(
    np.unique(cluster_labels)
)
actual_clusters = [
    label
    for label in unique_labels
    if label != -1
]
number_of_clusters = len(actual_clusters)
noise_count = int(
    np.sum(cluster_labels == -1)
)
clustered_count = len(cluster_labels) - noise_count
total_records = len(cluster_labels)

if total_records > 0:
    noise_percentage = (
        noise_count / total_records
    ) * 100
    clustered_percentage = (
        clustered_count / total_records
    ) * 100
else:
    noise_percentage = 0
    clustered_percentage = 0

print("\n" + "=" * 75)
print("DBSCAN RESULTS")
print("=" * 75)
print("\nTotal Records       :", total_records)
print("Number of Clusters  :", number_of_clusters)
print("Noise Points        :", noise_count)
print("Clustered Points    :", clustered_count)
print("Noise Percentage    :", round(noise_percentage, 2), "%")
print("Clustered Percentage:", round(clustered_percentage, 2), "%")

# ============================================================
# 15. SAVE CLUSTERED DATA
# ============================================================

clustered_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_clustered_data.csv"
)
clustered_data.to_csv(
    clustered_file,
    index=False
)

# ============================================================
# 16. CLUSTER DISTRIBUTION
# ============================================================

cluster_counts = (
    clustered_data["DBSCAN_Cluster"]
    .value_counts()
    .sort_index()
)
print("\nCluster Distribution:")
print(cluster_counts)

cluster_distribution = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Number_of_Records": cluster_counts.values
})

cluster_distribution_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_cluster_distribution.csv"
)
cluster_distribution.to_csv(
    cluster_distribution_file,
    index=False
)

# ============================================================
# 17. CLUSTER SUMMARY
# ============================================================

summary_records = []
for cluster_id in unique_labels:
    cluster_subset = clustered_data[
        clustered_data["DBSCAN_Cluster"] == cluster_id
    ]
    if cluster_id == -1:
        cluster_name = "Noise"
    else:
        cluster_name = "Cluster_" + str(cluster_id)
    
    rec = {
        "Cluster": cluster_name,
        "Cluster_ID": cluster_id,
        "Number_of_Records": len(cluster_subset)
    }
    for feat in FEATURES:
        rec[f"Average_{feat}"] = round(cluster_subset[feat].mean(), 3)
    
    summary_records.append(rec)

cluster_summary = pd.DataFrame(
    summary_records
)

cluster_summary_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_cluster_summary.csv"
)
cluster_summary.to_csv(
    cluster_summary_file,
    index=False
)

# ============================================================
# 18. PERFORMANCE RESULTS
# ============================================================

performance_results = pd.DataFrame({
    "Metric": [
        "Algorithm",
        "Input Dataset",
        "Original Dataset Rows",
        "Rows Used",
        "Rows Removed",
        "Number of Features",
        "Features Used",
        "Scaling Method",
        "DBSCAN eps",
        "DBSCAN min_samples",
        "Number of Clusters",
        "Noise Points",
        "Clustered Points",
        "Noise Percentage",
        "Clustered Percentage"
    ],
    "Value": [
        "DBSCAN",
        os.path.basename(INPUT_FILE),
        before_cleaning,
        after_cleaning,
        removed_rows,
        len(FEATURES),
        ", ".join(FEATURES),
        "StandardScaler",
        EPS,
        MIN_SAMPLES,
        number_of_clusters,
        noise_count,
        clustered_count,
        round(noise_percentage, 2),
        round(clustered_percentage, 2)
    ]
})

performance_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_performance_results.csv"
)
performance_results.to_csv(
    performance_file,
    index=False
)

# ============================================================
# 19. PCA FOR 2-D VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2
)
X_pca = pca.fit_transform(
    X_scaled
)

pca_data = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "DBSCAN_Cluster": cluster_labels
})

pca_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_PCA_coordinates.csv"
)
pca_data.to_csv(
    pca_file,
    index=False
)

# ============================================================
# 20. GRAPH 1 - FEATURE 1 VS FEATURE 2
# ============================================================

if len(FEATURES) >= 2:
    plt.figure(figsize=(10, 7))
    plt.scatter(
        dbscan_data[FEATURES[0]],
        dbscan_data[FEATURES[1]],
        c=cluster_labels,
        s=25,
        alpha=0.7
    )
    plt.xlabel(FEATURES[0])
    plt.ylabel(FEATURES[1])
    plt.title(f"DBSCAN Clustering: {FEATURES[0]} vs {FEATURES[1]}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph1 = os.path.join(
        OUTPUT_FOLDER,
        f"01_DBSCAN_{FEATURES[0]}_vs_{FEATURES[1]}.png"
    )
    plt.savefig(
        graph1,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ============================================================
# 21. GRAPH 2 - FEATURE 1 VS FEATURE 3
# ============================================================

if len(FEATURES) >= 3:
    plt.figure(figsize=(10, 7))
    plt.scatter(
        dbscan_data[FEATURES[0]],
        dbscan_data[FEATURES[2]],
        c=cluster_labels,
        s=25,
        alpha=0.7
    )
    plt.xlabel(FEATURES[0])
    plt.ylabel(FEATURES[2])
    plt.title(f"DBSCAN Clustering: {FEATURES[0]} vs {FEATURES[2]}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph2 = os.path.join(
        OUTPUT_FOLDER,
        f"02_DBSCAN_{FEATURES[0]}_vs_{FEATURES[2]}.png"
    )
    plt.savefig(
        graph2,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ============================================================
# 22. GRAPH 3 - FEATURE 2 VS FEATURE 3
# ============================================================

if len(FEATURES) >= 3:
    plt.figure(figsize=(10, 7))
    plt.scatter(
        dbscan_data[FEATURES[1]],
        dbscan_data[FEATURES[2]],
        c=cluster_labels,
        s=25,
        alpha=0.7
    )
    plt.xlabel(FEATURES[1])
    plt.ylabel(FEATURES[2])
    plt.title(f"DBSCAN Clustering: {FEATURES[1]} vs {FEATURES[2]}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph3 = os.path.join(
        OUTPUT_FOLDER,
        f"03_DBSCAN_{FEATURES[1]}_vs_{FEATURES[2]}.png"
    )
    plt.savefig(
        graph3,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ============================================================
# 23. GRAPH 4 - PCA CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    s=25,
    alpha=0.7
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Cluster Visualization using PCA")
plt.grid(True, alpha=0.3)
plt.tight_layout()

graph4 = os.path.join(
    OUTPUT_FOLDER,
    "04_DBSCAN_PCA_Clusters.png"
)
plt.savefig(
    graph4,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# 24. GRAPH 5 - CLUSTER DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))
plt.bar(
    cluster_distribution["Cluster"].astype(str),
    cluster_distribution["Number_of_Records"]
)
plt.xlabel("DBSCAN Cluster")
plt.ylabel("Number of Records")
plt.title("DBSCAN Cluster Distribution")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

graph5 = os.path.join(
    OUTPUT_FOLDER,
    "05_DBSCAN_Cluster_Distribution.png"
)
plt.savefig(
    graph5,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# 25. SAVE EXECUTION INFORMATION
# ============================================================

execution_information = pd.DataFrame({
    "Item": [
        "Input Dataset",
        "Output Folder",
        "Original Dataset Modified",
        "Features Used",
        "Preprocessing",
        "Scaling",
        "Algorithm",
        "eps",
        "min_samples"
    ],
    "Details": [
        INPUT_FILE,
        OUTPUT_FOLDER,
        "NO",
        ", ".join(FEATURES),
        "Raw dataset with numeric extraction & cleaning",
        "StandardScaler",
        "DBSCAN",
        EPS,
        MIN_SAMPLES
    ]
})

execution_file = os.path.join(
    OUTPUT_FOLDER,
    "execution_information.csv"
)
execution_information.to_csv(
    execution_file,
    index=False
)

# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 75)
print("\nOutput Folder:")
print(OUTPUT_FOLDER)
print("\nFiles Generated:")
for filename in sorted(
    os.listdir(OUTPUT_FOLDER)
):
    print(" ", filename)

print("\n" + "=" * 75)
print("DBSCAN ANALYSIS COMPLETED")
print("Original dataset was NOT modified.")
print("=" * 75)
