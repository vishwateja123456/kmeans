import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, silhouette_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA # Added for visualization
import seaborn as sns # Added for enhanced plots



# Configuration


DATA_DIR = r"C:\MS\FALL2025\IDS\Kmeans\data"  # Folder containing your CSVs
FILE_PATHS = []

MODE = "COMBINED"              # "COMBINED" or "PER_FILE"
K = 2                           # Number of clusters
SCALE = True                    # Apply StandardScaler
ALGORITHM = "elkan"             # "elkan" (Euclidean) or "lloyd"
N_INIT = 10                     # Number of centroid initializations
MAX_ITER = 300                  # Maximum iterations
RANDOM_SEED = 42
TARGET_COL_FALLBACK = " Label"

OUTDIR = r"C:\MS\FALL2025\IDS\Kmeans\results_kmeans_pure"
os.makedirs(OUTDIR, exist_ok=True)
print("Results will be saved to:", OUTDIR)



# Helper Functions

def list_all_csvs_recursive(base):
    """Recursively find all CSV files in a directory."""
    csvs = []
    if base and os.path.isdir(base):
        # Using os.walk for Strong, recursive file search
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".csv"):
                    csvs.append(os.path.join(root, f))
    return csvs


def detect_label_col(df):
    """Detect the label column automatically."""
    if TARGET_COL_FALLBACK in df.columns:
        return TARGET_COL_FALLBACK
    for c in df.columns:
        if "label" in c.lower():
            return c
    raise KeyError("No label column found in dataset.")


def labels_to_binary(series):
    """Convert labels to binary (0 = Benign, 1 = Attack)."""
    return series.apply(lambda x: 0 if "BENIGN" in str(x).upper() else 1).values


def to_numeric_features(df, drop_cols):
    """Keep numeric feature columns and handle NaN/Inf."""
    Xdf = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    return Xdf


def preprocess(Xdf, scale=True):
    """Impute missing values and optionally apply scaling."""
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(Xdf)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X.astype("float32"), imputer, scaler


def hungarian_map(y_true, y_pred):
    """Map cluster labels to true labels using Hungarian matching."""
    cm = confusion_matrix(y_true, y_pred)
    max_val = cm.max() if cm.size else 0
    cost = max_val - cm
    # Use linear_sum_assignment to find the optimal mapping
    r, c = linear_sum_assignment(cost)
    mapping = {cl: gt for gt, cl in zip(r, c)}
    y_mapped = np.array([mapping[cl] for cl in y_pred])
    return y_mapped, mapping


def run_kmeans_pure(X):
    """Run standard K-Means clustering."""
    km = KMeans(
        n_clusters=K,
        algorithm=ALGORITHM,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED
    )
    y_pred = km.fit_predict(X)
    return y_pred, km


def evaluate_with_mapping(X, y_true, y_pred):
    """Evaluate clustering using accuracy, precision, recall, F1, and silhouette."""
    y_map, mapping = hungarian_map(y_true, y_pred)
    acc = accuracy_score(y_true, y_map)
    pre = precision_score(y_true, y_map, zero_division=0)
    rec = recall_score(y_true, y_map, zero_division=0)
    f1 = f1_score(y_true, y_map, zero_division=0)

    if len(np.unique(y_pred)) > 1:
        n = len(y_pred)
        # Sample for fast silhouette score on large datasets
        if n > 20000:
            idx = np.random.RandomState(RANDOM_SEED).choice(n, size=20000, replace=False)
            sil = silhouette_score(X[idx], y_pred[idx])
        else:
            sil = silhouette_score(X, y_pred)
    else:
        sil = 0.0
    return acc, pre, rec, f1, sil, y_map


def plot_confusion(y_true, y_map, savepath):
    """Generate and save a confusion matrix heatmap using seaborn."""
    try:
        plt.figure(figsize=(7, 5))
        cm = confusion_matrix(y_true, y_map)
        df_cm = pd.DataFrame(
            cm,
            index=['True Benign (0)', 'True Attack (1)'],
            columns=['Prediction Benign (0)', 'Prediction Attack (1)']
        )
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title("Confusion Matrix (Hungarian Mapped)")
        plt.tight_layout()
        plt.savefig(savepath, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Confusion plot skipped: {e}")


def plot_elbow(X, savepath, max_k=10):
    """Generate and save the K-Means Elbow plot (WCSS vs. K)."""
    wcss = []
    # Determine the maximum K to check
    max_k = min(max_k, len(np.unique(X, axis=0)) - 1 if len(np.unique(X, axis=0)) > 1 else 1)
    if max_k < 2:
        print("Elbow plot skipped: Not enough unique samples.")
        return

    K_range = range(1, max_k + 1)
    
    # Use a sample for very large datasets
    n = len(X)
    X_sample = X
    if n > 50000:
        idx = np.random.RandomState(RANDOM_SEED).choice(n, size=50000, replace=False)
        X_sample = X[idx]

    print(f"Calculating WCSS for K=1 to {max_k}...")
    for k in K_range:
        km = KMeans(
            n_clusters=k,
            algorithm=ALGORITHM,
            n_init=N_INIT,
            max_iter=MAX_ITER,
            random_state=RANDOM_SEED
        )
        km.fit(X_sample)
        wcss.append(km.inertia_) # inertia is the WCSS

    plt.figure(figsize=(8, 6))
    plt.plot(K_range, wcss, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(K_range)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()
    print("Elbow plot saved.")


def plot_clusters(X, y_pred, savepath, name="K-Means Clusters"):
    """
    Reduce dimensionality with PCA to 2D and plot the predicted clusters.
    """
    try:
        # Use a sample for visualization simplicity
        n = len(X)
        X_plot = X
        y_plot = y_pred
        if n > 10000:
            idx = np.random.RandomState(RANDOM_SEED).choice(n, size=10000, replace=False)
            X_plot = X[idx]
            y_plot = y_pred[idx]

        # Apply PCA
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_plot)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_pca[:, 0], 
            X_pca[:, 1], 
            c=y_plot, 
            cmap='viridis', 
            marker='.', 
            alpha=0.6,
            s=15
        )
        
        # Add legend
        legend1 = plt.legend(*scatter.legend_elements(), 
                            loc="lower left", 
                            title="Clusters", 
                            fontsize='small')
        plt.gca().add_artist(legend1)

        plt.title(f'{name} - PCA-reduced Cluster Visualization')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2f} explained var.)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2f} explained var.)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(savepath, dpi=200)
        plt.close()
        print("Cluster plot (PCA) saved.")
    except Exception as e:
        print(f"Cluster plot (PCA) skipped: {e}")



# Load and Validate CSV Paths (Refactored)


def discover_csv_files_recursive(data_dir, file_paths_list):
    """Automatically discover CSV files under common directories."""
    search_roots = []
    
    if data_dir and os.path.isdir(data_dir):
        search_roots.append(data_dir)

    Kmeans_root = r"C:\MS\FALL2025\IDS\Kmeans"
    if os.path.isdir(Kmeans_root):
        search_roots.append(Kmeans_root)

    cwd = os.getcwd()
    if os.path.isdir(cwd):
        search_roots.append(cwd)

    found_csvs = []
    for root in search_roots:
        found_csvs.extend(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))

    # Remove duplicates and combine with existing paths
    unique_csvs = list(dict.fromkeys(found_csvs + file_paths_list))
    return unique_csvs


if not FILE_PATHS:
    FILE_PATHS = discover_csv_files_recursive(DATA_DIR, FILE_PATHS)

print("\nChecking CSV paths:")
valid_paths = []
for path in FILE_PATHS:
    if os.path.exists(path):
        print("Found:", path)
        valid_paths.append(path)

# Update the main list to only include valid paths
FILE_PATHS = valid_paths

if not valid_paths:
    raise RuntimeError("No valid CSV files found. Update DATA_DIR or FILE_PATHS manually.")



# Main Processing Logic


if MODE.upper() == "COMBINED":
    dataframes = []
    for path in valid_paths:
        print("Loading:", os.path.basename(path))
        df_part = pd.read_csv(path)
        dataframes.append(df_part)

    
    df = pd.concat(dataframes, ignore_index=True, sort=False, join='inner') # Line 268
    print("Combined dataset shape:", df.shape)

    label_col = detect_label_col(df)
    y_true = labels_to_binary(df[label_col])
    Xdf = to_numeric_features(df, drop_cols=['Flow ID', ' Timestamp', label_col])

    X, _, _ = preprocess(Xdf, scale=SCALE)
    y_pred, km = run_kmeans_pure(X)

    acc, pre, rec, f1, sil, y_map = evaluate_with_mapping(X, y_true, y_pred)

    print("\n[Combined Results]")
    print(f"K={K}, Algorithm={ALGORITHM}, Scale={SCALE}")
    print(f"Accuracy={acc:.4f}, Precision={pre:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Silhouette={sil:.4f}")

    results_df = pd.DataFrame([{
        "mode": "COMBINED",
        "k": K,
        "algorithm": ALGORITHM,
        "scaled": SCALE,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "silhouette": sil
    }])
    results_df.to_csv(os.path.join(OUTDIR, "metrics_combined.csv"), index=False)

    # Generate Plots
    plot_confusion(y_true, y_map, os.path.join(OUTDIR, "cm_combined.png"))
    plot_elbow(X, os.path.join(OUTDIR, "elbow_combined.png"))
    plot_clusters(X, y_pred, os.path.join(OUTDIR, "clusters_pca_combined.png"))
    
    print("Results saved in:", OUTDIR)

else:
    all_metrics = []
    first_file_processed = False
    
    for path in valid_paths:
        name = os.path.basename(path)
        print("\nProcessing file:", name)
        df = pd.read_csv(path)

        label_col = detect_label_col(df)
        y_true = labels_to_binary(df[label_col])
        Xdf = to_numeric_features(df, drop_cols=['Flow ID', ' Timestamp', label_col])

        X, _, _ = preprocess(Xdf, scale=SCALE)
        y_pred, km = run_kmeans_pure(X)

        acc, pre, rec, f1, sil, y_map = evaluate_with_mapping(X, y_true, y_pred)
        print(f"Accuracy={acc:.4f}, Precision={pre:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Silhouette={sil:.4f}")

        all_metrics.append({
            "dataset": name,
            "k": K,
            "algorithm": ALGORITHM,
            "scaled": SCALE,
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1,
            "silhouette": sil
        })
        
        # Generate Plots
        plot_confusion(y_true, y_map, os.path.join(OUTDIR, f"cm_{name.replace('.csv', '')}.png"))
        plot_clusters(
            X, 
            y_pred, 
            os.path.join(OUTDIR, f"clusters_pca_{name.replace('.csv', '')}.png"),
            name=f"Clusters for {name.replace('.csv', '')}"
        )
        
        # Plot Elbow curve only once if not already done
        if not first_file_processed:
             plot_elbow(X, os.path.join(OUTDIR, "elbow_first_file.png")) 
             first_file_processed = True


    pd.DataFrame(all_metrics).to_csv(os.path.join(OUTDIR, "metrics_per_file.csv"), index=False)
    print("\nPer-file results saved in:", OUTDIR)