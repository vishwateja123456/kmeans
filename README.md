# KMEANS


TITLE: **Adaptive Cluster Refinement Using K-Means for Cyberattack Detection on the CIC-IDS2017 Dataset**

This script applies **K-Means clustering** to network flow data, commonly used in Intrusion Detection System (IDS) analysis (e.g., CICIDS datasets). K-Means is unsupervised, meaning it groups data based on intrinsic similarity without using the ground-truth labels.

To evaluate its effectiveness against known security categories (Benign/Attack), This technique finds the statistically optimal way to map the generated cluster IDs (e.g., Cluster 0, Cluster 1) to the known labels. This crucial step allows us to calculate robust classification metrics (Accuracy, F1-Score) to measure how well the discovered data patterns align with the actual security status.


# Technical Requirements and Setup

 1. Software Environment

    The script is developed in Python and requires the following libraries. It is highly recommended to use a virtual environment.

2. Data and Directory Structure

   The script is configured to automatically locate and process the data based on the paths defined in the Configuration section.

# Core Pipeline and Methodology

1. Data PreprocessingData preparation is essential for K-Means (which is highly sensitive to feature scale).

    Handling Non-Numerics/Infinities: Features are converted to numeric types. Any missing values {NaN}or non-finite numbers {Inf} are replaced using the median of the respective feature column.

    Scaling (Conditional): If SCALE = True, a StandardScaler is applied. This standardizes features to have a mean of 0 and a standard deviation of 1, preventing large-range features from dominating the distance            calculations.

    Label Conversion: The ground-truth label column is converted to a binary array: Benign ---> 0 and Attack -- >1.

2. K-Means ClusteringThe sklearn.cluster.KMeans implementation is used, running multiple times (N_INIT) with different initializations to increase the chance of finding the global optimum.

3. Optimal Assignment (Hungarian Algorithm)This is the most critical evaluation step. Since cluster IDs are arbitrary, the Hungarian Algorithm solves the linear sum assignment problem to find the single-best, one-to-one mapping between the two arbitrary cluster IDs (e.g., 0, 1) and the two true labels (Benign, Attack) that maximizes the overall Accuracy.

Configuration Deep Dive
Review and adjust these key parameters within the Python file before running:

|Parameter    |        Type      |       Details                                                                                                                                        |
|-----------  |  --------------  | ---------------------------------------------------------------------------------------------------------------------------------------------------
| MODE        |        String    |     "COMBINED": Concatenates all discovered CSVs for a single clustering analysis. "PER_FILE": Runs K-Means separately on each individual CSV file.  
| K           |       Integer    |    The number of clusters. Set to K=2 for this specific binary evaluation (Benign/Attack).                                                           |
| ALGORITHM   |       String     |     'elkan' is a memory-efficient and often faster variant. 'lloyd' is the classic, standard K-Means approach.                                       |
| SCALE       |    Boolean       |    Set to True for standardization, which is highly recommended for distance-based algorithms like K-Means.                                          |


# Results and Evaluation
All results are saved to the configured OUTDIR.
Metric Outputs (CSV Files)
A CSV file (e.g., metrics_combined.csv) is generated containing the complete results. The evaluation metrics demonstrate the quality of the clustering relative to the known intrusion types:

Accuracy: Overall percentage of correctly mapped instances.

F1-Score: The harmonic mean of Precision and Recall, balancing both metrics.

Precision, Recall: Standard classification metrics (useful for analyzing attack detection performance).

Silhouette Score: An intrinsic measure of cluster quality; values near +1 indicate dense, well-separated clusters, while values near 0 indicate overlapping clusters.

# Visual Reports 

| PlotType              |   File Example              |                            Insight Provided                                                                                                                    |   
| ---------------       | ------------------          | -------------------------------------------------------------------------------------------------------------------------------------------------------------
| Confusion Matrix      |  cm_combined.png            |      A heatmap detailing the agreement between the mapped cluster assignments and the ground-truth labels.                                                      
| Elbow Plot            |  elbow_combined.png         |   Plots the WCSS (Within-Cluster Sum of Squares) vs.K. The 'elbow' point suggests where adding more clusters yields diminishing returns.                       |
| Cluster Visualization |  clusters_pca_combined.png  |  Uses Principal Component Analysis (PCA) to reduce the high-dimensional data to two dimensions, visualizing the spatial separation of the discovered clusters. |



# Execution
Run the script from terminal:
python script_name.py




