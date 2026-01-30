
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuration
INPUT_DIR = Path("/home/sanket/Make_My_route/battery-analytics-lab/experiments/CS2_35_analysis/resampled")
OUTPUT_DIR = Path("/home/sanket/Make_My_route/battery-analytics-lab/experiments/CS2_35_analysis/multivariate_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_resampled_data():
    # Find the parquet file in the input directory
    files = list(INPUT_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")
    # Return the most recent one if multiple
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    file_path = files[0]
    print(f"Loading resampled data from {file_path}")
    return pd.read_parquet(file_path)

def prepare_matrix(df, value_col='capacity_ah'):
    """
    Pivot data to shape (n_cycles, n_voltage_points)
    Rows = Cycles, Cols = Voltage Steps, Values = Capacity/Current
    """
    print(f"Preparing data matrix using {value_col}...")
    
    # Pivot
    matrix_df = df.pivot(index='cycle_number', columns='voltage_v', values=value_col)
    
    # Drop any cycles with NaNs (incomplete resampling)
    initial_shape = matrix_df.shape
    matrix_df = matrix_df.dropna()
    print(f"Matrix shape: {matrix_df.shape} (Dropped {initial_shape[0] - matrix_df.shape[0]} incomplete cycles)")
    
    return matrix_df

def perform_pca(matrix_df):
    print("Performing PCA...")
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(matrix_df)
    
    # PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_data)
    
    # Explained Variance
    variance = pca.explained_variance_ratio_
    print(f"Explained Variance: PC1={variance[0]:.2%}, PC2={variance[1]:.2%}, PC3={variance[2]:.2%}")
    
    # Plot PC1 vs PC2 (Trajectory)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=matrix_df.index, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Cycle Number')
    plt.xlabel(f'PC1 ({variance[0]:.1%} Variance)')
    plt.ylabel(f'PC2 ({variance[1]:.1%} Variance)')
    plt.title('PCA Trajectory of Battery Aging')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "pca_trajectory.png")
    plt.close()
    
    # Analyze Loadings/Components (Interpretation)
    voltage_points = matrix_df.columns
    plt.figure(figsize=(12, 6))
    plt.plot(voltage_points, pca.components_[0], label='PC1 Loading (Main Aging Mode)', color='blue')
    plt.plot(voltage_points, pca.components_[1], label='PC2 Loading (Secondary Mode)', color='red', linestyle='--')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Component Loading')
    plt.title('PCA Loadings: Which Voltage Regions Drive Variance?')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "pca_loadings.png")
    plt.close()
    
    return pca_result

def perform_clustering(matrix_df, pca_result, n_clusters=3):
    print("Performing Clustering...")
    
    # Clustering on PCA result is often more robust
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_result)
    
    # Plot Clusters
    plt.figure(figsize=(10, 8))
    # Use PC1 and PC2 for plotting
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
    
    # Annotate centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-Means Clustering: Aging Stages')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "aging_clusters.png")
    plt.close()
    
    # Save Cluster Assignments
    results_df = pd.DataFrame({
        'Cycle_Index': matrix_df.index,
        'Cluster': clusters
    })
    results_df.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)

def main():
    try:
        df = load_resampled_data()
        
        # We analyze Discharge Capacity curve shape
        # Note: In standardized file, Discharge_Capacity is cumulative. 
        # But resampler output for a single cycle usually contains the capacity values *at that voltage*.
        # We need to verify if it's cumulative or differential. 
        # Usually for analysis we want the capacity *profile*. 
        # Let's use 'capacity_ah' as resampled.
        
        matrix_df = prepare_matrix(df, value_col='capacity_ah')
        
        pca_result = perform_pca(matrix_df)
        perform_clustering(matrix_df, pca_result)
        
        print(f"Multivariate Analysis Complete. Results in {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Analysis Failed: {e}")
        # raise

if __name__ == "__main__":
    main()
