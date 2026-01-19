
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
DATA_PATH = "/home/sanket/Make_My_route/battery-analytics-lab/data/standardized/CS2_35_9_30_10_Channel_1-008_standardized.parquet"
OUTPUT_DIR = Path("/home/sanket/Make_My_route/battery-analytics-lab/experiments/CS2_35_analysis")
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_FILE = OUTPUT_DIR / "analysis_report.md"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    return df

def perform_eda(df):
    print("Performing EDA...")
    
    # 1. Descriptive Statistics
    desc_stats = df.describe().to_markdown()
    
    # 2. Key Plots
    # Voltage vs Time
    plt.figure(figsize=(12, 6))
    plt.plot(df['Test_Time'], df['Voltage'], label='Voltage')
    plt.xlabel('Test Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage Profile Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "voltage_vs_time.png")
    plt.close()

    # Current vs Time
    plt.figure(figsize=(12, 6))
    plt.plot(df['Test_Time'], df['Current'], label='Current', color='orange')
    plt.xlabel('Test Time (s)')
    plt.ylabel('Current (A)')
    plt.title('Current Profile Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "current_vs_time.png")
    plt.close()
    
    return desc_stats

def feature_engineering(df):
    print("Feature Engineering...")
    # Cycle Level Aggregation
    # Filter for charge/discharge steps if we knew them, but for now we aggregate by Cycle_Index
    
    cycle_stats = df.groupby('Cycle_Index').agg({
        'Discharge_Capacity': 'max',
        'Charge_Capacity': 'max',
        'Voltage': ['mean', 'min', 'max'],
        'Current': 'mean',
        'Test_Time': 'max' # End time of cycle
    }).reset_index()
    
    # Flatten columns
    cycle_stats.columns = ['_'.join(col).strip('_') for col in cycle_stats.columns.values]
    
    # Calculate Coulombic Efficiency
    cycle_stats['Coulombic_Efficiency'] = cycle_stats['Discharge_Capacity_max'] / cycle_stats['Charge_Capacity_max']
    
    # Filter out potential Cycle 0 or outliers if needed (simple check)
    cycle_stats = cycle_stats[cycle_stats['Discharge_Capacity_max'] > 0]
    
    # Plot Capacity Fade
    plt.figure(figsize=(10, 6))
    plt.plot(cycle_stats['Cycle_Index'], cycle_stats['Discharge_Capacity_max'], marker='o')
    plt.xlabel('Cycle Number')
    plt.ylabel('Max Discharge Capacity (Ah)')
    plt.title('Discharge Capacity Fade Over Cycles')
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "capacity_fade.png")
    plt.close()
    
    return cycle_stats

def interpretability_analysis(cycle_df):
    print("Interpretability Analysis...")
    # Target: Discharge Capacity (predicting health/capacity)
    # Features: Voltage stats, Current stats, Cycle Index
    
    features = ['Cycle_Index', 'Voltage_mean', 'Voltage_min', 'Voltage_max', 'Current_mean']
    target = 'Discharge_Capacity_max'
    
    X = cycle_df[features]
    y = cycle_df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Predicting Discharge Capacity)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png")
    plt.close()
    
    return r2, mse, features, importances, indices

def generate_report(desc_stats, r2, mse, features, importances, indices):
    print("Generating Report...")
    report_content = f"""# Analysis Report: CS2_35 Battery Data

## 1. Descriptive Analysis via EDA
The dataset was loaded successfully. Below are the descriptive statistics:

{desc_stats}

### Visualizations
- **Voltage Profile**: See `plots/voltage_vs_time.png`
- **Current Profile**: See `plots/current_vs_time.png`

## 2. Feature Engineering & Quality
We aggregated data at the cycle level to observe trends.
- **Capacity Fade**: `plots/capacity_fade.png` shows the degradation of discharge capacity over cycles.

## 3. Interpretability Analysis
We trained a Random Forest Regressor to predict `Discharge_Capacity` based on aggregated cycle features.

**Model Performance:**
- R2 Score: {r2:.4f}
- MSE: {mse:.4e}

**Feature Importance:**
Top features driving the capacity prediction:
"""
    for i in range(len(features)):
        report_content += f"- **{features[indices[i]]}**: {importances[indices[i]]:.4f}\n"

    report_content += "\nSee `plots/feature_importance.png` for visual representation.\n"
    
    with open(REPORT_FILE, "w") as f:
        f.write(report_content)
    
    print(f"Report saved to {REPORT_FILE}")

def main():
    try:
        df = load_data(DATA_PATH)
        desc_stats = perform_eda(df)
        cycle_df = feature_engineering(df)
        r2, mse, features, importances, indices = interpretability_analysis(cycle_df)
        generate_report(desc_stats, r2, mse, features, importances, indices)
        print("Analysis Complete.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
