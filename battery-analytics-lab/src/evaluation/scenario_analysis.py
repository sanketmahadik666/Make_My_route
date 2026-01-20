
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tabulate import tabulate

# Configuration
DATA_PATH = "/home/sanket/Make_My_route/battery-analytics-lab/data/standardized/CS2_35_9_30_10_Channel_1-008_standardized.parquet"
OUTPUT_DIR = Path("/home/sanket/Make_My_route/battery-analytics-lab/experiments/CS2_35_analysis")
RESULTS_DIR = OUTPUT_DIR / "scenario_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NOMINAL_CAPACITY = 1.1  # Ah (LiCoO2)

def load_data(path):
    print(f"Loading data from {path}...")
    return pd.read_parquet(path)

def identify_test_scenario(df):
    print("Identifying Test Scenario...")
    
    # Identify Charge and Discharge steps/phases
    # Assuming positive current is Charge (based on common convention, but checking signs)
    # The Schema says Current is Amperes. Often Charge is +, Discharge is -.
    # Let's check mean Voltage vs Current correlation or just max/min.
    
    max_current = df['Current'].max()
    min_current = df['Current'].min()
    max_voltage = df['Voltage'].max()
    min_voltage = df['Voltage'].min()
    
    # Infer C-rates
    # Assuming standard convention: Charge > 0, Discharge < 0 OR vice-versa depending on logger.
    # Usually: Charge increases Voltage.
    
    # Quick check on current direction
    # If correlation(Current, Voltage) > 0, then +Current is Charge (likely).
    corr_iv = df['Current'].corr(df['Voltage'])
    
    # Heuristic for C-rates
    # Max abs current / Nominal
    
    charge_current_approx = max_current if max_current > 0 else 0
    discharge_current_approx = abs(min_current) if min_current < 0 else 0
    
    charge_crate = charge_current_approx / NOMINAL_CAPACITY
    discharge_crate = discharge_current_approx / NOMINAL_CAPACITY
    
    scenario_info = {
        "Chemistry": "LiCoO2 (Assumed based on CS2 series)",
        "Nominal Capacity (Ah)": NOMINAL_CAPACITY,
        "Max Voltage (V)": round(max_voltage, 4),
        "Min Voltage (V)": round(min_voltage, 4),
        "Max Current (A)": round(max_current, 4),
        "Min Current (A)": round(min_current, 4),
        "Inferred Charge C-rate": round(charge_crate, 2),
        "Inferred Discharge C-rate": round(discharge_crate, 2),
        "Correlation(I, V)": round(corr_iv, 4)
    }
    
    print("\nTest Scenario Identification:")
    print(tabulate(scenario_info.items(), headers=["Parameter", "Value"], tablefmt="grid"))
    
    return scenario_info

def analyze_correlations(df):
    print("\nAnalyzing Correlations...")
    
    # Aggregate by Cycle
    # Calculate Cycle Capacity as Max - Min (since Capacity is cumulative)
    cycle_stats = df.groupby('Cycle_Index').agg({
        'Discharge_Capacity': lambda x: x.max() - x.min(),
        'Charge_Capacity': lambda x: x.max() - x.min(),
        'Internal_Resistance': 'mean',
        'AC_Impedance': 'mean',
        'Step_Time': 'sum', 
        'Test_Time': lambda x: x.max() - x.min()
    }).rename(columns={'Test_Time': 'Cycle_Duration'})
    
    # Rename for clarity
    cycle_stats.columns = ['Cycle_Discharge_Cap', 'Cycle_Charge_Cap', 'Avg_Internal_Resistance', 'Avg_AC_Impedance', 'Sum_Step_Time', 'Cycle_Duration']
    cycle_stats = cycle_stats.reset_index()
    
    # Filter valid cycles (e.g. non-zero capacity)
    # Nominal is 1.1Ah. Check for unreasonable values (e.g. partial cycles)
    valid_cycles = cycle_stats[cycle_stats['Cycle_Discharge_Cap'] > 0.1].copy()
    
    if valid_cycles.empty:
        print("Warning: No valid cycles found for correlation analysis.")
        return
        
    # Correlation Matrix
    cols_to_corr = ['Cycle_Index', 'Cycle_Discharge_Cap', 'Avg_Internal_Resistance', 'Cycle_Duration']
    corr_matrix = valid_cycles[cols_to_corr].corr()
    
    print("\nCorrelation Matrix:")
    print(tabulate(corr_matrix, headers="keys", tablefmt="grid"))
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Battery Parameters')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_matrix.png")
    plt.close()
    
    # Scatter Plots for Key Correlations
    # 1. Cycle vs Capacity (Fade)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=valid_cycles, x='Cycle_Index', y='Cycle_Discharge_Cap')
    plt.title('Cycle vs Discharge Capacity (Per Cycle)')
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "cycle_vs_capacity.png")
    plt.close()
    
    # 2. Resistance vs Capacity
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=valid_cycles, x='Avg_Internal_Resistance', y='Cycle_Discharge_Cap')
    plt.title('Internal Resistance vs Discharge Capacity')
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "resistance_vs_capacity.png")
    plt.close()

    return corr_matrix

def main():
    df = load_data(DATA_PATH)
    scenario = identify_test_scenario(df)
    correlations = analyze_correlations(df)
    
    report_path = RESULTS_DIR / "scenario_report.txt"
    with open(report_path, "w") as f:
        f.write("CS2_35 Scenario and Correlation Report\n")
        f.write("========================================\n\n")
        f.write("Identified Scenario:\n")
        for k, v in scenario.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        if correlations is not None:
             f.write("Correlations:\n")
             f.write(correlations.to_string())

    print(f"\nAnalysis complete. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
