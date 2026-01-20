
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Configuration
DATA_PATH = "/home/sanket/Make_My_route/battery-analytics-lab/data/standardized/CS2_35_9_30_10_Channel_1-008_standardized.parquet"
OUTPUT_DIR = Path("/home/sanket/Make_My_route/battery-analytics-lab/experiments/CS2_35_analysis/advanced_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path):
    print(f"Loading data from {path}...")
    return pd.read_parquet(path)

def calculate_dqdv(df):
    print("Calculating dQ/dV...")
    
    # Select a few representative cycles (Start, Middle, End)
    cycles_to_plot = [1, 10, 25, 40, 50]
    
    plt.figure(figsize=(12, 8))
    
    for cycle in cycles_to_plot:
        # Filter for Discharge stats
        # Assuming Current < 0 for discharge based on previous analysis findings
        cycle_data = df[(df['Cycle_Index'] == cycle) & (df['Current'] < -0.05)].copy()
        
        if cycle_data.empty:
            continue
            
        # Sort by voltage (descending for discharge usually, but interp needs increasing x)
        cycle_data = cycle_data.sort_values('Voltage')
        
        # Data preparation
        v = cycle_data['Voltage'].values
        # Discharge Capacity is cumulative in the file. 
        # But for dQ/dV we need the capacity *within* this step relative to the start.
        # So we subtract min.
        q = cycle_data['Discharge_Capacity'].values
        q = q - q.min()
        
        # Remove duplicates
        _, unique_indices = np.unique(v, return_index=True)
        v = v[unique_indices]
        q = q[unique_indices]
        
        if len(v) < 10:
            continue
            
        # Interpolate to uniform grid
        v_interp = np.linspace(v.min(), v.max(), 500)
        f_q = interp1d(v, q, kind='linear', fill_value="extrapolate")
        q_interp = f_q(v_interp)
        
        # Calculate dQ/dV
        dq = np.gradient(q_interp)
        dv = np.gradient(v_interp)
        dqdv = dq / dv
        
        # Smooth
        try:
            dqdv_smooth = savgol_filter(dqdv, window_length=21, polyorder=2)
        except:
            dqdv_smooth = dqdv
            
        plt.plot(v_interp, dqdv_smooth, label=f'Cycle {cycle}')
    
    plt.xlabel('Voltage (V)')
    plt.ylabel('dQ/dV (Ah/V)')
    plt.title('Incremental Capacity Analysis (dQ/dV) Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "dqdv_cycles.png")
    plt.close()

def analyze_impedance(df):
    print("Analyzing Impedance Evolution...")
    
    cycle_stats = df.groupby('Cycle_Index').agg({
        'AC_Impedance': 'mean',
        'ACI_Phase_Angle': 'mean',
        'Internal_Resistance': 'mean'
    }).reset_index()
    
    # 1. AC Impedance Magnitude
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=cycle_stats, x='Cycle_Index', y='AC_Impedance', marker='o')
    plt.title('AC Impedance Magnitude vs Cycle')
    plt.ylabel('Impedance (Ohms)')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "impedance_evolution.png")
    plt.close()
    
    # 2. Phase Angle
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=cycle_stats, x='Cycle_Index', y='ACI_Phase_Angle', marker='o', color='green')
    plt.title('AC Impedance Phase Angle vs Cycle')
    plt.ylabel('Phase Angle (Degrees)')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "phase_angle_evolution.png")
    plt.close()
    
    # 3. DC Internal Resistance
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=cycle_stats, x='Cycle_Index', y='Internal_Resistance', marker='o', color='red')
    plt.title('DC Internal Resistance vs Cycle')
    plt.ylabel('Resistance (Ohms)')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "dc_resistance_evolution.png")
    plt.close()

def main():
    try:
        df = load_data(DATA_PATH)
        calculate_dqdv(df)
        analyze_impedance(df)
        print(f"Advanced Analysis Complete. Results in {OUTPUT_DIR}")
    except Exception as e:
        print(f"Analysis Failed: {e}")
        # raise

if __name__ == "__main__":
    main()
