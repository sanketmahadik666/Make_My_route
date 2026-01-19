import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath('.'))

from src.access.data_loader import DataLoader

def main():
    print("DEMO: Loading Best Data from Final Index")
    print("-" * 40)
    
    # 1. Load the Approved Index
    index_path = "data/standardized/final_data_index.csv"
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found.")
        return

    approved_index = pd.read_csv(index_path)
    print(f"Index loaded. Found {len(approved_index)} approved files.")
    print("Top 3 files by quality score:")
    print(approved_index[['file_name', 'quality_score', 'status']].head(3))
    print("-" * 40)

    # 2. Load the best file
    loader = DataLoader()
    best_file_name = approved_index.iloc[0]['file_name']
    print(f"Loading best file: {best_file_name}...")
    
    df = loader.load_data(best_file_name)
    
    if not df.empty:
        print("Success!")
        print(f"Loaded {len(df)} rows.")
        print("Columns:", list(df.columns[:5]), "...")
        print("\nFirst 3 rows:")
        print(df.head(3)[['Test_Time', 'Voltage', 'Current', 'Charge_Capacity']])
    else:
        print("Error: Failed to load data.")

if __name__ == "__main__":
    main()
