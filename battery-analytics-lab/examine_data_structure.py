"""
Battery Analytics Lab - Data Structure Examination
Examines the actual structure of CS2_35 Excel files to understand available columns

Author: Battery Analytics Lab Team
Date: 2025-12-29
"""

import os
import pandas as pd
from pathlib import Path

def examine_excel_structure(file_path):
    """Examine the structure of an Excel file."""
    print(f"\n{'='*60}")
    print(f"EXAMINING: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Read Excel file and examine sheets
        xl_file = pd.ExcelFile(file_path)
        print(f"Sheet names: {xl_file.sheet_names}")
        
        # Examine each sheet
        for sheet_name in xl_file.sheet_names:
            print(f"\n--- Sheet: {sheet_name} ---")
            
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
                print(f"Shape: {df.shape}")
                print(f"Columns ({len(df.columns)}):")
                for i, col in enumerate(df.columns, 1):
                    print(f"  {i:2d}. {col}")
                
                # Show first few rows
                print(f"\nFirst 3 rows:")
                print(df.head(3).to_string())
                
            except Exception as e:
                print(f"Error reading sheet {sheet_name}: {e}")
    
    except Exception as e:
        print(f"Error examining file: {e}")

def main():
    """Main function to examine all CS2_35 files."""
    data_dir = Path("/home/sanket/Make_My_route/DATA/CS2_35")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Get all Excel files
    excel_files = list(data_dir.glob("*.xlsx"))
    excel_files.sort()
    
    print(f"Found {len(excel_files)} Excel files")
    
    # Examine first few files in detail
    for i, file_path in enumerate(excel_files[:3]):
        examine_excel_structure(file_path)
        
        if i < len(excel_files) - 1:
            print(f"\n{'#'*60}")
    
    # Quick overview of all files
    print(f"\n{'='*60}")
    print("QUICK OVERVIEW - ALL FILES")
    print(f"{'='*60}")
    
    for file_path in excel_files:
        try:
            xl_file = pd.ExcelFile(file_path)
            print(f"{file_path.name}: {xl_file.sheet_names}")
        except Exception as e:
            print(f"{file_path.name}: ERROR - {e}")

if __name__ == "__main__":
    main()