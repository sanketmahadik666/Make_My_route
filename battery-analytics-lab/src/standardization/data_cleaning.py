"""
Battery Analytics Lab - Data Cleaning Module
Step 4.2: Handling Missing Data and Outliers

This module handles the identification and treatment of voltage outliers 
and missing data gaps in battery datasets.

Requirements:
- Threshold Filtering: 2.0V < V < 4.5V for LCO cells
- Gap Filling: 
    - < 10 seconds: Linear interpolation
    - >= 10 seconds: Flag as "corrupted"
- Output: Cleaned dataset with quality flags

Author: Battery Analytics Lab Team
Date: 2026-01-22
Version: 1.0
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    Data cleaning module for battery analytics.
    Handles outlier removal and missing data interpolation.
    """
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.logger = self._setup_logging()
        self.cleaning_stats = {
            'outliers_removed': 0,
            'gaps_interpolated': 0,
            'large_gaps_flagged': 0,
            'processing_errors': []
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for data cleaning process."""
        logger = logging.getLogger('data_cleaner')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('battery-analytics-lab/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / 'data_cleaning.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def remove_voltage_outliers(self, df: pd.DataFrame, voltage_col: str = None) -> pd.DataFrame:
        """
        Apply threshold filtering to remove voltage spikes.
        Pass-filter: 2.0V < V < 4.5V
        
        Args:
            df: Input DataFrame
            voltage_col: Name of the voltage column. If None, tries to auto-detect.
            
        Returns:
            DataFrame with outliers removed
        """
        if df.empty:
            return df
            
        if not voltage_col:
            voltage_candidates = [c for c in df.columns if 'voltage' in c.lower() or 'volt' in c.lower()]
            if not voltage_candidates:
                self.logger.error("Could not find voltage column for outlier removal")
                return df
            voltage_col = voltage_candidates[0]
            
        initial_count = len(df)
        # Apply threshold filtering: 2.0V < V < 4.5V
        mask = (df[voltage_col] > 2.0) & (df[voltage_col] < 4.5)
        df_cleaned = df[mask].copy()
        
        removed = initial_count - len(df_cleaned)
        if removed > 0:
            self.logger.warning(f"Removed {removed} voltage outlier points from {voltage_col}")
            self.cleaning_stats['outliers_removed'] += removed
            
        return df_cleaned

    def handle_missing_data_gaps(self, df: pd.DataFrame, time_col: str = None, gap_threshold: float = 10.0) -> pd.DataFrame:
        """
        Handle missing data gaps by interpolation or flagging.
        
        Args:
            df: Input DataFrame
            time_col: Name of the time/timestamp column. If None, tries to auto-detect.
            gap_threshold: Threshold in seconds for flagging "corrupted" gaps. Default 10.0s.
            
        Returns:
            DataFrame with gaps handled
        """
        if df.empty:
            return df
            
        if not time_col:
            time_candidates = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
            if not time_candidates:
                self.logger.error("Could not find time column for gap handling")
                return df
            time_col = time_candidates[0]
            
        # Ensure sorted by time
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # Calculate time differences
        dt = df[time_col].diff()
        
        # Initialize quality flag if not present
        if 'data_quality_flag' not in df.columns:
            df['data_quality_flag'] = 'good'
            
        # Identify large gaps
        large_gaps_mask = dt >= gap_threshold
        if large_gaps_mask.any():
            num_large = large_gaps_mask.sum()
            self.logger.warning(f"Found {num_large} large time gaps (>= {gap_threshold}s)")
            df.loc[large_gaps_mask, 'data_quality_flag'] = 'corrupted'
            self.cleaning_stats['large_gaps_flagged'] += num_large
            
        # Identify small gaps for interpolation
        # A gap exists if dt > typical_dt but < gap_threshold
        # First, find typical_dt
        typical_dt = dt.median() if not dt.dropna().empty else 1.0
        
        # We only interpolate if there are actually missing timestamps in between
        # This is tricky without resampling. A simpler approach if rows are NOT missing but values are NaN:
        # But here rows ARE missing because they were dropped by outlier removal.
        
        small_gaps_mask = (dt > typical_dt * 1.5) & (dt < gap_threshold)
        
        if small_gaps_mask.any():
            self.logger.info(f"Interpolating {small_gaps_mask.sum()} small time gaps (< {gap_threshold}s)")
            
            # To properly interpolate, we need to insert the missing rows
            # We'll iterate through the gaps and insert rows
            new_rows = []
            for idx in df.index[small_gaps_mask]:
                t_prev = df.loc[idx-1, time_col]
                t_curr = df.loc[idx, time_col]
                
                # Number of points to insert
                n_points = int(round((t_curr - t_prev) / typical_dt)) - 1
                if n_points > 0:
                    for i in range(1, n_points + 1):
                        new_time = t_prev + i * (t_curr - t_prev) / (n_points + 1)
                        # Create a new row with NaN for other columns
                        new_row = {col: np.nan for col in df.columns}
                        new_row[time_col] = new_time
                        new_row['data_quality_flag'] = 'interpolated'
                        new_rows.append(new_row)
            
            if new_rows:
                df_interp = pd.DataFrame(new_rows)
                df = pd.concat([df, df_interp], ignore_index=True).sort_values(time_col).reset_index(drop=True)
                # Interpolate numeric columns
                df = df.interpolate(method='linear')
                self.cleaning_stats['gaps_interpolated'] += len(new_rows)
                
        return df
