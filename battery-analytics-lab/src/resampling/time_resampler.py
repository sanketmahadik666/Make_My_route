"""
Battery Analytics Lab - Time-Based Resampler
Phase 2: Data Resampling for Uniformity

This module implements time-based resampling, which interpolates data to a fixed
frequency. This preserves temporal structure but may result in variable vector
lengths for different cycles.

Author: Battery Analytics Lab Team
Date: 2026-01-22
Version: 1.0
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TimeResampler:
    """
    Implements time-based resampling for temporal data alignment.

    This class resamples battery cycling data by interpolating to a fixed frequency
    (e.g., 1 Hz), preserving the temporal structure of the data.
    """

    def __init__(self, config_path: str = "config/resampling.yaml"):
        """
        Initialize the TimeResampler with configuration.

        Args:
            config_path: Path to the resampling configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()

        # Extract time domain parameters
        time_config = self.config['resampling_strategies']['time_domain']
        self.frequency = time_config['frequency']  # Hz
        self.interpolation_method = time_config['interpolation_method']
        self.extrapolate = time_config['extrapolate']

        # Calculate sampling interval
        self.sampling_interval = 1.0 / self.frequency  # seconds

        # Processing statistics
        self.processing_stats = {
            'files_processed': 0,
            'cycles_processed': 0,
            'total_interpolation_points': 0,
            'successful_resamplings': 0,
            'failed_resamplings': 0
        }

        self.logger.info(f"TimeResampler initialized with {self.frequency} Hz frequency "
                        f"({self.sampling_interval:.3f}s intervals)")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for resampling process."""
        logger = logging.getLogger('time_resampler')
        logger.setLevel(getattr(logging, self.config['logging']['level']))

        # Create logs directory if it doesn't exist
        log_dir = Path('battery-analytics-lab/logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        log_file = self.config['logging']['log_file']
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(getattr(logging, self.config['logging']['level']))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config['logging']['level']))

        # Formatter
        formatter = logging.Formatter(self.config['logging']['log_format'])
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def resample_cycle_data(self,
                           df: pd.DataFrame,
                           cycle_number: Optional[int] = None,
                           cell_id: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Resample a single cycle's data to fixed time intervals.

        Args:
            df: DataFrame containing cycle data with timestamp column
            cycle_number: Cycle number for metadata
            cell_id: Cell identifier for metadata

        Returns:
            Tuple of (resampled_dataframe, metadata_dict)
        """
        try:
            # Validate input data
            validation_result = self._validate_cycle_data(df)
            if not validation_result['valid']:
                self.processing_stats['failed_resamplings'] += 1
                return pd.DataFrame(), {
                    'status': 'failed',
                    'error': 'validation_failed',
                    'validation_errors': validation_result['errors'],
                    'cycle_number': cycle_number,
                    'cell_id': cell_id
                }

            # Create time index for resampling
            time_index = self._create_time_index(df)

            # Resample data
            resampled_df = self._resample_to_time_grid(df, time_index)

            # Add metadata
            resampled_df = self._add_resampling_metadata(resampled_df, cycle_number, cell_id)

            # Prepare metadata
            metadata = {
                'status': 'success',
                'cycle_number': cycle_number,
                'cell_id': cell_id,
                'original_points': len(df),
                'resampled_points': len(resampled_df),
                'frequency': self.frequency,
                'sampling_interval': self.sampling_interval,
                'time_range': [time_index.min(), time_index.max()],
                'interpolation_method': self.interpolation_method,
                'processing_timestamp': datetime.now().isoformat()
            }

            self.processing_stats['successful_resamplings'] += 1
            self.processing_stats['cycles_processed'] += 1

            return resampled_df, metadata

        except Exception as e:
            self.processing_stats['failed_resamplings'] += 1
            self.logger.error(f"Error time-based resampling cycle data: {str(e)}")
            return pd.DataFrame(), {
                'status': 'error',
                'error': str(e),
                'cycle_number': cycle_number,
                'cell_id': cell_id,
                'processing_timestamp': datetime.now().isoformat()
            }

    def _validate_cycle_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that cycle data has required columns and sufficient quality."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check for timestamp column
        if 'timestamp' not in df.columns:
            validation_result['valid'] = False
            validation_result['errors'].append("Missing timestamp column")
            return validation_result

        # Check data completeness
        non_null_ratio = df['timestamp'].notna().sum() / len(df)
        if non_null_ratio < 0.8:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Insufficient timestamp data: {non_null_ratio:.2%} non-null"
            )

        # Check minimum data points
        if len(df.dropna(subset=['timestamp'])) < 3:
            validation_result['valid'] = False
            validation_result['errors'].append("Insufficient data points for time-based resampling")

        return validation_result

    def _create_time_index(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Create a regular time index for resampling."""
        # Get time range from data
        timestamps = pd.to_numeric(df['timestamp'].dropna())
        time_min = timestamps.min()
        time_max = timestamps.max()

        # Create regular time grid
        time_grid = np.arange(time_min, time_max + self.sampling_interval, self.sampling_interval)

        return pd.to_datetime(time_grid, unit='s')

    def _resample_to_time_grid(self, df: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Resample data to the regular time grid."""
        # Create a copy with datetime index
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='s')
        df_copy = df_copy.set_index('datetime')

        # Resample to regular frequency
        freq_str = f"{self.sampling_interval}S"
        resampled = df_copy.resample(freq_str).interpolate(method=self.interpolation_method)

        # Reset index and add timestamp back
        resampled = resampled.reset_index()
        resampled['timestamp'] = (resampled['datetime'] - resampled['datetime'].min()).dt.total_seconds()

        # Drop datetime column
        resampled = resampled.drop(columns=['datetime'])

        return resampled

    def _add_resampling_metadata(self,
                                df: pd.DataFrame,
                                cycle_number: Optional[int],
                                cell_id: Optional[str]) -> pd.DataFrame:
        """Add metadata columns to resampled DataFrame."""
        df_copy = df.copy()

        # Add resampling metadata
        df_copy['cycle_number'] = cycle_number
        df_copy['cell_id'] = cell_id
        df_copy['resampling_method'] = 'time_domain'
        df_copy['time_grid_index'] = range(len(df_copy))
        df_copy['processing_timestamp'] = datetime.now().isoformat()

        return df_copy

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()


def main():
    """Main function for testing the time resampler."""
    resampler = TimeResampler()

    print("Time-based Resampling Module")
    print("=" * 35)
    print(f"Frequency: {resampler.frequency} Hz")
    print(f"Sampling interval: {resampler.sampling_interval:.3f} seconds")
    print(f"Interpolation method: {resampler.interpolation_method}")
    print("Note: Time-based resampling is disabled in current configuration")


if __name__ == "__main__":
    main()
