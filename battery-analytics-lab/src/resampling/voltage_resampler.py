"""
Battery Analytics Lab - Voltage-Based Resampler
Phase 2: Data Resampling for Uniformity

This module implements voltage-based resampling, which interpolates capacity (Q)
and current (I) onto a fixed grid of voltage points. This ensures uniform input
vectors for machine learning models and aligns electrochemical features across
different cycles.

Author: Battery Analytics Lab Team
Date: 2026-01-22
Version: 1.0
"""

import pandas as pd
import numpy as np
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import sys
import os
from scipy import interpolate
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class VoltageResampler:
    """
    Implements voltage-based resampling for electrochemical data alignment.

    This class resamples battery cycling data by interpolating capacity and current
    onto a fixed voltage grid (3.0V to 4.2V in 10mV increments), ensuring uniform
    input vectors for ML models and aligning phase transitions across cycles.
    """

    def __init__(self, config_path: str = "config/resampling.yaml"):
        """
        Initialize the VoltageResampler with configuration.

        Args:
            config_path: Path to the resampling configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.validator = None  # Will be initialized when needed

        # Extract voltage grid parameters
        voltage_config = self.config['resampling_strategies']['voltage_domain']
        self.voltage_min = voltage_config['voltage_range']['min']
        self.voltage_max = voltage_config['voltage_range']['max']
        self.voltage_step = voltage_config['voltage_range']['step']
        self.interpolation_method = voltage_config['interpolation_method']
        self.extrapolate = voltage_config['extrapolate']
        self.variables_to_interpolate = voltage_config['variables_to_interpolate']

        # Create uniform voltage grid
        self.voltage_grid = np.arange(self.voltage_min, self.voltage_max + self.voltage_step, self.voltage_step)
        self.num_grid_points = len(self.voltage_grid)

        # Processing statistics
        self.processing_stats = {
            'files_processed': 0,
            'cycles_processed': 0,
            'total_interpolation_points': 0,
            'successful_resamplings': 0,
            'failed_resamplings': 0,
            'quality_warnings': 0
        }

        self.logger.info(f"VoltageResampler initialized with {self.num_grid_points} grid points "
                        f"({self.voltage_min}V to {self.voltage_max}V in {self.voltage_step*1000:.0f}mV steps)")

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
        logger = logging.getLogger('voltage_resampler')
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
        Resample a single cycle's data onto the voltage grid.

        Args:
            df: DataFrame containing cycle data with voltage_v, capacity_ah, current_a columns
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

            # Prepare data for interpolation
            processed_data = self._prepare_data_for_interpolation(df)

            # Perform voltage-based interpolation
            resampled_df = self._interpolate_to_voltage_grid(processed_data)

            # Add metadata columns
            resampled_df = self._add_resampling_metadata(resampled_df, cycle_number, cell_id)

            # Validate resampled data
            quality_metrics = self._validate_resampled_data(resampled_df, processed_data)

            # Prepare metadata
            metadata = {
                'status': 'success',
                'cycle_number': cycle_number,
                'cell_id': cell_id,
                'original_points': len(df),
                'resampled_points': len(resampled_df),
                'voltage_range': [self.voltage_min, self.voltage_max],
                'voltage_step': self.voltage_step,
                'interpolation_method': self.interpolation_method,
                'quality_metrics': quality_metrics,
                'processing_timestamp': datetime.now().isoformat()
            }

            self.processing_stats['successful_resamplings'] += 1
            self.processing_stats['cycles_processed'] += 1

            return resampled_df, metadata

        except Exception as e:
            self.processing_stats['failed_resamplings'] += 1
            self.logger.error(f"Error resampling cycle data: {str(e)}")
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

        required_cols = ['voltage_v', 'capacity_ah', 'current_a']
        quality_config = self.config['quality_assurance']

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_cols}")
            return validation_result

        # Check data completeness
        for col in required_cols:
            non_null_ratio = df[col].notna().sum() / len(df)
            min_coverage = quality_config['completeness_check']['min_coverage']
            if non_null_ratio < min_coverage:
                validation_result['valid'] = False
                validation_result['errors'].append(
                    f"Insufficient data coverage for {col}: {non_null_ratio:.2%} < {min_coverage:.2%}"
                )

        # Check minimum interpolation points
        if len(df.dropna(subset=required_cols)) < quality_config['min_interpolation_points']:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Insufficient data points for interpolation: {len(df)} < {quality_config['min_interpolation_points']}"
            )

        # Check voltage range
        voltage_range = df['voltage_v'].dropna()
        if voltage_range.empty:
            validation_result['valid'] = False
            validation_result['errors'].append("No valid voltage data found")
        else:
            data_min, data_max = voltage_range.min(), voltage_range.max()
            if data_max < self.voltage_min or data_min > self.voltage_max:
                validation_result['valid'] = False
                validation_result['errors'].append(
                    f"Voltage range [{data_min:.3f}, {data_max:.3f}]V outside grid range "
                    f"[{self.voltage_min}, {self.voltage_max}]V"
                )

        # Check for monotonic voltage (should generally increase during discharge)
        if not self._is_voltage_monotonic(df):
            validation_result['warnings'].append("Voltage not monotonically increasing")

        return validation_result

    def _is_voltage_monotonic(self, df: pd.DataFrame, direction: str = 'increasing') -> bool:
        """Check if voltage is generally monotonic in the specified direction."""
        voltage = df['voltage_v'].dropna()
        if len(voltage) < 3:
            return True  # Not enough points to determine

        # Check for general trend (allowing some noise)
        diffs = np.diff(voltage.values)
        positive_trend = np.sum(diffs > 0) / len(diffs)

        # Require at least 70% of differences to be in the expected direction
        return positive_trend >= 0.7

    def _prepare_data_for_interpolation(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare data by cleaning and sorting for interpolation."""
        # Remove rows with missing voltage values
        clean_df = df.dropna(subset=['voltage_v']).copy()

        # Sort by voltage to ensure monotonic ordering
        clean_df = clean_df.sort_values('voltage_v')

        # Remove duplicate voltage values (keep first occurrence)
        clean_df = clean_df.drop_duplicates(subset='voltage_v', keep='first')

        # Extract arrays for interpolation
        processed_data = {
            'voltage': clean_df['voltage_v'].values,
            'capacity': clean_df['capacity_ah'].values,
            'current': clean_df['current_a'].values
        }

        # Add optional variables if available
        for var in self.variables_to_interpolate:
            if var in clean_df.columns:
                processed_data[var] = clean_df[var].values

        return processed_data

    def _interpolate_to_voltage_grid(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Interpolate data onto the fixed voltage grid."""
        # Create result DataFrame with voltage grid
        result_df = pd.DataFrame({'voltage_v': self.voltage_grid})

        # Interpolate each variable
        for var_name, var_data in data.items():
            if var_name == 'voltage':
                continue  # Skip voltage, it's already the grid

            try:
                # Perform linear interpolation
                interp_func = interpolate.interp1d(
                    data['voltage'], var_data,
                    kind=self.interpolation_method,
                    bounds_error=not self.extrapolate,
                    fill_value=np.nan if not self.extrapolate else 'extrapolate'
                )

                interpolated_values = interp_func(self.voltage_grid)

                # Map variable names back to standardized column names
                col_name = var_name
                if var_name == 'capacity':
                    col_name = 'capacity_ah'
                elif var_name == 'current':
                    col_name = 'current_a'

                result_df[col_name] = interpolated_values

            except Exception as e:
                self.logger.warning(f"Interpolation failed for {var_name}: {str(e)}")
                result_df[var_name] = np.nan

        return result_df

    def _add_resampling_metadata(self,
                                df: pd.DataFrame,
                                cycle_number: Optional[int],
                                cell_id: Optional[str]) -> pd.DataFrame:
        """Add metadata columns to resampled DataFrame."""
        df_copy = df.copy()

        # Add resampling metadata
        df_copy['cycle_number'] = cycle_number
        df_copy['cell_id'] = cell_id
        df_copy['resampling_method'] = 'voltage_domain'
        df_copy['voltage_grid_index'] = range(len(df_copy))
        df_copy['processing_timestamp'] = datetime.now().isoformat()

        return df_copy

    def _validate_resampled_data(self,
                                resampled_df: pd.DataFrame,
                                original_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate the quality of resampled data."""
        metrics = {}

        # Check data completeness
        for col in ['capacity_ah', 'current_a']:
            if col in resampled_df.columns:
                completeness = resampled_df[col].notna().sum() / len(resampled_df)
                metrics[f'{col}_completeness'] = completeness

        # Check interpolation accuracy (compare with original data where possible)
        if 'capacity_ah' in resampled_df.columns and len(original_data.get('voltage', [])) > 0:
            # Find original voltage points that fall within grid
            orig_voltage = original_data['voltage']
            orig_capacity = original_data['capacity']

            mask = (orig_voltage >= self.voltage_min) & (orig_voltage <= self.voltage_max)
            if mask.sum() > 0:
                # Interpolate original data at these points for comparison
                try:
                    interp_func = interpolate.interp1d(
                        resampled_df['voltage_v'], resampled_df['capacity_ah'],
                        kind='linear', bounds_error=False, fill_value=np.nan
                    )
                    predicted_capacity = interp_func(orig_voltage[mask])
                    actual_capacity = orig_capacity[mask]

                    # Calculate RMSE where both are valid
                    valid_mask = ~(np.isnan(predicted_capacity) | np.isnan(actual_capacity))
                    if valid_mask.sum() > 0:
                        rmse = np.sqrt(np.mean((predicted_capacity[valid_mask] - actual_capacity[valid_mask])**2))
                        metrics['capacity_interpolation_rmse'] = rmse
                except:
                    pass

        # Check voltage grid uniformity
        voltage_diffs = np.diff(resampled_df['voltage_v'])
        uniformity_deviation = np.std(voltage_diffs)
        metrics['voltage_grid_uniformity'] = uniformity_deviation

        return metrics

    def resample_file(self,
                     file_path: str,
                     output_dir: str = None,
                     parallel: bool = True) -> Dict[str, Any]:
        """
        Resample all cycles in a standardized data file.

        Args:
            file_path: Path to standardized parquet file
            output_dir: Output directory (uses config default if None)
            parallel: Whether to process cycles in parallel

        Returns:
            Dictionary containing processing results and metadata
        """
        try:
            self.logger.info(f"Starting voltage resampling of file: {file_path}")

            # Load standardized data
            df = pd.read_parquet(file_path)
            file_path_obj = Path(file_path)

            # Extract metadata
            cell_id = df['cell_id'].iloc[0] if 'cell_id' in df.columns else file_path_obj.stem.split('_')[0]
            file_metadata = {
                'source_file': file_path,
                'cell_id': cell_id,
                'total_records': len(df),
                'processing_start': datetime.now().isoformat()
            }

            # Group data by cycle
            if 'Cycle_Index' in df.columns:
                cycle_column = 'Cycle_Index'
            elif 'cycle_number' in df.columns:
                cycle_column = 'cycle_number'
            else:
                # Try to infer cycles from phase changes
                cycle_column = self._infer_cycles(df)

            # Process each cycle
            resampled_cycles = []
            cycle_metadata = []

            if parallel and len(df[cycle_column].unique()) > 1:
                results = self._process_cycles_parallel(df, cycle_column, cell_id)
            else:
                results = self._process_cycles_sequential(df, cycle_column, cell_id)

            # Collect results
            for cycle_num, (resampled_df, metadata) in results.items():
                if not resampled_df.empty:
                    resampled_df['cycle_number'] = cycle_num
                    resampled_cycles.append(resampled_df)
                    cycle_metadata.append(metadata)

            # Combine all resampled cycles
            if resampled_cycles:
                final_df = pd.concat(resampled_cycles, ignore_index=True)

                # Set output directory
                if output_dir is None:
                    output_dir = self.config['output_spec']['directory'] + self.config['output_spec']['subdirectories']['voltage_domain']

                # Save resampled data
                output_path = self._save_resampled_data(final_df, output_dir, cell_id)

                file_metadata.update({
                    'status': 'success',
                    'output_path': output_path,
                    'cycles_processed': len(resampled_cycles),
                    'total_resampled_points': len(final_df),
                    'processing_end': datetime.now().isoformat(),
                    'cycle_metadata': cycle_metadata
                })
            else:
                file_metadata.update({
                    'status': 'no_valid_cycles',
                    'cycles_processed': 0,
                    'error': 'No cycles could be successfully resampled'
                })

            self.processing_stats['files_processed'] += 1
            return file_metadata

        except Exception as e:
            self.logger.error(f"Error resampling file {file_path}: {str(e)}")
            return {
                'status': 'error',
                'source_file': file_path,
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }

    def _infer_cycles(self, df: pd.DataFrame) -> str:
        """Infer cycle numbers from phase transitions if not explicitly available."""
        # Simple approach: add cycle column based on rest periods
        df = df.copy()
        df['Cycle_Index'] = 1

        if 'phase_type' in df.columns:
            current_cycle = 1
            in_rest_after_discharge = False

            for i in range(len(df)):
                phase = df.loc[i, 'phase_type']
                if phase == 'discharge':
                    in_rest_after_discharge = True
                elif phase == 'charge' and in_rest_after_discharge:
                    current_cycle += 1
                    in_rest_after_discharge = False

                df.loc[i, 'Cycle_Index'] = current_cycle

        return 'Cycle_Index'

    def _process_cycles_sequential(self,
                                  df: pd.DataFrame,
                                  cycle_column: str,
                                  cell_id: str) -> Dict[int, Tuple[pd.DataFrame, Dict]]:
        """Process cycles sequentially."""
        results = {}

        for cycle_num in df[cycle_column].unique():
            if pd.isna(cycle_num):
                continue

            cycle_data = df[df[cycle_column] == cycle_num]
            resampled_df, metadata = self.resample_cycle_data(cycle_data, cycle_num, cell_id)
            results[cycle_num] = (resampled_df, metadata)

        return results

    def _process_cycles_parallel(self,
                                df: pd.DataFrame,
                                cycle_column: str,
                                cell_id: str) -> Dict[int, Tuple[pd.DataFrame, Dict]]:
        """Process cycles in parallel using multiprocessing."""
        results = {}
        cycle_groups = []

        # Prepare cycle data for parallel processing
        for cycle_num in df[cycle_column].unique():
            if pd.isna(cycle_num):
                continue
            cycle_data = df[df[cycle_column] == cycle_num].copy()
            cycle_groups.append((cycle_num, cycle_data, cell_id))

        # Process in parallel
        num_workers = min(self.config['processing_params']['num_workers'], len(cycle_groups))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_cycle = {
                executor.submit(self._resample_single_cycle, cycle_num, cycle_data, cell_id): cycle_num
                for cycle_num, cycle_data, cell_id in cycle_groups
            }

            # Collect results
            for future in as_completed(future_to_cycle):
                cycle_num = future_to_cycle[future]
                try:
                    resampled_df, metadata = future.result()
                    results[cycle_num] = (resampled_df, metadata)
                except Exception as e:
                    self.logger.error(f"Parallel processing failed for cycle {cycle_num}: {str(e)}")
                    results[cycle_num] = (pd.DataFrame(), {'status': 'error', 'error': str(e)})

        return results

    def _resample_single_cycle(self,
                              cycle_num: int,
                              cycle_data: pd.DataFrame,
                              cell_id: str) -> Tuple[pd.DataFrame, Dict]:
        """Helper method for parallel cycle processing."""
        return self.resample_cycle_data(cycle_data, cycle_num, cell_id)

    def _save_resampled_data(self,
                            df: pd.DataFrame,
                            output_dir: str,
                            cell_id: str) -> str:
        """Save resampled data to parquet format."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{cell_id}_voltage_domain_resampled_{timestamp}.parquet"
        output_file = output_path / filename

        # Save to parquet
        df.to_parquet(output_file, index=False, compression=self.config['output_spec']['compression'])

        self.logger.info(f"Saved resampled data to {output_file}")
        return str(output_file)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()


def main():
    """Main function for testing the voltage resampler."""
    import argparse

    parser = argparse.ArgumentParser(description='Resample battery data using voltage-based interpolation.')
    parser.add_argument('--input', '-i', type=str, help='Input standardized parquet file')
    parser.add_argument('--output', '-o', type=str, help='Output directory')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')

    args = parser.parse_args()

    resampler = VoltageResampler()

    if args.input:
        result = resampler.resample_file(args.input, args.output, args.parallel)
        print(f"Resampling result: {result['status']}")
        if 'cycles_processed' in result:
            print(f"Cycles processed: {result['cycles_processed']}")
    else:
        print("Voltage-based Resampling Module")
        print("=" * 40)
        print(f"Voltage grid: {resampler.voltage_min}V to {resampler.voltage_max}V")
        print(f"Step size: {resampler.voltage_step*1000:.0f} mV")
        print(f"Grid points: {resampler.num_grid_points}")
        print(f"Interpolation method: {resampler.interpolation_method}")


if __name__ == "__main__":
    main()
