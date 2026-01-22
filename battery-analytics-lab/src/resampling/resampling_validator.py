"""
Battery Analytics Lab - Resampling Validator
Phase 2: Data Resampling for Uniformity

This module provides validation functionality for resampled data to ensure
quality and consistency of the resampling process.

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


class ResamplingValidator:
    """
    Validates resampled data quality and consistency.

    This class provides comprehensive validation for resampled battery data,
    ensuring that the resampling process maintains data integrity and meets
    quality standards.
    """

    def __init__(self, config_path: str = "config/resampling.yaml"):
        """
        Initialize the ResamplingValidator with configuration.

        Args:
            config_path: Path to the resampling configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()

        # Extract validation criteria
        self.validation_criteria = self.config['validation_criteria']
        self.quality_config = self.config['quality_assurance']

        # Validation statistics
        self.validation_stats = {
            'validations_performed': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'quality_warnings': 0,
            'critical_errors': 0
        }

        self.logger.info("ResamplingValidator initialized")

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
        """Set up logging for validation process."""
        logger = logging.getLogger('resampling_validator')
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

    def validate_resampled_data(self,
                               resampled_df: pd.DataFrame,
                               original_df: Optional[pd.DataFrame] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation of resampled data.

        Args:
            resampled_df: The resampled DataFrame to validate
            original_df: Optional original data for comparison
            metadata: Optional metadata about the resampling process

        Returns:
            Dictionary containing validation results
        """
        self.validation_stats['validations_performed'] += 1

        validation_results = {
            'overall_status': 'unknown',
            'validation_timestamp': datetime.now().isoformat(),
            'checks_performed': [],
            'passed_checks': [],
            'failed_checks': [],
            'warnings': [],
            'quality_metrics': {},
            'recommendations': []
        }

        try:
            # Basic structure validation
            structure_check = self._validate_data_structure(resampled_df)
            validation_results['checks_performed'].append('data_structure')
            if structure_check['valid']:
                validation_results['passed_checks'].append('data_structure')
            else:
                validation_results['failed_checks'].extend(structure_check['errors'])

            # Voltage grid validation (for voltage-domain resampling)
            if 'voltage_v' in resampled_df.columns:
                grid_check = self._validate_voltage_grid(resampled_df)
                validation_results['checks_performed'].append('voltage_grid')
                if grid_check['valid']:
                    validation_results['passed_checks'].append('voltage_grid')
                    validation_results['quality_metrics'].update(grid_check['metrics'])
                else:
                    validation_results['failed_checks'].extend(grid_check['errors'])

            # Data completeness validation
            completeness_check = self._validate_data_completeness(resampled_df)
            validation_results['checks_performed'].append('data_completeness')
            if completeness_check['valid']:
                validation_results['passed_checks'].append('data_completeness')
                validation_results['quality_metrics'].update(completeness_check['metrics'])
            else:
                validation_results['failed_checks'].extend(completeness_check['errors'])
                validation_results['warnings'].extend(completeness_check['warnings'])

            # Interpolation quality validation
            if original_df is not None:
                quality_check = self._validate_interpolation_quality(resampled_df, original_df)
                validation_results['checks_performed'].append('interpolation_quality')
                validation_results['quality_metrics'].update(quality_check['metrics'])
                validation_results['warnings'].extend(quality_check['warnings'])

            # Determine overall status
            critical_failures = [check for check in validation_results['failed_checks']
                               if not check.startswith('warning:')]

            if len(critical_failures) == 0:
                validation_results['overall_status'] = 'passed'
                self.validation_stats['passed_validations'] += 1
            else:
                validation_results['overall_status'] = 'failed'
                self.validation_stats['failed_validations'] += 1

            # Generate recommendations
            validation_results['recommendations'] = self._generate_recommendations(validation_results)

        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.validation_stats['critical_errors'] += 1

        return validation_results

    def _validate_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the basic structure of resampled data."""
        result = {'valid': True, 'errors': [], 'warnings': []}

        # Check for required columns based on resampling type
        if 'resampling_method' in df.columns:
            method = df['resampling_method'].iloc[0]
            if method == 'voltage_domain':
                required_cols = ['voltage_v', 'capacity_ah', 'current_a']
            elif method == 'time_domain':
                required_cols = ['timestamp', 'voltage_v', 'current_a']
            else:
                required_cols = ['voltage_v', 'current_a']  # fallback

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                result['valid'] = False
                result['errors'].append(f"Missing required columns: {missing_cols}")

        # Check for metadata columns
        expected_metadata = ['cycle_number', 'cell_id', 'processing_timestamp']
        missing_metadata = [col for col in expected_metadata if col not in df.columns]
        if missing_metadata:
            result['warnings'].append(f"Missing metadata columns: {missing_metadata}")

        # Check data types
        if 'voltage_v' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['voltage_v']):
                result['errors'].append("voltage_v column is not numeric")

        return result

    def _validate_voltage_grid(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate voltage grid uniformity and range."""
        result = {'valid': True, 'errors': [], 'warnings': [], 'metrics': {}}

        if 'voltage_v' not in df.columns:
            result['valid'] = False
            result['errors'].append("No voltage column found for grid validation")
            return result

        voltages = df['voltage_v'].dropna().values

        # Check grid uniformity
        if len(voltages) > 1:
            diffs = np.diff(voltages)
            uniformity_deviation = np.std(diffs)
            result['metrics']['voltage_grid_uniformity'] = uniformity_deviation

            max_allowed_deviation = self.validation_criteria['voltage_grid_uniformity']
            if uniformity_deviation > max_allowed_deviation:
                result['valid'] = False
                result['errors'].append(
                    f"Voltage grid non-uniformity ({uniformity_deviation:.6f}) exceeds threshold ({max_allowed_deviation})"
                )

        # Check voltage range
        v_min, v_max = voltages.min(), voltages.max()
        expected_min = self.config['resampling_strategies']['voltage_domain']['voltage_range']['min']
        expected_max = self.config['resampling_strategies']['voltage_domain']['voltage_range']['max']

        if v_min < expected_min - 0.001 or v_max > expected_max + 0.001:
            result['warnings'].append(
                f"Voltage range [{v_min:.3f}, {v_max:.3f}]V outside expected range [{expected_min}, {expected_max}]V"
            )

        # Check monotonicity
        if self.validation_criteria.get('monotonicity_check', False):
            is_monotonic = self._check_monotonicity(voltages)
            result['metrics']['voltage_monotonic'] = is_monotonic
            if not is_monotonic:
                result['warnings'].append("Voltage not monotonically increasing")

        return result

    def _check_monotonicity(self, values: np.ndarray, threshold: float = 0.7) -> bool:
        """Check if values are generally monotonic."""
        if len(values) < 3:
            return True

        diffs = np.diff(values)
        positive_fraction = np.sum(diffs >= 0) / len(diffs)
        return positive_fraction >= threshold

    def _validate_data_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness and identify gaps."""
        result = {'valid': True, 'errors': [], 'warnings': [], 'metrics': {}}

        required_vars = self.quality_config['completeness_check']['required_variables']
        min_coverage = self.quality_config['completeness_check']['min_coverage']

        for var in required_vars:
            if var in df.columns:
                completeness = df[var].notna().sum() / len(df)
                result['metrics'][f'{var}_completeness'] = completeness

                if completeness < min_coverage:
                    result['valid'] = False
                    result['errors'].append(
                        f"Insufficient {var} completeness: {completeness:.2%} < {min_coverage:.2%}"
                    )
                elif completeness < min_coverage + 0.1:  # Warning threshold
                    result['warnings'].append(
                        f"Low {var} completeness: {completeness:.2%}"
                    )

        return result

    def _validate_interpolation_quality(self,
                                       resampled_df: pd.DataFrame,
                                       original_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate interpolation quality by comparing with original data."""
        result = {'metrics': {}, 'warnings': []}

        if 'voltage_v' in resampled_df.columns and 'voltage_v' in original_df.columns:
            try:
                # Find overlapping voltage range
                resampled_v = resampled_df['voltage_v'].dropna()
                original_v = original_df['voltage_v'].dropna()

                v_overlap = (resampled_v.min() <= original_v) & (original_v <= resampled_v.max())
                overlapping_original = original_df.loc[original_v.index[v_overlap]]

                if len(overlapping_original) > 0:
                    # Calculate interpolation error for capacity
                    if 'capacity_ah' in resampled_df.columns and 'capacity_ah' in overlapping_original.columns:
                        from scipy import interpolate

                        # Create interpolation function from resampled data
                        interp_func = interpolate.interp1d(
                            resampled_v, resampled_df['capacity_ah'].dropna(),
                            kind='linear', bounds_error=False, fill_value=np.nan
                        )

                        # Predict capacity at original voltage points
                        predicted_capacity = interp_func(overlapping_original['voltage_v'])
                        actual_capacity = overlapping_original['capacity_ah']

                        # Calculate RMSE for valid predictions
                        valid_mask = ~(np.isnan(predicted_capacity) | np.isnan(actual_capacity))
                        if valid_mask.sum() > 0:
                            rmse = np.sqrt(np.mean(
                                (predicted_capacity[valid_mask] - actual_capacity.values[valid_mask])**2
                            ))
                            result['metrics']['capacity_interpolation_rmse'] = rmse

                            max_allowed_error = self.validation_criteria.get('interpolation_accuracy', 0.1)
                            if rmse > max_allowed_error:
                                result['warnings'].append(
                                    f"High interpolation error: RMSE={rmse:.4f} > {max_allowed_error}"
                                )

            except Exception as e:
                result['warnings'].append(f"Could not validate interpolation quality: {str(e)}")

        return result

    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if validation_results['overall_status'] == 'failed':
            recommendations.append("Critical validation failures detected - review resampling parameters")

        failed_checks = validation_results.get('failed_checks', [])
        warnings = validation_results.get('warnings', [])

        if any('voltage_grid' in check for check in failed_checks):
            recommendations.append("Adjust voltage grid parameters or check interpolation method")

        if any('completeness' in check.lower() for check in failed_checks):
            recommendations.append("Review data preprocessing to improve completeness before resampling")

        if any('monotonic' in str(warnings).lower()):
            recommendations.append("Consider phase-based cycle segmentation for better monotonicity")

        if not recommendations:
            recommendations.append("Resampling validation passed - data ready for ML models")

        return recommendations

    def validate_resampling_batch(self,
                                 resampled_files: List[str],
                                 original_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate a batch of resampled files.

        Args:
            resampled_files: List of paths to resampled parquet files
            original_files: Optional list of paths to original files for comparison

        Returns:
            Batch validation summary
        """
        batch_results = {
            'batch_timestamp': datetime.now().isoformat(),
            'files_processed': 0,
            'files_passed': 0,
            'files_failed': 0,
            'total_cycles_validated': 0,
            'file_results': []
        }

        for i, resampled_file in enumerate(resampled_files):
            try:
                # Load resampled data
                resampled_df = pd.read_parquet(resampled_file)

                # Load original data if available
                original_df = None
                if original_files and i < len(original_files):
                    try:
                        original_df = pd.read_parquet(original_files[i])
                    except:
                        pass

                # Validate file
                file_result = self.validate_resampled_data(resampled_df, original_df)
                file_result['file_path'] = resampled_file

                batch_results['file_results'].append(file_result)
                batch_results['files_processed'] += 1

                if file_result['overall_status'] == 'passed':
                    batch_results['files_passed'] += 1
                else:
                    batch_results['files_failed'] += 1

                # Count cycles
                if 'cycle_number' in resampled_df.columns:
                    cycle_count = resampled_df['cycle_number'].nunique()
                    batch_results['total_cycles_validated'] += cycle_count

            except Exception as e:
                self.logger.error(f"Error validating file {resampled_file}: {str(e)}")
                batch_results['file_results'].append({
                    'file_path': resampled_file,
                    'overall_status': 'error',
                    'error': str(e)
                })
                batch_results['files_failed'] += 1

        batch_results['success_rate'] = (
            batch_results['files_passed'] / batch_results['files_processed']
            if batch_results['files_processed'] > 0 else 0
        )

        return batch_results

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()


def main():
    """Main function for testing the resampling validator."""
    validator = ResamplingValidator()

    print("Resampling Validator Module")
    print("=" * 30)
    print("Validation criteria loaded:")
    print(f"  Voltage grid uniformity: {validator.validation_criteria['voltage_grid_uniformity']}")
    print(f"  Data completeness: {validator.quality_config['completeness_check']['min_coverage']:.2%}")
    print(f"  Interpolation accuracy: {validator.validation_criteria['interpolation_accuracy']}")


if __name__ == "__main__":
    main()
