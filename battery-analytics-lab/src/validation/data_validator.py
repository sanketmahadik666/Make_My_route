"""
Battery Analytics Lab - Data Validation Module
Phase 1: Data Ingestion & Standardization

This module handles validation of standardized data against quality criteria
and routes compliant/non-compliant data to appropriate directories.

Author: Battery Analytics Lab Team
Date: 2025-12-29
Version: 1.0
"""

import pandas as pd
import numpy as np
import yaml
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataValidator:
    """
    Main class for validating standardized battery data and routing based on quality.
    """
    
    def __init__(self, config_path: str = "battery-analytics-lab/config/feature_schema.yaml"):
        """
        Initialize the DataValidator with configuration.
        
        Args:
            config_path: Path to the feature schema configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.validation_stats = {
            'files_validated': 0,
            'files_passed': 0,
            'files_failed': 0,
            'validation_errors': 0,
            'total_anomalies_detected': 0
        }
        
        # Create incident reports directory
        self.incident_dir = Path('battery-analytics-lab/logs/incidents')
        self.incident_dir.mkdir(parents=True, exist_ok=True)
    
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
        logger = logging.getLogger('data_validator')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('battery-analytics-lab/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / 'validation.log')
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
    
    def validate_file(self, 
                     file_path: str, 
                     output_passed_dir: str = "battery-analytics-lab/data/validated/passed/",
                     output_failed_dir: str = "battery-analytics-lab/data/validated/failed/") -> Dict[str, Any]:
        """
        Validate a single standardized data file and route to appropriate directory.
        
        Args:
            file_path: Path to the standardized data file
            output_passed_dir: Directory for validated (passed) files
            output_failed_dir: Directory for rejected (failed) files
            
        Returns:
            Dictionary containing validation results and routing information
        """
        try:
            self.logger.info(f"Starting validation of file: {file_path}")
            
            # Read standardized data
            df = self._read_standardized_file(file_path)
            
            # Perform validation checks
            validation_results = self._perform_validation_checks(df, file_path)
            
            # Route file based on validation results
            routing_result = self._route_file(file_path, validation_results, output_passed_dir, output_failed_dir)
            
            # Update statistics
            self.validation_stats['files_validated'] += 1
            if validation_results['passed']:
                self.validation_stats['files_passed'] += 1
            else:
                self.validation_stats['files_failed'] += 1
            
            self.validation_stats['total_anomalies_detected'] += len(validation_results.get('anomalies', []))
            
            self.logger.info(f"Validation completed for {file_path}: {'PASSED' if validation_results['passed'] else 'FAILED'}")
            
            return {
                'status': 'completed',
                'file_path': file_path,
                'validation_passed': validation_results['passed'],
                'quality_score': validation_results['quality_score'],
                'validation_details': validation_results,
                'routing_info': routing_result,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.validation_stats['validation_errors'] += 1
            self.logger.error(f"Error validating file {file_path}: {str(e)}")
            return {
                'status': 'error',
                'file_path': file_path,
                'error_message': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def _read_standardized_file(self, file_path: str) -> pd.DataFrame:
        """Read standardized data file."""
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.logger.info(f"Read {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def _perform_validation_checks(self, df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive validation checks on the data.
        
        Returns:
            Dictionary containing validation results and details
        """
        validation_results = {
            'passed': True,
            'quality_score': 0.0,
            'checks_performed': [],
            'anomalies': [],
            'warnings': [],
            'failed_checks': []
        }
        
        try:
            # 1. Schema compliance check
            schema_result = self._check_schema_compliance(df)
            validation_results['checks_performed'].append('schema_compliance')
            if not schema_result['passed']:
                validation_results['passed'] = False
                validation_results['failed_checks'].append('schema_compliance')
                validation_results['anomalies'].extend(schema_result['anomalies'])
            
            # 2. Data completeness check
            completeness_result = self._check_data_completeness(df)
            validation_results['checks_performed'].append('data_completeness')
            if not completeness_result['passed']:
                validation_results['warnings'].extend(completeness_result['warnings'])
            
            # 3. Value range validation
            range_result = self._check_value_ranges(df)
            validation_results['checks_performed'].append('value_ranges')
            if not range_result['passed']:
                validation_results['passed'] = False
                validation_results['failed_checks'].append('value_ranges')
                validation_results['anomalies'].extend(range_result['anomalies'])
            
            # 4. Data quality metrics
            quality_result = self._calculate_quality_metrics(df)
            validation_results['quality_score'] = quality_result['overall_score']
            validation_results['quality_details'] = quality_result
            
            # 5. Cycle consistency check
            cycle_result = self._check_cycle_consistency(df)
            validation_results['checks_performed'].append('cycle_consistency')
            if not cycle_result['passed']:
                validation_results['warnings'].extend(cycle_result['warnings'])
            
            # Overall pass/fail determination
            min_quality_score = 0.6  # Minimum acceptable quality score
            if validation_results['quality_score'] < min_quality_score:
                validation_results['passed'] = False
                validation_results['failed_checks'].append('quality_score')
            
            self.logger.info(f"Validation checks completed. Overall result: {'PASS' if validation_results['passed'] else 'FAIL'}")
            
        except Exception as e:
            self.logger.error(f"Error during validation checks: {str(e)}")
            validation_results['passed'] = False
            validation_results['failed_checks'].append('validation_process')
            validation_results['anomalies'].append(f"Validation process error: {str(e)}")
        
        return validation_results
    
    def _check_schema_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check if data complies with expected schema."""
        required_columns = self.config['standardized_data_schema']['required_columns']
        available_columns = list(df.columns)
        
        missing_columns = []
        for req_col in required_columns:
            if req_col not in available_columns:
                missing_columns.append(req_col)
        
        result = {
            'passed': len(missing_columns) == 0,
            'anomalies': []
        }
        
        if missing_columns:
            result['anomalies'].append(f"Missing required columns: {missing_columns}")
        
        return result
    
    def _check_data_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness and missing value ratios."""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 1.0
        
        completeness_thresholds = self.config['validation_criteria']['quality_thresholds']
        max_allowed_missing = 1.0 - completeness_thresholds['voltage_completeness']
        
        result = {
            'passed': missing_ratio <= max_allowed_missing,
            'warnings': []
        }
        
        if missing_ratio > max_allowed_missing:
            result['warnings'].append(f"High missing data ratio: {missing_ratio:.2%}")
        
        # Check for columns with excessive missing data
        for col in df.columns:
            col_missing_ratio = df[col].isnull().sum() / len(df)
            if col_missing_ratio > 0.5:  # More than 50% missing
                result['warnings'].append(f"Column '{col}' has {col_missing_ratio:.2%} missing values")
        
        return result
    
    def _check_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check if values are within expected ranges."""
        value_ranges = self.config['raw_data_schema']['value_ranges']
        result = {
            'passed': True,
            'anomalies': []
        }
        
        for col, range_info in value_ranges.items():
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    min_val = col_data.min()
                    max_val = col_data.max()
                    
                    # Check against allowed ranges
                    if min_val < range_info['min'] or max_val > range_info['max']:
                        result['passed'] = False
                        result['anomalies'].append(
                            f"Column '{col}' values outside range [{range_info['min']}-{range_info['max']} {range_info['unit']}]: "
                            f"actual range [{min_val:.3f}-{max_val:.3f}]"
                        )
        
        return result
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics."""
        metrics = {}
        
        # Completeness score
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness_score = 1.0 - (missing_cells / total_cells if total_cells > 0 else 1.0)
        metrics['completeness_score'] = completeness_score
        
        # Consistency score (based on value ranges and expected patterns)
        consistency_score = self._calculate_consistency_score(df)
        metrics['consistency_score'] = consistency_score
        
        # Calculate overall score
        overall_score = (completeness_score + consistency_score) / 2.0
        metrics['overall_score'] = overall_score
        
        return metrics
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        try:
            # Check for reasonable value patterns
            consistency_checks = []
            
            # Check for voltage consistency (should be relatively stable)
            if 'voltage_v' in df.columns:
                voltage_std = df['voltage_v'].std()
                voltage_consistency = 1.0 / (1.0 + voltage_std) if voltage_std > 0 else 1.0
                consistency_checks.append(voltage_consistency)
            
            # Check for reasonable current patterns
            if 'current_a' in df.columns:
                current_range = df['current_a'].max() - df['current_a'].min()
                current_consistency = 1.0 / (1.0 + current_range / 10.0)  # Normalize by expected range
                consistency_checks.append(current_consistency)
            
            # Return average consistency score
            return np.mean(consistency_checks) if consistency_checks else 0.5
            
        except:
            return 0.5  # Default moderate consistency score
    
    def _check_cycle_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check cycle numbering and consistency."""
        result = {
            'passed': True,
            'warnings': []
        }
        
        if 'cycle_number' in df.columns:
            # Check for reasonable cycle numbering
            cycle_numbers = df['cycle_number'].unique()
            if len(cycle_numbers) == 0:
                result['warnings'].append("No cycle numbers found")
            elif any(cycle_numbers < 1):
                result['warnings'].append("Invalid cycle numbers (less than 1)")
            
            # Check for reasonable cycle distribution
            cycle_counts = df['cycle_number'].value_counts()
            if len(cycle_counts) > 0:
                avg_cycle_length = cycle_counts.mean()
                if avg_cycle_length < 10:  # Very short cycles might indicate issues
                    result['warnings'].append(f"Very short cycles detected (avg length: {avg_cycle_length:.1f} points)")
        
        return result
    
    def _route_file(self, file_path: str, validation_results: Dict[str, Any], 
                   output_passed_dir: str, output_failed_dir: str) -> Dict[str, Any]:
        """Route validated file to appropriate directory."""
        source_file = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if validation_results['passed']:
            # Route to passed directory
            output_dir = Path(output_passed_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            new_filename = f"{source_file.stem}_validated_{timestamp}{source_file.suffix}"
            destination = output_dir / new_filename
            
        else:
            # Route to failed directory
            output_dir = Path(output_failed_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            new_filename = f"{source_file.stem}_rejected_{timestamp}{source_file.suffix}"
            destination = output_dir / new_filename
        
        # Copy file to destination
        try:
            shutil.copy2(source_file, destination)
            
            return {
                'routed_to': str(destination),
                'validation_status': 'passed' if validation_results['passed'] else 'failed',
                'timestamp': timestamp,
                'file_copied': True
            }
            
        except Exception as e:
            self.logger.error(f"Error routing file {file_path}: {str(e)}")
            return {
                'routed_to': None,
                'validation_status': 'error',
                'error': str(e),
                'file_copied': False
            }
    
    def generate_incident_report(self, validation_result: Dict[str, Any]) -> str:
        """Generate incident report for failed validations."""
        if validation_result.get('validation_passed', True):
            return None  # No incident report needed for passed validations
        
        report_filename = f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = self.incident_dir / report_filename
        
        report_content = f"""# Data Validation Incident Report

**Generated:** {datetime.now().isoformat()}
**File:** {validation_result['file_path']}
**Validation Status:** FAILED

## Summary
This file failed validation checks and has been routed to the failed directory.

## Validation Details
- **Quality Score:** {validation_result.get('quality_score', 'N/A'):.3f}
- **Validation Passed:** {validation_result.get('validation_passed', False)}

## Failed Checks
{chr(10).join(f"- {check}" for check in validation_result.get('validation_details', {}).get('failed_checks', []))}

## Detected Anomalies
{chr(10).join(f"- {anomaly}" for anomaly in validation_result.get('validation_details', {}).get('anomalies', []))}

## Warnings
{chr(10).join(f"- {warning}" for warning in validation_result.get('validation_details', {}).get('warnings', []))}

## Recommended Actions
1. Review the source data file for data quality issues
2. Check data acquisition procedures
3. Consider data cleaning or preprocessing improvements
4. Update validation thresholds if necessary

## Technical Details
```
{validation_result}
```

---
*Generated by Battery Analytics Lab Validation System v1.0*
"""
        
        try:
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"Incident report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating incident report: {str(e)}")
            return None
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()


def main():
    """Main function for testing the validation module."""
    # Example usage
    validator = DataValidator()
    
    # Test with a sample file
    test_file = "battery-analytics-lab/data/standardized/test_standardized.parquet"
    if Path(test_file).exists():
        result = validator.validate_file(test_file)
        print(f"Validation result: {result}")
        print(f"Validation stats: {validator.get_validation_stats()}")
    else:
        print(f"Test file not found: {test_file}")


if __name__ == "__main__":
    main()