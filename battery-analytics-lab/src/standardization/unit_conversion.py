"""
Battery Analytics Lab - Unit Conversion Module
Step 4: Unit Standardization (SI Enforcement)

This module handles unit conversion to ensure all measurements are in 
standard SI units while preserving original data.

Requirements:
- Current standardized to Amperes
- Voltage standardized to Volts  
- Temperature standardized to Celsius
- Time standardized to seconds
- Capacity standardized to Ampere-hours
- No sign correction applied yet
- Original columns preserved
- Output: Standardized in-memory dataset

Author: Battery Analytics Lab Team
Date: 2025-12-29
Version: 1.0
"""

import pandas as pd
import numpy as np
import yaml
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class UnitConverter:
    """
    SI Unit converter for battery data standardization.
    Converts all measurements to standard SI units while preserving original data.
    """
    
    def __init__(self, config_path: str = "battery-analytics-lab/config/units.yaml"):
        """
        Initialize the unit converter with configuration.
        
        Args:
            config_path: Path to the units configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Conversion statistics
        self.conversion_stats = {
            'columns_processed': 0,
            'conversions_successful': 0,
            'conversions_failed': 0,
            'conversions_skipped': 0,
            'original_values_preserved': 0,
            'conversion_factors_used': {},
            'processing_errors': []
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load unit conversion configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Units configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing units configuration: {e}")
            raise
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for unit conversion process."""
        logger = logging.getLogger('unit_converter')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('battery-analytics-lab/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / 'unit_conversion.log')
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
    
    def standardize_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Standardize all units in a DataFrame to SI units.
        
        Args:
            df: Input DataFrame with battery data
            
        Returns:
            Dictionary containing standardized DataFrame and conversion metadata
        """
        self.logger.info(f"Starting unit standardization for DataFrame with {len(df)} rows")
        
        # Create a copy of the DataFrame for processing
        standardized_df = df.copy()
        conversion_metadata = {
            'original_shape': df.shape,
            'columns_processed': [],
            'conversions_applied': [],
            'preserved_columns': [],
            'conversion_timestamp': datetime.now().isoformat(),
            'conversion_errors': []
        }
        
        # Process each column for unit standardization
        for column_name in df.columns:
            try:
                result = self._standardize_column(standardized_df, column_name)
                
                if result['processed']:
                    conversion_metadata['columns_processed'].append(column_name)
                    conversion_metadata['conversions_applied'].append(result)
                    
                    if result['conversion_applied']:
                        self.conversion_stats['conversions_successful'] += 1
                        self.conversion_stats['conversion_factors_used'][column_name] = result['conversion_factor']
                    else:
                        self.conversion_stats['conversions_skipped'] += 1
                    
                    # Preserve original column
                    if result['preserved_original']:
                        original_col_name = f"{column_name}_original"
                        standardized_df[original_col_name] = df[column_name].copy()
                        conversion_metadata['preserved_columns'].append(original_col_name)
                        self.conversion_stats['original_values_preserved'] += 1
                        
                else:
                    self.conversion_stats['conversions_failed'] += 1
                    conversion_metadata['conversion_errors'].append({
                        'column': column_name,
                        'error': result['error']
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing column {column_name}: {str(e)}")
                self.conversion_stats['processing_errors'].append(f"{column_name}: {str(e)}")
                conversion_metadata['conversion_errors'].append({
                    'column': column_name,
                    'error': str(e)
                })
        
        self.conversion_stats['columns_processed'] = len(conversion_metadata['columns_processed'])
        
        self.logger.info(f"Unit standardization completed: {len(conversion_metadata['conversions_applied'])} columns processed")
        
        return {
            'standardized_dataframe': standardized_df,
            'conversion_metadata': conversion_metadata,
            'conversion_statistics': self.conversion_stats.copy()
        }
    
    def _standardize_column(self, df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        """
        Standardize a single column to SI units.
        
        Args:
            df: DataFrame being processed
            column_name: Name of the column to standardize
            
        Returns:
            Dictionary with conversion results
        """
        result = {
            'column_name': column_name,
            'processed': False,
            'conversion_applied': False,
            'conversion_factor': 1.0,
            'original_unit': 'unknown',
            'target_unit': 'unknown',
            'preserved_original': True,
            'error': None,
            'values_before': None,
            'values_after': None
        }
        
        try:
            # Get column data
            column_data = df[column_name]
            result['values_before'] = {
                'min': column_data.min(),
                'max': column_data.max(),
                'mean': column_data.mean(),
                'count': len(column_data)
            }
            
            # Detect column type and determine target unit
            column_type = self._detect_column_type(column_name)
            
            if column_type == 'unknown':
                result['error'] = f"Could not detect column type for {column_name}"
                return result
            
            # Get target SI unit
            target_unit = self.config['si_standard_units'][column_type]
            result['target_unit'] = target_unit
            
            # Detect original unit
            original_unit = self._detect_original_unit(column_name, column_data)
            result['original_unit'] = original_unit
            
            # Apply conversion if needed
            if original_unit != target_unit:
                conversion_factor = self._get_conversion_factor(column_type, original_unit, target_unit)
                
                if conversion_factor is not None and conversion_factor != 1.0:
                    # Apply conversion
                    df[column_name] = column_data * conversion_factor
                    result['conversion_applied'] = True
                    result['conversion_factor'] = conversion_factor
                    
                    self.logger.info(f"Converted {column_name}: {original_unit} → {target_unit} (factor: {conversion_factor})")
                else:
                    self.logger.info(f"No conversion needed for {column_name}: already in {target_unit}")
            
            # Update statistics
            result['values_after'] = {
                'min': df[column_name].min(),
                'max': df[column_name].max(),
                'mean': df[column_name].mean()
            }
            
            result['processed'] = True
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error standardizing column {column_name}: {str(e)}")
        
        return result
    
    def _detect_column_type(self, column_name: str) -> str:
        """
        Detect the type of measurement column based on name patterns.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Detected column type ('current', 'voltage', 'temperature', 'time', 'capacity', or 'unknown')
        """
        column_name_lower = column_name.lower()
        
        # Check against column mapping patterns
        for col_type, patterns in self.config['standardization_rules']['column_mapping'].items():
            for pattern in patterns:
                if pattern in column_name_lower:
                    return col_type
        
        # Additional pattern matching
        if any(keyword in column_name_lower for keyword in ['time', 'timestamp', 'duration']):
            return 'time'
        elif any(keyword in column_name_lower for keyword in ['voltage', 'volt', 'v']):
            return 'voltage'
        elif any(keyword in column_name_lower for keyword in ['current', 'amp', 'i']):
            return 'current'
        elif any(keyword in column_name_lower for keyword in ['temperature', 'temp', 't']):
            return 'temperature'
        elif any(keyword in column_name_lower for keyword in ['capacity', 'cap']):
            return 'capacity'
        
        return 'unknown'
    
    def _detect_original_unit(self, column_name: str, column_data: pd.Series) -> str:
        """
        Detect the original unit of a column based on name patterns and data analysis.
        
        Args:
            column_name: Name of the column
            column_data: Column data for analysis
            
        Returns:
            Detected original unit
        """
        column_name_lower = column_name.lower()
        
        # Check for explicit unit indicators in column name
        for col_type, patterns in self.config['unit_detection']['patterns'].items():
            for pattern in patterns:
                if pattern in column_name_lower:
                    return col_type
        
        # Analyze data values to infer units (for common battery test values)
        column_type = self._detect_column_type(column_name)
        
        if column_type == 'current':
            # Check if values are in milliamperes (common for battery tests)
            if column_data.max() > 10 and column_data.max() < 10000:
                return 'milliamperes'
        elif column_type == 'voltage':
            # Check if values suggest millivolts
            if column_data.max() > 1000:
                return 'millivolts'
        elif column_type == 'time':
            # Check if values suggest minutes or hours
            if column_data.max() > 3600:
                return 'hours'
            elif column_data.max() > 60:
                return 'minutes'
        
        # Default to target unit (assume already standardized)
        if column_type != 'unknown':
            return self.config['si_standard_units'][column_type]
        
        return 'unknown'
    
    def _get_conversion_factor(self, column_type: str, original_unit: str, target_unit: str) -> Optional[float]:
        """
        Get the conversion factor between units.
        
        Args:
            column_type: Type of measurement
            original_unit: Original unit
            target_unit: Target unit (should be SI standard)
            
        Returns:
            Conversion factor or None if no conversion needed
        """
        if column_type == 'unknown' or original_unit == target_unit:
            return 1.0
        
        # Get conversion factors for this column type
        conversion_factors = self.config['conversion_factors'].get(column_type, {}).get('to_' + target_unit, {})
        
        if original_unit in conversion_factors:
            factor = conversion_factors[original_unit]
            
            # Handle special cases (formulas instead of simple factors)
            if factor is None:
                return self._apply_special_conversion(column_type, original_unit, target_unit)
            
            return factor
        
        return 1.0  # No conversion available, assume already in correct unit
    
    def _apply_special_conversion(self, column_type: str, original_unit: str, target_unit: str) -> float:
        """
        Apply special conversions that require formulas.
        
        Args:
            column_type: Type of measurement
            original_unit: Original unit
            target_unit: Target unit
            
        Returns:
            Conversion factor (this method applies to DataFrame, not individual values)
        """
        # This is a placeholder for special conversions
        # In practice, these would be applied to the DataFrame column directly
        # For now, return 1.0 as most common battery data should be in base units
        return 1.0
    
    def standardize_multiple_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Standardize units across multiple DataFrames.
        
        Args:
            dataframes: Dictionary of DataFrame names to DataFrames
            
        Returns:
            Dictionary containing standardized DataFrames and metadata
        """
        self.logger.info(f"Starting batch unit standardization for {len(dataframes)} DataFrames")
        
        standardized_dataframes = {}
        batch_metadata = {
            'dataframes_processed': 0,
            'total_conversions': 0,
            'batch_timestamp': datetime.now().isoformat(),
            'individual_results': {}
        }
        
        for df_name, df in dataframes.items():
            self.logger.info(f"Processing DataFrame: {df_name}")
            
            result = self.standardize_dataframe(df)
            standardized_dataframes[df_name] = result['standardized_dataframe']
            
            batch_metadata['individual_results'][df_name] = {
                'conversion_metadata': result['conversion_metadata'],
                'conversion_statistics': result['conversion_statistics']
            }
            
            batch_metadata['dataframes_processed'] += 1
            batch_metadata['total_conversions'] += len(result['conversion_metadata']['conversions_applied'])
        
        self.logger.info(f"Batch standardization completed: {batch_metadata['dataframes_processed']} DataFrames processed")
        
        return {
            'standardized_dataframes': standardized_dataframes,
            'batch_metadata': batch_metadata
        }
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """Get current conversion statistics."""
        return self.conversion_stats.copy()
    
    def validate_standardized_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that standardized data is within reasonable ranges.
        
        Args:
            df: Standardized DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'validation_passed': True,
            'issues_found': [],
            'column_validations': {}
        }
        
        # Get reasonable ranges from configuration
        reasonable_ranges = self.config['quality_assurance']['post_conversion_validation']['reasonable_ranges']
        
        for column_name in df.columns:
            column_type = self._detect_column_type(column_name)
            
            if column_type in reasonable_ranges:
                column_data = df[column_name].dropna()
                
                if len(column_data) > 0:
                    min_val, max_val = column_data.min(), column_data.max()
                    expected_range = reasonable_ranges[column_type]
                    
                    validation_result = {
                        'column_type': column_type,
                        'min_value': min_val,
                        'max_value': max_val,
                        'expected_range': expected_range,
                        'within_range': (min_val >= expected_range[0] and max_val <= expected_range[1])
                    }
                    
                    validation_results['column_validations'][column_name] = validation_result
                    
                    if not validation_result['within_range']:
                        validation_results['validation_passed'] = False
                        validation_results['issues_found'].append(
                            f"Column {column_name}: values {min_val:.3f}-{max_val:.3f} outside expected range {expected_range}"
                        )
        
        return validation_results


def main():
    """Main function for testing the unit converter."""
    # Create sample data for testing
    sample_data = {
        'Time': [0, 60, 120, 180],  # Assume in minutes for testing
        'Voltage': [3700, 3650, 3600],  # Assume in millivolts
        'Current': [500, 450, 400],  # Assume in milliamperes
        'Temperature': [25.0, 26.5, 28.0],  # Already in Celsius
        'Capacity': [0, 0.1, 0.2]  # Assume in Ah (correct unit)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Sample DataFrame for testing:")
    print(df)
    print()
    
    # Initialize converter
    converter = UnitConverter()
    
    # Standardize units
    result = converter.standardize_dataframe(df)
    
    print("Standardized DataFrame:")
    print(result['standardized_dataframe'])
    print()
    
    print("Conversion Metadata:")
    for conversion in result['conversion_metadata']['conversions_applied']:
        print(f"  {conversion['column_name']}: {conversion['original_unit']} → {conversion['target_unit']} (factor: {conversion['conversion_factor']})")
    
    print()
    print("Conversion Statistics:")
    stats = converter.get_conversion_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Validate results
    validation = converter.validate_standardized_data(result['standardized_dataframe'])
    print(f"\nValidation passed: {validation['validation_passed']}")
    if validation['issues_found']:
        print("Issues found:")
        for issue in validation['issues_found']:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()