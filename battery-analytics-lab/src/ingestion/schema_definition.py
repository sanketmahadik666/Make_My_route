"""
Battery Analytics Lab - Schema Definition for Controlled Ingestion
Step 2: Controlled Ingestion (Complete CS2_35 Field Coverage)

This module defines the schema for controlled data loading with ALL CS2_35 fields.

Author: Battery Analytics Lab Team
Date: 2025-12-29
Version: 2.0 - Complete CS2_35 Coverage
"""

from typing import List, Dict, Any, Optional
from pathlib import Path


class SchemaDefinition:
    """
    Defines the schema for controlled data ingestion with ALL CS2_35 fields.
    
    Requirements:
    - Load ALL fields from CS2_35 dataset (not just basic fields)
    - Load only sheet index = 1 (second sheet)
    - Load only declared schema columns
    - Preserve original column names
    - No unit conversion or value correction
    """
    
    def __init__(self):
        """Initialize schema definition with complete CS2_35 dataset specifications."""
        self.schema_version = "2.0"
        self.dataset_name = "CS2_35"
        self.allowed_sheet_index = 1  # Second sheet only
        
        # ALL CS2_35 columns as specified by user
        self.expected_columns = [
            'Data_Point',
            'Test_Time',
            'Date_Time',
            'Step_Time',
            'Step_Index',
            'Cycle_Index',
            'Current',
            'Voltage',
            'Charge_Capacity',
            'Discharge_Capacity',
            'Charge_Energy',
            'Discharge_Energy',
            'dV/dt',
            'Internal_Resistance',
            'Is_FC_Data',
            'AC_Impedance',
            'ACI_Phase_Angle'
        ]
        
        # Extended column mapping for various naming conventions
        self.column_mapping = {
            # Time columns
            'data_point': 'Data_Point',
            'test_time': 'Test_Time',
            'datetime': 'Date_Time',
            'date_time': 'Date_Time',
            'step_time': 'Step_Time',
            'step_index': 'Step_Index',
            'cycle_index': 'Cycle_Index',
            
            # Electrical measurements
            'current': 'Current',
            'voltage': 'Voltage',
            
            # Capacity measurements
            'charge_capacity': 'Charge_Capacity',
            'discharge_capacity': 'Discharge_Capacity',
            'charge_energy': 'Charge_Energy',
            'discharge_energy': 'Discharge_Energy',
            
            # Derived measurements
            'dV/dt': 'dV/dt',
            'dvdt': 'dV/dt',
            'internal_resistance': 'Internal_Resistance',
            'internalresistance': 'Internal_Resistance',
            
            # AC impedance measurements
            'is_fc_data': 'Is_FC_Data',
            'ac_impedance': 'AC_Impedance',
            'acimpedance': 'AC_Impedance',
            'aci_phase_angle': 'ACI_Phase_Angle',
            'aciphaseangle': 'ACI_Phase_Angle',
            
            # Legacy column names
            'time': 'Test_Time',  # Legacy alias
            'temperature': 'Temperature',  # May be present in some files
            'capacity': 'Capacity',  # Legacy alias
            
            # Unit-suffixed columns (from CSVs)
            'test_time(s)': 'Test_Time',
            'step_time(s)': 'Step_Time',
            'current(a)': 'Current',
            'voltage(v)': 'Voltage',
            'charge_capacity(ah)': 'Charge_Capacity',
            'discharge_capacity(ah)': 'Discharge_Capacity',
            'charge_energy(wh)': 'Charge_Energy',
            'discharge_energy(wh)': 'Discharge_Energy',
            'dv/dt(v/s)': 'dV/dt',
            'internal_resistance(ohm)': 'Internal_Resistance',
            'ac_impedance(ohm)': 'AC_Impedance',
            'aci_phase_angle(deg)': 'ACI_Phase_Angle'
        }
        
        # File patterns to match
        self.file_patterns = [
            "CS2_35_*.xlsx",
            "CS2_35_*.csv"
        ]
        
        # Data source directory
        self.source_directory = "/home/sanket/Make_My_route/DATA/CS2_35"
        
        # Processing constraints
        self.constraints = {
            'sheet_index': 1,  # Second sheet only
            'preserve_column_names': True,
            'no_unit_conversion': True,
            'no_value_correction': True,
            'load_all_fields': True,  # Load ALL fields, not just basic ones
            'output_format': 'pandas_dataframe'  # In memory only
        }
    
    def get_expected_columns(self) -> List[str]:
        """Get list of ALL expected column names."""
        return self.expected_columns.copy()
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Get column name mapping for normalization."""
        return self.column_mapping.copy()
    
    def get_file_patterns(self) -> List[str]:
        """Get file patterns to match."""
        return self.file_patterns.copy()
    
    def get_source_directory(self) -> str:
        """Get source data directory."""
        return self.source_directory
    
    def get_constraints(self) -> Dict[str, Any]:
        """Get processing constraints."""
        return self.constraints.copy()
    
    def get_all_field_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get complete field definitions for all CS2_35 columns."""
        return {
            'Data_Point': {
                'type': 'integer',
                'unit': 'count',
                'description': 'Sequential data point identifier',
                'nullable': False
            },
            'Test_Time': {
                'type': 'float64',
                'unit': 'seconds',
                'description': 'Elapsed time since test start',
                'nullable': False
            },
            'Date_Time': {
                'type': 'datetime64',
                'unit': 'datetime',
                'description': 'Absolute timestamp of measurement',
                'nullable': False
            },
            'Step_Time': {
                'type': 'float64',
                'unit': 'seconds',
                'description': 'Elapsed time within current test step',
                'nullable': False
            },
            'Step_Index': {
                'type': 'integer',
                'unit': 'count',
                'description': 'Sequential step identifier',
                'nullable': False
            },
            'Cycle_Index': {
                'type': 'integer',
                'unit': 'count',
                'description': 'Cycle number for cycling tests',
                'nullable': True
            },
            'Current': {
                'type': 'float64',
                'unit': 'amperes',
                'description': 'Current flowing through battery cell',
                'nullable': False
            },
            'Voltage': {
                'type': 'float64',
                'unit': 'volts',
                'description': 'Terminal voltage of battery cell',
                'nullable': False
            },
            'Charge_Capacity': {
                'type': 'float64',
                'unit': 'ampere-hours',
                'description': 'Cumulative charge capacity during charging',
                'nullable': True
            },
            'Discharge_Capacity': {
                'type': 'float64',
                'unit': 'ampere-hours',
                'description': 'Cumulative discharge capacity',
                'nullable': True
            },
            'Charge_Energy': {
                'type': 'float64',
                'unit': 'watt-hours',
                'description': 'Cumulative energy during charging',
                'nullable': True
            },
            'Discharge_Energy': {
                'type': 'float64',
                'unit': 'watt-hours',
                'description': 'Cumulative energy during discharge',
                'nullable': True
            },
            'dV/dt': {
                'type': 'float64',
                'unit': 'volts_per_second',
                'description': 'Rate of voltage change',
                'nullable': True
            },
            'Internal_Resistance': {
                'type': 'float64',
                'unit': 'ohms',
                'description': 'Internal resistance of battery cell',
                'nullable': True
            },
            'Is_FC_Data': {
                'type': 'boolean',
                'unit': 'boolean',
                'description': 'Flag for fractional capacity data',
                'nullable': False
            },
            'AC_Impedance': {
                'type': 'float64',
                'unit': 'ohms',
                'description': 'AC impedance magnitude',
                'nullable': True
            },
            'ACI_Phase_Angle': {
                'type': 'float64',
                'unit': 'degrees',
                'description': 'Phase angle of AC impedance',
                'nullable': True
            }
        }
    
    def validate_file_compatibility(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate if a file is compatible with this schema.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'compatible': False,
            'file_name': file_path.name,
            'issues': [],
            'warnings': []
        }
        
        # Check file extension
        if not file_path.suffix.lower() in ['.xlsx', '.csv']:
            validation_result['issues'].append(f"File extension {file_path.suffix} not supported")
            return validation_result
        
        # Check file name pattern
        file_name = file_path.name
        pattern_matched = any(
            file_name.startswith(pattern.replace('*', '')) 
            for pattern in self.file_patterns
        )
        
        if not pattern_matched:
            validation_result['issues'].append(f"File name pattern doesn't match expected format")
            return validation_result
        
        validation_result['compatible'] = True
        return validation_result
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get complete schema information."""
        return {
            'schema_version': self.schema_version,
            'dataset_name': self.dataset_name,
            'total_expected_columns': len(self.expected_columns),
            'expected_columns': self.get_expected_columns(),
            'column_mapping': self.get_column_mapping(),
            'file_patterns': self.get_file_patterns(),
            'source_directory': self.get_source_directory(),
            'constraints': self.get_constraints(),
            'allowed_sheet_index': self.allowed_sheet_index,
            'field_definitions': self.get_all_field_definitions(),
            'note': 'This schema loads ALL fields from CS2_35 dataset, not just basic fields'
        }
    
    def check_column_coverage(self, available_columns: List[str]) -> Dict[str, Any]:
        """
        Check which expected columns are present in available columns.
        
        Args:
            available_columns: List of columns found in data file
            
        Returns:
            Dictionary with coverage analysis
        """
        coverage_analysis = {
            'total_expected': len(self.expected_columns),
            'available_columns': len(available_columns),
            'matched_columns': [],
            'missing_columns': [],
            'coverage_percentage': 0.0
        }
        
        available_lower = [col.lower() for col in available_columns]
        
        for expected_col in self.expected_columns:
            expected_lower = expected_col.lower()
            
            # Check for exact match
            if expected_col in available_columns:
                coverage_analysis['matched_columns'].append(expected_col)
            else:
                # Check for mapped match
                mapped_found = False
                for available_col in available_columns:
                    available_lower_col = available_col.lower()
                    
                    # Check direct mapping
                    if expected_lower in self.column_mapping:
                        if self.column_mapping[expected_lower].lower() == available_lower_col:
                            coverage_analysis['matched_columns'].append(expected_col)
                            mapped_found = True
                            break
                    
                    # Check partial matches
                    if expected_lower in available_lower_col or available_lower_col in expected_lower:
                        coverage_analysis['matched_columns'].append(expected_col)
                        mapped_found = True
                        break
                
                if not mapped_found:
                    coverage_analysis['missing_columns'].append(expected_col)
        
        coverage_analysis['coverage_percentage'] = (
            len(coverage_analysis['matched_columns']) / coverage_analysis['total_expected'] * 100
        )
        
        return coverage_analysis


def main():
    """Main function for testing schema definition."""
    schema = SchemaDefinition()
    
    print("Complete CS2_35 Schema Definition Test")
    print("=" * 50)
    print(f"Dataset: {schema.dataset_name}")
    print(f"Schema Version: {schema.schema_version}")
    print(f"Total Expected Columns: {len(schema.get_expected_columns())}")
    print()
    print("Expected Columns (ALL fields from CS2_35):")
    for i, col in enumerate(schema.get_expected_columns(), 1):
        print(f"  {i:2d}. {col}")
    print()
    print(f"Source Directory: {schema.get_source_directory()}")
    print(f"Constraints: {schema.get_constraints()}")
    
    # Test file validation
    test_file = Path("/home/sanket/Make_My_route/DATA/CS2_35/CS2_35_1_10_11.xlsx")
    if test_file.exists():
        validation = schema.validate_file_compatibility(test_file)
        print(f"\nFile validation for {test_file.name}:")
        print(f"Compatible: {validation['compatible']}")
        if validation['issues']:
            print(f"Issues: {validation['issues']}")
    else:
        print(f"\nTest file not found: {test_file}")


if __name__ == "__main__":
    main()