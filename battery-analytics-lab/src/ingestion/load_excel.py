"""
Battery Analytics Lab - Excel Loader for Controlled Ingestion
Step 2: Controlled Ingestion (Second Sheet Only)

This module handles loading Excel files with controlled, minimal processing.

Requirements:
- Load only sheet index = 1 (second sheet)
- Load only declared schema columns
- Preserve original column names
- No unit conversion or value correction
- Log load success or failure per file
- Output: Pandas DataFrames per file (in memory only)

Author: Battery Analytics Lab Team
Date: 2025-12-29
Version: 1.0
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import traceback

from .schema_definition import SchemaDefinition


class ExcelLoader:
    """
    Controlled Excel loader with minimal processing.
    Loads only the second sheet (index = 1) with schema-defined columns.
    """
    
    def __init__(self, schema: Optional[SchemaDefinition] = None):
        """
        Initialize the Excel loader.
        
        Args:
            schema: SchemaDefinition instance for column validation
        """
        self.schema = schema or SchemaDefinition()
        self.logger = self._setup_logging()
        self.load_stats = {
            'files_attempted': 0,
            'files_successful': 0,
            'files_failed': 0,
            'total_rows_loaded': 0,
            'load_errors': []
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the Excel loader."""
        logger = logging.getLogger('controlled_excel_loader')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('battery-analytics-lab/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler for controlled ingestion logs
        file_handler = logging.FileHandler(log_dir / 'controlled_ingestion.log')
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
    
    def load_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a single Excel file with controlled processing.
        
        Args:
            file_path: Path to the Excel file to load
            
        Returns:
            Dictionary containing load results and DataFrame
        """
        self.logger.info(f"Starting controlled load of file: {file_path.name}")
        self.load_stats['files_attempted'] += 1
        
        load_result = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'load_status': 'pending',
            'dataframe': None,
            'rows_loaded': 0,
            'columns_loaded': [],
            'sheet_index_used': 1,
            'error_message': None,
            'load_timestamp': datetime.now().isoformat(),
            'processing_time_seconds': 0
        }
        
        start_time = datetime.now()
        
        try:
            # Validate file compatibility
            validation = self.schema.validate_file_compatibility(file_path)
            if not validation['compatible']:
                raise ValueError(f"File not compatible: {validation['issues']}")
            
            # Load Excel file - sheet index = 1 (second sheet) only
            self.logger.info(f"Loading sheet index 1 from {file_path.name}")
            df = pd.read_excel(file_path, sheet_name=1, engine='openpyxl')
            
            # Apply schema column filtering
            df_filtered = self._apply_schema_columns(df)
            
            # Store results
            load_result['dataframe'] = df_filtered
            load_result['rows_loaded'] = len(df_filtered)
            load_result['columns_loaded'] = list(df_filtered.columns)
            load_result['load_status'] = 'success'
            load_result['total_sheets_in_file'] = len(pd.ExcelFile(file_path).sheet_names)
            
            # Update statistics
            self.load_stats['files_successful'] += 1
            self.load_stats['total_rows_loaded'] += len(df_filtered)
            
            self.logger.info(f"Successfully loaded {len(df_filtered)} rows from {file_path.name}")
            self.logger.info(f"Columns loaded: {list(df_filtered.columns)}")
            
        except Exception as e:
            self.load_stats['files_failed'] += 1
            self.load_stats['load_errors'].append(f"{file_path.name}: {str(e)}")
            load_result['load_status'] = 'failed'
            load_result['error_message'] = str(e)
            load_result['dataframe'] = None
            
            self.logger.error(f"Failed to load {file_path.name}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            # Calculate processing time
            end_time = datetime.now()
            load_result['processing_time_seconds'] = (end_time - start_time).total_seconds()
        
        return load_result
    
    def _apply_schema_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply schema column filtering while preserving original column names.
        
        Args:
            df: Original DataFrame from Excel
            
        Returns:
            Filtered DataFrame with only schema-defined columns
        """
        # Get expected columns from schema
        expected_columns = self.schema.get_expected_columns()
        
        # Find matching columns (case-insensitive)
        available_columns = list(df.columns)
        matching_columns = []
        
        for expected_col in expected_columns:
            # Try exact match first
            if expected_col in available_columns:
                matching_columns.append(expected_col)
            else:
                # Try case-insensitive and partial matching
                for avail_col in available_columns:
                    if (expected_col.lower() in avail_col.lower() or 
                        avail_col.lower() in expected_col.lower()):
                        matching_columns.append(avail_col)
                        break
        
        if matching_columns:
            # Return DataFrame with only matching columns, preserving original names
            filtered_df = df[matching_columns].copy()
            self.logger.info(f"Filtered columns: {matching_columns}")
        else:
            self.logger.warning(f"No matching columns found for schema: {expected_columns}")
            self.logger.warning(f"Available columns: {available_columns}")
            # Return empty DataFrame with expected column structure
            filtered_df = pd.DataFrame(columns=expected_columns)
        
        return filtered_df
    
    def load_multiple_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Load multiple Excel files with controlled processing.
        
        Args:
            file_paths: List of paths to Excel files
            
        Returns:
            Dictionary containing results for all files
        """
        self.logger.info(f"Starting controlled batch load of {len(file_paths)} files")
        
        results = []
        successful_loads = 0
        failed_loads = 0
        
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(f"Processing file {i}/{len(file_paths)}: {file_path.name}")
            
            result = self.load_single_file(file_path)
            results.append(result)
            
            if result['load_status'] == 'success':
                successful_loads += 1
            else:
                failed_loads += 1
            
            # Log progress
            if i % 5 == 0 or i == len(file_paths):
                self.logger.info(f"Progress: {i}/{len(file_paths)} files processed")
        
        # Compile batch results
        batch_result = {
            'batch_summary': {
                'total_files': len(file_paths),
                'successful_loads': successful_loads,
                'failed_loads': failed_loads,
                'success_rate': successful_loads / len(file_paths) * 100 if file_paths else 0,
                'total_rows_loaded': self.load_stats['total_rows_loaded'],
                'batch_start_time': datetime.now().isoformat(),
                'batch_end_time': datetime.now().isoformat()
            },
            'individual_results': results,
            'load_statistics': self.load_stats.copy()
        }
        
        self.logger.info(f"Batch load completed: {successful_loads}/{len(file_paths)} files successful")
        return batch_result
    
    def get_dataframes_from_results(self, batch_result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Extract DataFrames from batch results for further processing.
        
        Args:
            batch_result: Results from load_multiple_files
            
        Returns:
            Dictionary mapping file names to DataFrames
        """
        dataframes = {}
        
        for result in batch_result['individual_results']:
            if result['load_status'] == 'success' and result['dataframe'] is not None:
                file_name = result['file_name'].replace('.xlsx', '')
                dataframes[file_name] = result['dataframe']
        
        return dataframes
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get current load statistics."""
        return self.load_stats.copy()


def main():
    """Main function for testing the Excel loader."""
    # Initialize components
    schema = SchemaDefinition()
    loader = ExcelLoader(schema)
    
    # Test with sample files
    source_dir = Path("/home/sanket/Make_My_route/DATA/CS2_35")
    
    if source_dir.exists():
        # Find Excel files
        excel_files = list(source_dir.glob("CS2_35_*.xlsx"))
        
        if excel_files:
            print(f"Found {len(excel_files)} Excel files to test")
            
            # Test loading first 3 files
            test_files = excel_files[:3]
            results = loader.load_multiple_files(test_files)
            
            print(f"\nLoad Results:")
            print(f"Successful: {results['batch_summary']['successful_loads']}")
            print(f"Failed: {results['batch_summary']['failed_loads']}")
            print(f"Total rows: {results['batch_summary']['total_rows_loaded']}")
            
            # Extract DataFrames
            dataframes = loader.get_dataframes_from_results(results)
            print(f"\nDataFrames extracted: {len(dataframes)}")
            
            for name, df in dataframes.items():
                print(f"  {name}: {df.shape} - {list(df.columns)}")
        
        else:
            print(f"No Excel files found in {source_dir}")
    else:
        print(f"Source directory not found: {source_dir}")


if __name__ == "__main__":
    main()