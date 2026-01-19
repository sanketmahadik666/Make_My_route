"""
Battery Analytics Lab - Data Standardization Module
Phase 1: Data Ingestion & Standardization

This module handles the standardization of raw battery data into a consistent format
for downstream analysis.

Author: Battery Analytics Lab Team
Date: 2025-12-29
Version: 1.0
"""

import pandas as pd
import numpy as np
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ingestion.schema_definition import SchemaDefinition

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataStandardizer:
    """
    Main class for standardizing battery data from raw format to standardized format.
    """
    
    def __init__(self, config_path: str = "config/feature_schema.yaml"):
        """
        Initialize the DataStandardizer with configuration.
        
        Args:
            config_path: Path to the feature schema configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.processing_stats = {
            'files_processed': 0,
            'total_records': 0,
            'standardization_errors': 0,
            'warnings_issued': 0
        }
        self.schema = SchemaDefinition()
    
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
        """Set up logging for standardization process."""
        logger = logging.getLogger('data_standardizer')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('battery-analytics-lab/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / 'standardization.log')
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
    
    def _validate_file_extension(self, file_path: str) -> bool:
        """Check if file extension is supported."""
        return Path(file_path).suffix.lower() in ['.xlsx', '.csv']

    def standardize_file(self, 
                             file_path: str, 
                             output_dir: str = "data/standardized/") -> Dict[str, Any]:
        """
        Standardize a file (Excel or CSV) containing battery data.
        
        Args:
            file_path: Path to the source file
            output_dir: Directory to save standardized data
            
        Returns:
            Dictionary containing processing results and metadata
        """
        try:
            self.logger.info(f"Starting standardization of file: {file_path}")
            
            file_path_obj = Path(file_path)
            
            # Read file based on extension
            if file_path_obj.suffix.lower() == '.csv':
                df = self._read_csv_file(file_path)
            elif file_path_obj.suffix.lower() == '.xlsx':
                df = self._read_excel_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, df)
            
            # Standardize column names and data
            standardized_df = self._standardize_data(df)
            
            # Add metadata columns
            standardized_df = self._attach_metadata(standardized_df, metadata)
            
            # Save standardized data
            output_path = self._save_standardized_data(standardized_df, output_dir, metadata)
            
            # Update statistics
            self.processing_stats['files_processed'] += 1
            self.processing_stats['total_records'] += len(standardized_df)
            
            self.logger.info(f"Successfully standardized {len(standardized_df)} records from {file_path}")
            
            return {
                'status': 'success',
                'file_path': file_path,
                'output_path': output_path,
                'records_processed': len(standardized_df),
                'metadata': metadata,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.processing_stats['standardization_errors'] += 1
            self.logger.error(f"Error standardizing file {file_path}: {str(e)}")
            return {
                'status': 'error',
                'file_path': file_path,
                'error_message': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def _read_csv_file(self, file_path: str) -> pd.DataFrame:
        """Read CSV file and return DataFrame."""
        try:
            # Read CSV with pandas, assuming header is on first row
            df = pd.read_csv(file_path)
            self.logger.info(f"Read {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            raise

    def _read_excel_file(self, file_path: str) -> pd.DataFrame:
        """Read Excel file and return DataFrame."""
        try:
            # Try reading different sheet names
            xl_file = pd.ExcelFile(file_path)
            sheet_names = xl_file.sheet_names
            
            # Use first sheet or look for data sheet
            if 'Data' in sheet_names:
                df = pd.read_excel(file_path, sheet_name='Data')
            elif 'Sheet1' in sheet_names:
                df = pd.read_excel(file_path, sheet_name='Sheet1')
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_names[0])
            
            self.logger.info(f"Read {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading Excel file {file_path}: {str(e)}")
            raise
    
    def _extract_metadata(self, file_path: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata from file and data."""
        file_name = Path(file_path).name
        cell_id = self._extract_cell_id(file_name)
        
        # Extract test date from filename if possible
        test_date = self._extract_test_date(file_name)
        
        # Load external JSON metadata
        external_metadata = self._load_json_metadata(file_path)
        
        metadata = {
            'cell_id': cell_id,
            'test_date': test_date,
            'file_source': file_name,
            'standardization_version': self.config.get('schema_version', '1.0'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_columns': list(df.columns),
            'original_row_count': len(df),
            'data_quality_score': self._calculate_initial_quality_score(df),
            'extraction_metadata': external_metadata
        }
        
        return metadata
    
    def _load_json_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load external JSON metadata if available."""
        try:
            file_path_obj = Path(file_path)
            # Logic to find metadata file
            # Assumes data in .../data/raw/calce/FILE.csv
            # Metadata in .../data/raw/calce/metadata/FILE_BASE_extraction_metadata.json
            
            stem = file_path_obj.stem 
            
            # Common suffixes to remove to find the "parent" Excel file name
            suffixes_to_remove = ['_Channel_1-008', '_Info', '_Statistics_1-008']
            base_name = stem
            for suffix in suffixes_to_remove:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            
            metadata_filename = f"{base_name}_extraction_metadata.json"
            # Look in parallel 'metadata' directory or subdirectory
            metadata_dir = file_path_obj.parent / 'metadata'
            metadata_path = metadata_dir / metadata_filename
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.logger.info(f"Loaded external metadata from {metadata_path}")
                    return json.load(f)
            else:
                self.logger.warning(f"External metadata file not found at {metadata_path}")
                return {}
        except Exception as e:
            self.logger.warning(f"Could not load external metadata for {file_path}: {e}")
            return {}
    
    def _extract_cell_id(self, file_name: str) -> str:
        """Extract cell ID from filename."""
        # Extract from CS2_35 format: CS2_35_MM_DD_YY.xlsx
        parts = file_name.replace('.xlsx', '').split('_')
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[1]}"
        return file_name.replace('.xlsx', '').replace('.csv', '')
    
    def _extract_test_date(self, file_name: str) -> Optional[str]:
        """Extract test date from filename."""
        try:
            # Extract date from filename format: CS2_35_MM_DD_YY.xlsx or .csv
            parts = file_name.replace('.xlsx', '').replace('.csv', '').split('_')
            if len(parts) >= 5:
                month, day, year = parts[2], parts[3], parts[4]
                # Convert 2-digit year to 4-digit
                if len(year) == 2:
                    year = f"20{year}"
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except:
            pass
        return None
    
    def _calculate_initial_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate initial data quality score."""
        try:
            # Check for required columns
            required_cols = self.config.get('raw_data_schema', {}).get('required_columns', {})
            available_cols = list(df.columns)
            
            # Count matching columns (case-insensitive)
            matched_cols = 0
            for req_col in required_cols.keys():
                for avail_col in available_cols:
                    if req_col.lower() in avail_col.lower():
                        matched_cols += 1
                        break
            
            column_match_score = matched_cols / len(required_cols)
            
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            completeness_score = 1.0 - missing_ratio
            
            # Combine scores
            quality_score = (column_match_score + completeness_score) / 2
            return min(quality_score, 1.0)
            
        except:
            return 0.0
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types."""
        standardized_df = df.copy()
        
        # Map column names to standardized format
        column_mapping = self._create_column_mapping(standardized_df.columns)
        standardized_df = standardized_df.rename(columns=column_mapping)
        
        # Standardize data types and units
        standardized_df = self._standardize_data_types(standardized_df)
        
        # Add cycle numbering (simplified)
        standardized_df = self._add_cycle_numbering(standardized_df)
        
        # Add phase type classification
        standardized_df = self._classify_phases(standardized_df)
        
        return standardized_df
    
    def _create_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Create mapping from original to standardized column names using SchemaDefinition."""
        mapping = {}
        schema_mapping = self.schema.get_column_mapping()
        
        for col in columns:
            col_lower = col.lower()
            
            # Check mappings from schema definition first
            if col_lower in schema_mapping:
                mapping[col] = schema_mapping[col_lower]
                continue
                
            # Fallback to existing manual rules if not in schema
            
            # Time/timestamp columns
            if any(keyword in col_lower for keyword in ['time', 'timestamp', 'date']):
                mapping[col] = 'timestamp'
            
            # Voltage columns
            elif any(keyword in col_lower for keyword in ['voltage', 'volt', 'v']):
                mapping[col] = 'voltage_v'
            
            # Current columns
            elif any(keyword in col_lower for keyword in ['current', 'amp', 'a']):
                mapping[col] = 'current_a'
            
            # Capacity columns
            elif any(keyword in col_lower for keyword in ['capacity', 'cap', 'ah']):
                mapping[col] = 'capacity_ah'
            
            # Temperature columns
            elif any(keyword in col_lower for keyword in ['temperature', 'temp', 'c']):
                mapping[col] = 'temperature_c'
            
            # Default: keep original name
            else:
                mapping[col] = col
        
        return mapping
    
    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types and apply unit conversions."""
        # Convert numeric columns
        numeric_cols = ['timestamp', 'voltage_v', 'current_a', 'capacity_ah', 'temperature_c']
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    self.logger.warning(f"Could not convert {col} to numeric type")
        
        # Apply unit conversions if needed
        df = self._apply_unit_conversions(df)
        
        return df
    
    def _apply_unit_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply unit conversions to match standardized format."""
        # Convert timestamp to seconds from start
        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            df['timestamp'] = df['timestamp'] - df['timestamp'].min()
        
        # Normalize values to expected ranges
        if 'raw_data_schema' in self.config and 'value_ranges' in self.config['raw_data_schema']:
            ranges = self.config['raw_data_schema']['value_ranges']
            
            for col, range_info in ranges.items():
                if col in df.columns:
                    # Convert to specified units if needed
                    df[col] = self._convert_to_standard_units(df[col], col, range_info)
        
        return df
    
    def _convert_to_standard_units(self, series: pd.Series, column: str, range_info: Dict) -> pd.Series:
        """Convert data to standard units."""
        # This is a simplified implementation
        # In practice, you would implement specific conversion logic
        return series
    
    def _add_cycle_numbering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cycle numbering based on current direction changes."""
        if 'current_a' in df.columns:
            # Detect charge/discharge cycles based on current sign changes
            df['cycle_number'] = 1  # Simplified: assign all data to cycle 1
        else:
            df['cycle_number'] = 1
        
        return df
    
    def _classify_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify data phases (charge/discharge/rest)."""
        if 'current_a' in df.columns:
            # Simplified classification
            df['phase_type'] = 'unknown'
            
            # Positive current = charging
            charge_mask = df['current_a'] > 0.1
            df.loc[charge_mask, 'phase_type'] = 'charge'
            
            # Negative current = discharging
            discharge_mask = df['current_a'] < -0.1
            df.loc[discharge_mask, 'phase_type'] = 'discharge'
            
            # Near zero current = rest
            rest_mask = (df['current_a'] >= -0.1) & (df['current_a'] <= 0.1)
            df.loc[rest_mask, 'phase_type'] = 'rest'
        else:
            df['phase_type'] = 'unknown'
        
        return df
    
    def _attach_metadata(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Attach metadata columns to the DataFrame."""
        for key, value in metadata.items():
            # Convert list/dict to string to avoid length mismatch errors
            if isinstance(value, (list, dict)):
                df[key] = str(value)
            else:
                df[key] = value
        
        return df
    
    def _save_standardized_data(self, df: pd.DataFrame, output_dir: str, metadata: Dict[str, Any]) -> str:
        """Save standardized data to parquet format."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        file_name = metadata['file_source'].replace('.xlsx', '_standardized.parquet').replace('.csv', '_standardized.parquet')
        output_file = output_path / file_name
        
        # Save to parquet
        df.to_parquet(output_file, index=False, compression='snappy')
        
        return str(output_file)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()


    def generate_metadata_index(self, output_dir: str = "data/standardized/") -> str:
        """
        Generate a summary index of all standardized files.
        
        Args:
            output_dir: Directory containing standardized files
            
        Returns:
            Path to the generated index file
        """
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                return ""
            
            metadata_list = []
            
            # Scan for parquet files
            for parquet_file in output_path.glob("*_standardized.parquet"):
                try:
                    # Read metadata from parquet file (requires fastparquet or pyarrow)
                    # For efficiency, we'll read just the metadata if possible, or head
                    # Using pandas read_parquet for simplicity
                    df = pd.read_parquet(parquet_file)
                    
                    # Extract relevant columns - assuming attached metadata columns correspond to our keys
                    file_info = {
                        'file_name': parquet_file.name,
                        'cell_id': df['cell_id'].iloc[0] if 'cell_id' in df.columns else 'unknown',
                        'test_date': df['test_date'].iloc[0] if 'test_date' in df.columns else None,
                        'original_source': df['file_source'].iloc[0] if 'file_source' in df.columns else None,
                        'num_records': len(df),
                        'ingestion_date': datetime.fromtimestamp(parquet_file.stat().st_mtime).isoformat()
                    }
                    metadata_list.append(file_info)
                except Exception as e:
                    self.logger.warning(f"Error reading metadata from {parquet_file}: {e}")
            
            if metadata_list:
                index_df = pd.DataFrame(metadata_list)
                index_path = output_path / "metadata_index.csv"
                index_df.to_csv(index_path, index=False)
                self.logger.info(f"Generated metadata index at {index_path} with {len(index_df)} entries")
                return str(index_path)
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error generating metadata index: {e}")
            return ""


def main():
    """Main function for testing the standardization module."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Standardize battery data files.')
    parser.add_argument('--input', '-i', type=str, help='Input file or directory')
    parser.add_argument('--output', '-o', type=str, default='data/standardized/', help='Output directory')
    
    args = parser.parse_args()
    
    standardizer = DataStandardizer()
    
    if args.input:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Process single file
            if standardizer._validate_file_extension(str(input_path)):
                result = standardizer.standardize_file(str(input_path), args.output)
                print(f"Result: {result['status']}")
            else:
                print(f"Skipping {input_path}: Unsupported extension")
                
        elif input_path.is_dir():
            # Process directory
            print(f"Processing directory: {input_path}")
            # Recursively find .csv and .xlsx files
            files = []
            files.extend(list(input_path.glob('**/*Channel*.csv'))) # Prioritize Channel files as per user request
            files.extend(list(input_path.glob('**/*Channel*.xlsx')))
            
            print(f"Found {len(files)} potential data files.")
            
            for file_path in files:
                try:
                    print(f"Processing {file_path.name}...")
                    standardizer.standardize_file(str(file_path), args.output)
                except Exception as e:
                    print(f"Failed to process {file_path.name}: {e}")
                    
    else:
        # Default test behavior if no args provided
        test_file = "/home/sanket/Make_My_route/battery-analytics-lab/data/raw/calce/CS2_35_1_10_11_Channel_1-008.csv"
        if Path(test_file).exists():
            print(f"Running default test on {test_file}")
            result = standardizer.standardize_file(test_file)
            print(f"Standardization result: {result}")
    
    # Generate index after processing
    index_path = standardizer.generate_metadata_index(args.output if args.input else "data/standardized/")
    print(f"Metadata index generated at: {index_path}")
    print(f"Processing stats: {standardizer.get_processing_stats()}")

    
def _validate_file_extension(self, file_path: str) -> bool:
    """Helper to check extension."""
    return Path(file_path).suffix.lower() in ['.xlsx', '.csv']

# Monkey patch or add method to class if missing, but better to just use inline check in main or add to class above.
# Adding helper to class above main would be better, but for now using inline logic in main is safer if I can't edit class easily in this chunk.
# Actually, I'll just put the logic in main.


if __name__ == "__main__":
    main()