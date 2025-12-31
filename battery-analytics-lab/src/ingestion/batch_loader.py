"""
Battery Analytics Lab - Batch Loader for Controlled Ingestion
Step 2: Controlled Ingestion (Second Sheet Only)

This module orchestrates the controlled batch loading of Excel files.

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

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from .schema_definition import SchemaDefinition
from .load_excel import ExcelLoader


class BatchLoader:
    """
    Batch loader for controlled Excel file processing.
    Orchestrates file discovery, validation, and loading with minimal processing.
    """
    
    def __init__(self, schema: Optional[SchemaDefinition] = None):
        """
        Initialize the batch loader.
        
        Args:
            schema: SchemaDefinition instance for validation
        """
        self.schema = schema or SchemaDefinition()
        self.loader = ExcelLoader(self.schema)
        self.logger = self._setup_logging()
        
        # Batch processing statistics
        self.batch_stats = {
            'batch_id': None,
            'start_time': None,
            'end_time': None,
            'total_files_discovered': 0,
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'total_rows_loaded': 0,
            'average_processing_time': 0
        }
        
        self.logger.info("BatchLoader initialized for controlled ingestion")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the batch loader."""
        logger = logging.getLogger('controlled_batch_loader')
        logger.setLevel(logging.INFO)
        
        # Use the same handlers as ExcelLoader
        log_dir = Path('battery-analytics-lab/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / 'controlled_batch.log')
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
    
    def discover_files(self) -> List[Path]:
        """
        Discover files matching the schema patterns in the source directory.
        
        Returns:
            List of discovered file paths
        """
        source_dir = Path(self.schema.get_source_directory())
        self.logger.info(f"Discovering files in: {source_dir}")
        
        if not source_dir.exists():
            self.logger.error(f"Source directory does not exist: {source_dir}")
            return []
        
        # Get file patterns from schema
        patterns = self.schema.get_file_patterns()
        discovered_files = []
        
        for pattern in patterns:
            files = list(source_dir.glob(pattern))
            discovered_files.extend(files)
        
        # Remove duplicates and sort
        discovered_files = sorted(list(set(discovered_files)))
        
        self.batch_stats['total_files_discovered'] = len(discovered_files)
        self.logger.info(f"Discovered {len(discovered_files)} files matching schema patterns")
        
        return discovered_files
    
    def run_controlled_batch(self, 
                           file_limit: Optional[int] = None,
                           output_dataframes: bool = True) -> Dict[str, Any]:
        """
        Run the controlled batch ingestion process.
        
        Args:
            file_limit: Optional limit on number of files to process
            output_dataframes: Whether to return DataFrames in results
            
        Returns:
            Dictionary containing batch results and optionally DataFrames
        """
        self.logger.info("Starting controlled batch ingestion process")
        
        # Generate batch ID
        batch_id = f"controlled_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.batch_stats['batch_id'] = batch_id
        self.batch_stats['start_time'] = datetime.now()
        
        # Discover files
        discovered_files = self.discover_files()
        
        if not discovered_files:
            self.logger.warning("No files discovered for controlled ingestion")
            return self._generate_batch_result([], output_dataframes)
        
        # Apply file limit if specified
        if file_limit:
            discovered_files = discovered_files[:file_limit]
            self.logger.info(f"Processing limited to {file_limit} files for testing")
        
        self.batch_stats['files_processed'] = len(discovered_files)
        
        # Process files using the ExcelLoader
        self.logger.info(f"Processing {len(discovered_files)} files with controlled ingestion")
        batch_results = self.loader.load_multiple_files(discovered_files)
        
        # Update batch statistics
        self.batch_stats['end_time'] = datetime.now()
        self.batch_stats['files_successful'] = batch_results['batch_summary']['successful_loads']
        self.batch_stats['files_failed'] = batch_results['batch_summary']['failed_loads']
        self.batch_stats['total_rows_loaded'] = batch_results['batch_summary']['total_rows_loaded']
        
        # Calculate average processing time
        total_time = 0
        for result in batch_results['individual_results']:
            total_time += result.get('processing_time_seconds', 0)
        
        self.batch_stats['average_processing_time'] = (
            total_time / len(batch_results['individual_results']) 
            if batch_results['individual_results'] else 0
        )
        
        # Generate final results
        final_results = self._generate_batch_result(
            batch_results['individual_results'], 
            output_dataframes
        )
        
        self.logger.info(f"Controlled batch ingestion completed: {batch_id}")
        return final_results
    
    def _generate_batch_result(self, 
                             individual_results: List[Dict[str, Any]], 
                             include_dataframes: bool) -> Dict[str, Any]:
        """
        Generate the final batch result dictionary.
        
        Args:
            individual_results: List of individual file load results
            include_dataframes: Whether to include DataFrames in output
            
        Returns:
            Complete batch result dictionary
        """
        result = {
            'batch_info': {
                'batch_id': self.batch_stats['batch_id'],
                'start_time': self.batch_stats['start_time'].isoformat() if self.batch_stats['start_time'] else None,
                'end_time': self.batch_stats['end_time'].isoformat() if self.batch_stats['end_time'] else None,
                'schema_version': self.schema.schema_version,
                'dataset_name': self.schema.dataset_name,
                'processing_constraints': self.schema.get_constraints()
            },
            'summary_statistics': {
                'total_files_discovered': self.batch_stats['total_files_discovered'],
                'files_processed': self.batch_stats['files_processed'],
                'files_successful': self.batch_stats['files_successful'],
                'files_failed': self.batch_stats['files_failed'],
                'success_rate_percent': (
                    self.batch_stats['files_successful'] / max(self.batch_stats['files_processed'], 1) * 100
                ),
                'total_rows_loaded': self.batch_stats['total_rows_loaded'],
                'average_processing_time_seconds': self.batch_stats['average_processing_time']
            },
            'individual_results': individual_results,
            'load_statistics': self.loader.get_load_statistics()
        }
        
        # Include DataFrames if requested
        if include_dataframes:
            result['dataframes'] = self.loader.get_dataframes_from_results({
                'individual_results': individual_results
            })
        
        # Log final summary
        self._log_batch_summary(result)
        
        return result
    
    def _log_batch_summary(self, result: Dict[str, Any]):
        """Log a summary of the batch processing results."""
        summary = result['summary_statistics']
        
        self.logger.info("CONTROLLED BATCH INGESTION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Batch ID: {result['batch_info']['batch_id']}")
        self.logger.info(f"Files discovered: {summary['total_files_discovered']}")
        self.logger.info(f"Files processed: {summary['files_processed']}")
        self.logger.info(f"Files successful: {summary['files_successful']}")
        self.logger.info(f"Files failed: {summary['files_failed']}")
        self.logger.info(f"Success rate: {summary['success_rate_percent']:.1f}%")
        self.logger.info(f"Total rows loaded: {summary['total_rows_loaded']:,}")
        self.logger.info(f"Average processing time: {summary['average_processing_time_seconds']:.2f} seconds")
        self.logger.info("=" * 50)
        
        # Log individual file results
        for file_result in result['individual_results']:
            status = "✓ SUCCESS" if file_result['load_status'] == 'success' else "✗ FAILED"
            rows = file_result.get('rows_loaded', 0)
            self.logger.info(f"{status} {file_result['file_name']}: {rows} rows")
            
            if file_result['load_status'] == 'failed':
                self.logger.error(f"  Error: {file_result.get('error_message', 'Unknown error')}")
    
    def print_batch_summary(self, result: Dict[str, Any]):
        """Print a formatted summary of batch results to console."""
        summary = result['summary_statistics']
        
        print("\n" + "=" * 60)
        print("BATTERY ANALYTICS LAB - CONTROLLED BATCH INGESTION SUMMARY")
        print("=" * 60)
        print(f"Batch ID: {result['batch_info']['batch_id']}")
        print(f"Schema Version: {result['batch_info']['schema_version']}")
        print(f"Dataset: {result['batch_info']['dataset_name']}")
        print()
        print("📊 Processing Statistics:")
        print(f"   Files discovered: {summary['total_files_discovered']}")
        print(f"   Files processed: {summary['files_processed']}")
        print(f"   Files successful: {summary['files_successful']}")
        print(f"   Files failed: {summary['files_failed']}")
        print(f"   Success rate: {summary['success_rate_percent']:.1f}%")
        print(f"   Total rows loaded: {summary['total_rows_loaded']:,}")
        print(f"   Avg processing time: {summary['average_processing_time_seconds']:.2f}s")
        print()
        print("🔧 Processing Constraints:")
        constraints = result['batch_info']['processing_constraints']
        print(f"   Sheet index: {constraints['sheet_index']} (second sheet only)")
        print(f"   Preserve column names: {constraints['preserve_column_names']}")
        print(f"   No unit conversion: {constraints['no_unit_conversion']}")
        print(f"   No value correction: {constraints['no_value_correction']}")
        print(f"   Output format: {constraints['output_format']}")
        print("=" * 60)
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get current batch processing statistics."""
        return self.batch_stats.copy()


def main():
    """Main function for testing the batch loader."""
    # Initialize batch loader
    batch_loader = BatchLoader()
    
    # Run controlled batch with test limit
    print("Running controlled batch ingestion test...")
    results = batch_loader.run_controlled_batch(file_limit=3, output_dataframes=True)
    
    # Print summary
    batch_loader.print_batch_summary(results)
    
    # Show DataFrame information
    if 'dataframes' in results:
        print(f"\n📋 Loaded DataFrames: {len(results['dataframes'])}")
        for name, df in results['dataframes'].items():
            print(f"   {name}: {df.shape} - Columns: {list(df.columns)}")
            
            # Show basic info about first DataFrame
            if len(results['dataframes']) == 1:
                print(f"\n📊 Sample Data (first 5 rows):")
                print(df.head())
                print(f"\n📈 Data Types:")
                print(df.dtypes)


if __name__ == "__main__":
    main()