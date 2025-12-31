"""
Battery Analytics Lab - Data Ingestion Module
Phase 1: Data Ingestion & Standardization

This module handles the ingestion of raw battery data files and orchestrates
the standardization and validation pipeline.

Author: Battery Analytics Lab Team
Date: 2025-12-29
Version: 1.0
"""

import os
import sys
import yaml
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import traceback

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from standardization.data_standardizer import DataStandardizer
from validation.data_validator import DataValidator

class DataIngestor:
    """
    Main class for orchestrating the data ingestion pipeline:
    Raw Data → Standardization → Validation → Routing
    """
    
    def __init__(self, 
                 config_path: str = "battery-analytics-lab/config/feature_schema.yaml",
                 source_data_dir: str = "/home/sanket/Make_My_route/DATA/CS2_35"):
        """
        Initialize the DataIngestor with configuration and source directories.
        
        Args:
            config_path: Path to the feature schema configuration file
            source_data_dir: Directory containing raw data files
        """
        self.config_path = config_path
        self.source_data_dir = Path(source_data_dir)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize pipeline components
        self.standardizer = DataStandardizer(config_path)
        self.validator = DataValidator(config_path)
        
        # Processing statistics
        self.processing_stats = {
            'total_files_discovered': 0,
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'total_records_processed': 0,
            'processing_errors': 0,
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0
        }
        
        # Processing results storage
        self.processing_results = []
        
        self.logger.info("DataIngestor initialized successfully")
    
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
        """Set up comprehensive logging for ingestion process."""
        logger = logging.getLogger('data_ingestion')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('battery-analytics-lab/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main ingestion log file
        ingestion_handler = logging.FileHandler(log_dir / 'ingestion.log')
        ingestion_handler.setLevel(logging.INFO)
        
        # Error log file
        error_handler = logging.FileHandler(log_dir / 'ingestion_errors.log')
        error_handler.setLevel(logging.ERROR)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ingestion_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(ingestion_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def discover_data_files(self) -> List[Path]:
        """
        Discover all eligible data files in the source directory.
        
        Returns:
            List of discovered data file paths
        """
        self.logger.info(f"Discovering data files in: {self.source_data_dir}")
        
        if not self.source_data_dir.exists():
            self.logger.error(f"Source data directory does not exist: {self.source_data_dir}")
            return []
        
        # Look for Excel files based on configured format
        file_pattern = self.config['raw_data_schema']['file_extension']
        data_files = list(self.source_data_dir.glob(f"*{file_pattern}"))
        
        # Filter for CS2_35 dataset files
        cs2_files = [f for f in data_files if 'CS2_35' in f.name]
        
        self.processing_stats['total_files_discovered'] = len(cs2_files)
        self.logger.info(f"Discovered {len(cs2_files)} CS2_35 data files")
        
        return sorted(cs2_files)
    
    def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single data file through the complete pipeline.
        
        Args:
            file_path: Path to the data file to process
            
        Returns:
            Dictionary containing processing results
        """
        self.logger.info(f"Processing file: {file_path.name}")
        
        processing_result = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'start_time': datetime.now().isoformat(),
            'status': 'pending',
            'standardization_result': None,
            'validation_result': None,
            'error_message': None,
            'processing_duration_seconds': 0
        }
        
        try:
            # Step 1: Standardization
            self.logger.info(f"Starting standardization for {file_path.name}")
            standardization_result = self.standardizer.standardize_excel_file(str(file_path))
            processing_result['standardization_result'] = standardization_result
            
            if standardization_result['status'] != 'success':
                raise Exception(f"Standardization failed: {standardization_result.get('error_message', 'Unknown error')}")
            
            # Step 2: Validation
            self.logger.info(f"Starting validation for {file_path.name}")
            standardized_file_path = standardization_result['output_path']
            validation_result = self.validator.validate_file(standardized_file_path)
            processing_result['validation_result'] = validation_result
            
            # Update processing statistics
            self.processing_stats['files_processed'] += 1
            self.processing_stats['total_records_processed'] += standardization_result.get('records_processed', 0)
            
            if validation_result.get('validation_passed', False):
                self.processing_stats['files_successful'] += 1
                processing_result['status'] = 'success'
                self.logger.info(f"Successfully processed and validated: {file_path.name}")
            else:
                self.processing_stats['files_failed'] += 1
                processing_result['status'] = 'validation_failed'
                self.logger.warning(f"Processing completed but validation failed: {file_path.name}")
            
            # Generate incident report if validation failed
            if not validation_result.get('validation_passed', True):
                incident_report = self.validator.generate_incident_report(validation_result)
                if incident_report:
                    processing_result['incident_report'] = incident_report
            
        except Exception as e:
            self.processing_stats['processing_errors'] += 1
            self.processing_stats['files_failed'] += 1
            processing_result['status'] = 'error'
            processing_result['error_message'] = str(e)
            self.logger.error(f"Error processing file {file_path.name}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            processing_result['end_time'] = datetime.now().isoformat()
            
            # Calculate processing duration
            start_dt = datetime.fromisoformat(processing_result['start_time'])
            end_dt = datetime.fromisoformat(processing_result['end_time'])
            processing_result['processing_duration_seconds'] = (end_dt - start_dt).total_seconds()
        
        return processing_result
    
    def ingest_all_data(self, file_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all discovered data files through the complete pipeline.
        
        Args:
            file_limit: Optional limit on number of files to process (for testing)
            
        Returns:
            Dictionary containing overall processing results and statistics
        """
        self.logger.info("Starting comprehensive data ingestion process")
        self.processing_stats['start_time'] = datetime.now()
        
        # Discover files
        data_files = self.discover_data_files()
        
        if not data_files:
            self.logger.warning("No data files discovered. Aborting ingestion.")
            return self._generate_final_report()
        
        # Apply file limit if specified
        if file_limit:
            data_files = data_files[:file_limit]
            self.logger.info(f"Processing limited to {file_limit} files for testing")
        
        # Process each file
        self.logger.info(f"Processing {len(data_files)} files")
        
        for i, file_path in enumerate(data_files, 1):
            self.logger.info(f"Processing file {i}/{len(data_files)}: {file_path.name}")
            
            try:
                result = self.process_single_file(file_path)
                self.processing_results.append(result)
                
                # Log progress
                if i % 5 == 0 or i == len(data_files):
                    self.logger.info(f"Progress: {i}/{len(data_files)} files processed")
                    self.logger.info(f"Success rate: {self.processing_stats['files_successful']}/{self.processing_stats['files_processed']}")
                
            except Exception as e:
                self.logger.error(f"Unexpected error processing {file_path.name}: {str(e)}")
                continue
        
        # Finalize processing
        self.processing_stats['end_time'] = datetime.now()
        self.processing_stats['duration_seconds'] = (
            self.processing_stats['end_time'] - self.processing_stats['start_time']
        ).total_seconds()
        
        # Generate final report
        final_report = self._generate_final_report()
        
        self.logger.info("Data ingestion process completed")
        return final_report
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final processing report."""
        report = {
            'processing_summary': self.processing_stats.copy(),
            'detailed_results': self.processing_results.copy(),
            'generated_at': datetime.now().isoformat(),
            'config_used': self.config['schema_version']
        }
        
        # Save detailed report to file
        report_file = Path('battery-analytics-lab/logs') / f"ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        try:
            with open(report_file, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, indent=2)
            self.logger.info(f"Detailed report saved to: {report_file}")
        except Exception as e:
            self.logger.error(f"Error saving detailed report: {str(e)}")
        
        # Print summary to console
        self._print_processing_summary()
        
        return report
    
    def _print_processing_summary(self):
        """Print processing summary to console."""
        stats = self.processing_stats
        
        print("\n" + "="*60)
        print("BATTERY ANALYTICS LAB - DATA INGESTION SUMMARY")
        print("="*60)
        print(f"Total files discovered: {stats['total_files_discovered']}")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Files successful: {stats['files_successful']}")
        print(f"Files failed: {stats['files_failed']}")
        print(f"Total records processed: {stats['total_records_processed']:,}")
        print(f"Processing errors: {stats['processing_errors']}")
        print(f"Success rate: {stats['files_successful']/max(stats['files_processed'], 1)*100:.1f}%")
        
        if stats['start_time'] and stats['end_time']:
            print(f"Processing duration: {stats['duration_seconds']:.1f} seconds")
            avg_time_per_file = stats['duration_seconds'] / max(stats['files_processed'], 1)
            print(f"Average time per file: {avg_time_per_file:.1f} seconds")
        
        print("="*60)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def update_metadata_records(self, processing_results: List[Dict[str, Any]]):
        """
        Update metadata records based on processing results.
        
        Args:
            processing_results: List of processing results to update metadata with
        """
        self.logger.info("Updating metadata records")
        
        try:
            # Update cell registry
            self._update_cell_registry(processing_results)
            
            # Update experiment log
            self._update_experiment_log(processing_results)
            
            self.logger.info("Metadata records updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating metadata records: {str(e)}")
    
    def _update_cell_registry(self, processing_results: List[Dict[str, Any]]):
        """Update cell registry with processing information."""
        registry_file = Path('battery-analytics-lab/metadata/cell_registry.csv')
        
        # Create registry if it doesn't exist
        if not registry_file.exists():
            registry_file.parent.mkdir(parents=True, exist_ok=True)
            initial_registry = pd.DataFrame(columns=[
                'cell_id', 'file_source', 'test_date', 'processing_timestamp',
                'standardization_status', 'validation_status', 'quality_score',
                'records_processed', 'data_quality_score'
            ])
            initial_registry.to_csv(registry_file, index=False)
        
        # Load existing registry
        registry = pd.read_csv(registry_file)
        
        # Process results and add new entries
        new_entries = []
        for result in processing_results:
            if result.get('standardization_result') and result['standardization_result'].get('status') == 'success':
                metadata = result['standardization_result']['metadata']
                validation_passed = result.get('validation_result', {}).get('validation_passed', False)
                quality_score = result.get('validation_result', {}).get('quality_score', 0.0)
                
                entry = {
                    'cell_id': metadata.get('cell_id', 'unknown'),
                    'file_source': metadata.get('file_source', 'unknown'),
                    'test_date': metadata.get('test_date', 'unknown'),
                    'processing_timestamp': metadata.get('processing_timestamp', datetime.now().isoformat()),
                    'standardization_status': 'success',
                    'validation_status': 'passed' if validation_passed else 'failed',
                    'quality_score': quality_score,
                    'records_processed': result['standardization_result'].get('records_processed', 0),
                    'data_quality_score': metadata.get('data_quality_score', 0.0)
                }
                new_entries.append(entry)
        
        # Add new entries to registry
        if new_entries:
            new_registry = pd.DataFrame(new_entries)
            updated_registry = pd.concat([registry, new_registry], ignore_index=True)
            updated_registry.to_csv(registry_file, index=False)
            self.logger.info(f"Updated cell registry with {len(new_entries)} new entries")
    
    def _update_experiment_log(self, processing_results: List[Dict[str, Any]]):
        """Update experiment log with processing summary."""
        log_file = Path('battery-analytics-lab/metadata/experiment_log.csv')
        
        # Create log if it doesn't exist
        if not log_file.exists():
            log_file.parent.mkdir(parents=True, exist_ok=True)
            initial_log = pd.DataFrame(columns=[
                'processing_id', 'timestamp', 'operation_type', 'files_processed',
                'records_ingested', 'validation_passed', 'validation_failed',
                'processing_status', 'notes'
            ])
            initial_log.to_csv(log_file, index=False)
        
        # Load existing log
        experiment_log = pd.read_csv(log_file)
        
        # Calculate summary statistics
        total_files = len(processing_results)
        successful_files = sum(1 for r in processing_results if r.get('status') == 'success')
        failed_files = total_files - successful_files
        total_records = sum(r.get('standardization_result', {}).get('records_processed', 0) for r in processing_results)
        
        # Create new log entry
        log_entry = {
            'processing_id': f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'operation_type': 'phase1_ingestion_standardization',
            'files_processed': total_files,
            'records_ingested': total_records,
            'validation_passed': successful_files,
            'validation_failed': failed_files,
            'processing_status': 'completed' if total_files > 0 else 'no_files',
            'notes': f"Processed CS2_35 dataset with {successful_files}/{total_files} files passing validation"
        }
        
        # Add new entry to log
        new_log_entry = pd.DataFrame([log_entry])
        updated_log = pd.concat([experiment_log, new_log_entry], ignore_index=True)
        updated_log.to_csv(log_file, index=False)
        
        self.logger.info("Updated experiment log with processing summary")


def main():
    """Main function for testing the ingestion module."""
    # Initialize ingestor
    ingestor = DataIngestor()
    
    # Process all data files
    results = ingestor.ingest_all_data(file_limit=3)  # Limit to 3 files for testing
    
    # Update metadata
    if results['detailed_results']:
        ingestor.update_metadata_records(results['detailed_results'])
    
    print(f"Final results: {results['processing_summary']}")


if __name__ == "__main__":
    main()