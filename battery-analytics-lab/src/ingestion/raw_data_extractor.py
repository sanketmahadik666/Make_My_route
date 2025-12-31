#!/usr/bin/env python3
"""
Raw Data Extraction Pipeline for CALCE CS2_35 Dataset

This script extracts ALL features/columns from CS2_35 Excel files and saves them
in standardized CSV format for further analysis.

Author: Battery Analytics Lab
Date: 2025-12-29
Version: 1.0
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
import openpyxl
from typing import Dict, List, Tuple, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/raw_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RawDataExtractor:
    """
    Comprehensive raw data extractor for CALCE CS2_35 battery dataset.
    
    This class handles:
    - Scanning all CS2_35 Excel files
    - Extracting ALL columns/features from each file
    - Preserving original data structure
    - Saving to standardized CSV format
    - Maintaining data traceability
    """
    
    def __init__(self, source_dir: str, output_dir: str):
        """
        Initialize the raw data extractor.
        
        Args:
            source_dir: Directory containing CS2_35 Excel files
            output_dir: Directory for extracted CSV files
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'source_directory': str(source_dir),
            'output_directory': str(output_dir),
            'files_processed': [],
            'total_features_extracted': 0,
            'extraction_summary': {}
        }
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata subdirectory
        self.metadata_dir = self.output_dir / 'metadata'
        self.metadata_dir.mkdir(exist_ok=True)
        
        logger.info(f"RawDataExtractor initialized")
        logger.info(f"Source: {source_dir}")
        logger.info(f"Output: {output_dir}")
    
    def discover_excel_files(self) -> List[Path]:
        """
        Discover all CS2_35 Excel files in the source directory.
        
        Returns:
            List of Excel file paths
        """
        excel_files = list(self.source_dir.glob('*.xlsx'))
        excel_files.extend(list(self.source_dir.glob('*.xls')))
        
        # Filter for CS2_35 files
        cs2_35_files = [f for f in excel_files if 'CS2_35' in f.name]
        
        logger.info(f"Discovered {len(cs2_35_files)} CS2_35 Excel files")
        for file in cs2_35_files:
            logger.info(f"  - {file.name}")
        
        return sorted(cs2_35_files)
    
    def extract_excel_structure(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive structure information from Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary containing file structure information
        """
        try:
            # Load workbook to get sheet information
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            sheet_names = workbook.sheetnames
            
            structure_info = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'sheet_names': sheet_names,
                'sheets_info': {}
            }
            
            # Get info for each sheet
            for sheet_name in sheet_names:
                try:
                    # Read sheet with pandas to get structure
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    structure_info['sheets_info'][sheet_name] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': list(df.columns),
                        'data_types': df.dtypes.to_dict(),
                        'memory_usage': df.memory_usage(deep=True).sum(),
                        'null_counts': df.isnull().sum().to_dict()
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not read sheet '{sheet_name}' from {file_path.name}: {e}")
                    structure_info['sheets_info'][sheet_name] = {
                        'error': str(e)
                    }
            
            workbook.close()
            return structure_info
            
        except Exception as e:
            logger.error(f"Error analyzing file structure {file_path.name}: {e}")
            return {'file_name': file_path.name, 'error': str(e)}
    
    def extract_all_sheets_to_csv(self, file_path: Path) -> Dict[str, str]:
        """
        Extract all sheets from Excel file and save as separate CSV files.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary mapping sheet names to CSV file paths
        """
        csv_files = {}
        
        try:
            # Get file structure first
            structure_info = self.extract_excel_structure(file_path)
            
            # Process each sheet
            for sheet_name in structure_info['sheet_names']:
                try:
                    # Read sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    if df.empty:
                        logger.warning(f"Sheet '{sheet_name}' in {file_path.name} is empty")
                        continue
                    
                    # Create safe filename
                    safe_sheet_name = sheet_name.replace(' ', '_').replace('/', '_')
                    csv_filename = f"{file_path.stem}_{safe_sheet_name}.csv"
                    csv_path = self.output_dir / csv_filename
                    
                    # Save to CSV with proper formatting
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    
                    csv_files[sheet_name] = str(csv_path)
                    
                    logger.info(f"Extracted sheet '{sheet_name}' from {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Log column details
                    for col in df.columns:
                        logger.debug(f"  Column: {col} (type: {df[col].dtype})")
                    
                except Exception as e:
                    logger.error(f"Error extracting sheet '{sheet_name}' from {file_path.name}: {e}")
                    continue
            
            return csv_files
            
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            return {}
    
    def create_comprehensive_metadata(self, file_path: Path, csv_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Create comprehensive metadata for extracted files.
        
        Args:
            file_path: Original Excel file path
            csv_files: Dictionary of extracted CSV files
            
        Returns:
            Metadata dictionary
        """
        structure_info = self.extract_excel_structure(file_path)
        
        metadata = {
            'extraction_info': {
                'source_file': file_path.name,
                'source_path': str(file_path),
                'extraction_timestamp': datetime.now().isoformat(),
                'extractor_version': '1.0'
            },
            'original_structure': structure_info,
            'extracted_files': csv_files,
            'feature_summary': {
                'total_sheets_extracted': len(csv_files),
                'total_features_by_sheet': {}
            }
        }
        
        # Analyze features in each extracted CSV
        for sheet_name, csv_path in csv_files.items():
            try:
                df = pd.read_csv(csv_path)
                metadata['feature_summary']['total_features_by_sheet'][sheet_name] = {
                    'columns': len(df.columns),
                    'rows': len(df),
                    'column_names': list(df.columns),
                    'data_types': df.dtypes.to_dict()
                }
            except Exception as e:
                logger.warning(f"Could not analyze extracted CSV {csv_path}: {e}")
                metadata['feature_summary']['total_features_by_sheet'][sheet_name] = {
                    'error': str(e)
                }
        
        return metadata
    
    def extract_all_features(self) -> Dict[str, Any]:
        """
        Extract all features from all CS2_35 files.
        
        Returns:
            Comprehensive extraction summary
        """
        logger.info("Starting comprehensive raw data extraction...")
        
        # Discover files
        excel_files = self.discover_excel_files()
        
        if not excel_files:
            logger.error("No CS2_35 Excel files found!")
            return {'error': 'No files found'}
        
        extraction_summary = {
            'files_discovered': len(excel_files),
            'files_processed': 0,
            'total_csv_files_created': 0,
            'total_features_extracted': 0,
            'extraction_details': {}
        }
        
        # Process each file
        for excel_file in excel_files:
            logger.info(f"Processing file: {excel_file.name}")
            
            try:
                # Extract all sheets to CSV
                csv_files = self.extract_all_sheets_to_csv(excel_file)
                
                if csv_files:
                    # Create metadata for this file
                    metadata = self.create_comprehensive_metadata(excel_file, csv_files)
                    
                    # Save metadata
                    metadata_filename = f"{excel_file.stem}_extraction_metadata.json"
                    metadata_path = self.metadata_dir / metadata_filename
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, default=str)
                    
                    # Update summary
                    file_summary = {
                        'csv_files_created': len(csv_files),
                        'csv_file_paths': list(csv_files.values()),
                        'metadata_file': str(metadata_path),
                        'status': 'success'
                    }
                    
                    extraction_summary['files_processed'] += 1
                    extraction_summary['total_csv_files_created'] += len(csv_files)
                    extraction_summary['extraction_details'][excel_file.name] = file_summary
                    
                    logger.info(f"Successfully processed {excel_file.name}: {len(csv_files)} CSV files created")
                    
                else:
                    logger.error(f"Failed to extract data from {excel_file.name}")
                    extraction_summary['extraction_details'][excel_file.name] = {
                        'status': 'failed',
                        'error': 'No CSV files created'
                    }
            
            except Exception as e:
                logger.error(f"Critical error processing {excel_file.name}: {e}")
                extraction_summary['extraction_details'][excel_file.name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate total features
        for file_details in extraction_summary['extraction_details'].values():
            if file_details['status'] == 'success':
                metadata_path = file_details['metadata_file']
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    for sheet_info in metadata['feature_summary']['total_features_by_sheet'].values():
                        if 'columns' in sheet_info:
                            extraction_summary['total_features_extracted'] += sheet_info['columns']
                except:
                    pass
        
        return extraction_summary
    
    def generate_extraction_report(self, summary: Dict[str, Any]) -> None:
        """
        Generate comprehensive extraction report.
        
        Args:
            summary: Extraction summary dictionary
        """
        report_filename = self.output_dir / 'extraction_report.json'
        
        final_report = {
            'extraction_metadata': self.metadata,
            'extraction_summary': summary,
            'report_generated': datetime.now().isoformat()
        }
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Extraction report saved to: {report_filename}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("RAW DATA EXTRACTION SUMMARY")
        print("="*80)
        print(f"Files discovered: {summary.get('files_discovered', 0)}")
        print(f"Files processed successfully: {summary.get('files_processed', 0)}")
        print(f"Total CSV files created: {summary.get('total_csv_files_created', 0)}")
        print(f"Total features extracted: {summary.get('total_features_extracted', 0)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Metadata directory: {self.metadata_dir}")
        print("="*80)
        
        # Print file details
        print("\nFILE PROCESSING DETAILS:")
        print("-" * 50)
        for filename, details in summary.get('extraction_details', {}).items():
            status = details.get('status', 'unknown')
            csv_count = details.get('csv_files_created', 0)
            print(f"{filename}: {status} ({csv_count} CSV files)")
        
        print("\n" + "="*80)

def main():
    """
    Main execution function for raw data extraction.
    """
    # Define paths
    source_directory = "/home/sanket/Make_My_route/DATA/CS2_35"
    output_directory = "/home/sanket/Make_My_route/battery-analytics-lab/data/raw/calce"
    
    # Create extractor instance
    extractor = RawDataExtractor(source_directory, output_directory)
    
    # Execute extraction
    try:
        summary = extractor.extract_all_features()
        
        if 'error' not in summary:
            extractor.generate_extraction_report(summary)
            logger.info("Raw data extraction completed successfully!")
            return summary
        else:
            logger.error(f"Extraction failed: {summary['error']}")
            return summary
            
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    main()