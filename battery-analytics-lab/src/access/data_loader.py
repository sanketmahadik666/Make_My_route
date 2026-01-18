import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Utility class to load and filter standardized battery data.
    """
    
    def __init__(self, data_dir: str = "data/standardized/"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Path to the directory containing standardized files and metadata index.
                      Defaults to 'data/standardized/'.
        """
        # Resolve absolute path relative to project root if needed
        # Assuming we are running from project root or src/access
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_absolute():
            # Try to resolve relative to current working directory
            # If that doesn't exist, try relative to project root assumptions
             if not self.data_dir.exists():
                 # Fallback: climb up to find 'data'
                 current = Path.cwd()
                 while current != current.parent:
                     test_path = current / data_dir
                     if test_path.exists():
                         self.data_dir = test_path
                         break
                     current = current.parent
        
        self.index_path = self.data_dir / "metadata_index.csv"
        self._index_df = None
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist.")
    
    def load_index(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the metadata index.
        
        Args:
            force_reload: If True, reload the index even if already cached.
            
        Returns:
            DataFrame containing the metadata index.
        """
        if self._index_df is not None and not force_reload:
            return self._index_df
            
        if not self.index_path.exists():
            logger.warning(f"Metadata index not found at {self.index_path}")
            return pd.DataFrame()
            
        try:
            self._index_df = pd.read_csv(self.index_path)
            # Ensure date columns are datetime objects
            if 'test_date' in self._index_df.columns:
                self._index_df['test_date'] = pd.to_datetime(self._index_df['test_date'])
            if 'ingestion_date' in self._index_df.columns:
                self._index_df['ingestion_date'] = pd.to_datetime(self._index_df['ingestion_date'])
            return self._index_df
        except Exception as e:
            logger.error(f"Error loading metadata index: {e}")
            return pd.DataFrame()
    
    def get_files(self, filter_dict: Optional[Dict[str, Union[str, int, float]]] = None) -> List[str]:
        """
        Get a list of file paths matching the filter criteria.
        
        Args:
            filter_dict: Dictionary of column name -> value to filter by.
                         Example: {'cell_id': 'CS2_35'}
                         
        Returns:
            List of absolute file paths to the standardized parquet files.
        """
        df = self.load_index()
        if df.empty:
            return []
            
        if filter_dict:
            for key, value in filter_dict.items():
                if key in df.columns:
                    df = df[df[key] == value]
                else:
                    logger.warning(f"Filter key '{key}' not found in index columns: {df.columns.tolist()}")
        
        # Construct full paths
        file_paths = []
        for file_name in df['file_name']:
            full_path = self.data_dir / file_name
            if full_path.exists():
                file_paths.append(str(full_path))
            else:
                logger.warning(f"File listed in index but not found: {full_path}")
                
        return file_paths
    
    def load_data(self, file_identifier: str) -> pd.DataFrame:
        """
        Load a standardized parquet file.
        
        Args:
            file_identifier: Can be a full file path, a filename, or a reliable unique substring (like original source).
                             If it's an absolute path, it loads directly.
                             Otherwise, it searches in the index/directory.
            
        Returns:
            DataFrame containing the battery data.
        """
        path_obj = Path(file_identifier)
        
        # Case 1: Absolute path or existing relative path
        if path_obj.exists() and path_obj.is_file():
            try:
                return pd.read_parquet(path_obj)
            except Exception as e:
                logger.error(f"Error loading file {path_obj}: {e}")
                return pd.DataFrame()
                
        # Case 2: Filename looking for match in directory
        possible_path = self.data_dir / file_identifier
        if possible_path.exists():
             try:
                return pd.read_parquet(possible_path)
             except Exception as e:
                logger.error(f"Error loading file {possible_path}: {e}")
                return pd.DataFrame()
        
        # Case 3: Search in index
        df = self.load_index()
        if not df.empty:
            # Check if identifier is in file_name or original_source
            matches = df[df['file_name'] == file_identifier]
            if matches.empty:
                matches = df[df['original_source'] == file_identifier]
            
            if not matches.empty:
                target_file = matches.iloc[0]['file_name']
                full_path = self.data_dir / target_file
                if full_path.exists():
                    return pd.read_parquet(full_path)
        
        logger.error(f"Could not locate file for identifier: {file_identifier}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Simple test
    loader = DataLoader(data_dir="/home/sanket/Make_My_route/battery-analytics-lab/data/standardized/")
    index = loader.load_index()
    print("Index shape:", index.shape)
    print("Columns:", index.columns.tolist() if not index.empty else "Empty")
    
    if not index.empty:
        # Try loading the first file
        first_file_name = index.iloc[0]['file_name']
        print(f"\nLoading {first_file_name}...")
        data = loader.load_data(first_file_name)
        print("Data shape:", data.shape)
        print("Data columns:", data.columns.tolist())
