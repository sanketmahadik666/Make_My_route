import sys
import os
import yaml
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.access.data_loader import DataLoader

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'quality_check.log')
    ]
)
logger = logging.getLogger('QualityCheck')

class QualityValidator:
    def __init__(self, schema_path: str = "config/schema.yaml"):
        self.loader = DataLoader(data_dir=str(PROJECT_ROOT / "data/standardized"))
        self.schema_path = PROJECT_ROOT / schema_path
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Dict:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found at {self.schema_path}")
        with open(self.schema_path, 'r') as f:
            return yaml.safe_load(f)

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate a single file against the schema."""
        try:
            df = self.loader.load_data(file_path)
            if df.empty:
                return {"status": "FAIL", "reason": "Empty dataframe", "score": 0}

            validation_errors = []
            
            # 1. Check Required Fields
            missing_fields = []
            for field in self.schema.get('required_fields', []):
                if field['name'] not in df.columns:
                    missing_fields.append(field['name'])
            
            if missing_fields:
                return {
                    "status": "FAIL", 
                    "reason": f"Missing required fields: {missing_fields}",
                    "score": 0
                }

            # 2. Check Constraints & Data Quality
            score = 100
            for field in self.schema.get('required_fields', []):
                col = field['name']
                
                # Check NaNs
                null_pct = df[col].isnull().mean()
                if null_pct > self.schema['quality_thresholds']['max_missing_pct']:
                    validation_errors.append(f"{col} has {null_pct:.2%} missing values")
                    score -= 20
                
                # Check Value Constraints
                constraints = field.get('constraints', {})
                if constraints:
                    if 'min' in constraints:
                        # Allow small tolerance for floating point noise around 0
                        min_val = constraints['min']
                        if (df[col] < min_val - 1e-6).any():
                             # Be practical: some noise is expected
                             fail_pct = (df[col] < min_val).mean()
                             if fail_pct > 0.05: # >5% violation
                                validation_errors.append(f"{col} violates min {min_val}")
                                score -= 10
                    
                    if 'max' in constraints:
                         max_val = constraints['max']
                         if (df[col] > max_val).any():
                             fail_pct = (df[col] > max_val).mean()
                             if fail_pct > 0.05:
                                 validation_errors.append(f"{col} violates max {max_val}")
                                 score -= 10

            # 3. Check Row Count
            if len(df) < self.schema['quality_thresholds']['min_rows']:
                validation_errors.append(f"Insufficient rows: {len(df)}")
                score -= 50

            # Final Decision
            status = "PASS" if score >= 80 and not validation_errors else "WARNING"
            if score < 50: 
                status = "FAIL"
            
            return {
                "status": status,
                "score": max(0, score),
                "issues": validation_errors,
                "rows": len(df),
                "test_date": df['Date_Time'].iloc[0] if 'Date_Time' in df.columns and not df.empty else None
            }

        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            return {"status": "ERROR", "reason": str(e), "score": 0}

    def infer_scenario(self, file_name: str, validation_res: Dict) -> str:
        """Infer scenario name based on file attributes."""
        # Heuristic: Name based on data quality and potentially Test Date
        if validation_res['status'] != 'PASS':
            return "Rejected_Quality"
        
        # Simple heuristic for now: Standard Testing
        # Could be enhanced if we looked at Step_Index patterns
        return "Continuous_Cycling"

    def run_validation(self):
        """Run validation on all files and generate index."""
        logger.info("Starting Data Validation...")
        files = self.loader.get_files()
        results = []

        for file_path in files:
            file_name = os.path.basename(file_path)
            logger.info(f"Validating {file_name}...")
            
            res = self.validate_file(file_path)
            scenario = self.infer_scenario(file_name, res)
            
            results.append({
                "file_name": file_name,
                "scenario": scenario,
                "status": res.get("status", "ERROR"),
                "quality_score": res.get("score", 0),
                "num_records": res.get("rows", 0),
                "issues": "; ".join(res.get("issues", [])),
                "test_date": res.get("test_date")
            })

        # Create DataFrame
        final_df = pd.DataFrame(results)
        
        # Sort by date
        if 'test_date' in final_df.columns:
            final_df['test_date'] = pd.to_datetime(final_df['test_date'])
            final_df = final_df.sort_values('test_date')

        # Save
        output_path = PROJECT_ROOT / "data/standardized/final_data_index.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(f"Validation complete. Index saved to {output_path}")
        
        # Summary
        print("\nValidation Summary:")
        print(final_df['status'].value_counts())
        print(f"\nFinal Index Location: {output_path}")
        return final_df

if __name__ == "__main__":
    validator = QualityValidator()
    validator.run_validation()
