"""
Universal Data Loader
Supports CSV, Excel, JSON, Parquet, and more
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
import json

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Universal Data Loader
    Automatically detects file format and loads data efficiently
    """
    
    SUPPORTED_FORMATS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.json': 'json',
        '.parquet': 'parquet',
        '.feather': 'feather',
        '.pkl': 'pickle',
        '.pickle': 'pickle',
        '.tsv': 'tsv',
        '.txt': 'txt'
    }
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        
    @property
    def supported_formats(self) -> list:
        return list(self.SUPPORTED_FORMATS.keys()) + ['.sql', '.db']

    def load_sql(self, connection_string: str, query: str) -> Optional[pd.DataFrame]:
        """
        Load data from a SQL database.
        """
        try:
            from sqlalchemy import create_engine
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                self.data = pd.read_sql(query, conn)
            
            self._extract_metadata(Path("sql_query"))
            logger.info(f"Loaded {len(self.data)} rows from SQL")
            return self.data
        except ImportError:
            logger.error("SQLAlchemy not installed")
            return None
        except Exception as e:
            logger.error(f"SQL Error: {str(e)}")
            return None

    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from file with automatic format detection"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        ext = path.suffix.lower()
        file_type = self.SUPPORTED_FORMATS.get(ext)
        
        if file_type is None:
            raise ValueError(f"Unsupported file format: {ext}")
        
        logger.info(f"Loading {file_type} file: {path.name}")
        
        # Load based on file type
        if file_type == 'csv':
            self.data = self._load_csv(path, **kwargs)
        elif file_type == 'excel':
            self.data = self._load_excel(path, **kwargs)
        elif file_type == 'json':
            self.data = self._load_json(path, **kwargs)
        elif file_type == 'parquet':
            self.data = self._load_parquet(path, **kwargs)
        elif file_type == 'feather':
            self.data = pd.read_feather(path)
        elif file_type == 'pickle':
            self.data = pd.read_pickle(path)
        elif file_type in ['tsv', 'txt']:
            self.data = self._load_csv(path, sep='\t', **kwargs)
        
        # Store metadata
        self._extract_metadata(path)
        
        logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
        return self.data
    
    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV with intelligent encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding, **kwargs)
            except (UnicodeDecodeError, Exception):
                continue
        
        # Last resort: ignore errors
        return pd.read_csv(path, encoding='utf-8', errors='ignore', **kwargs)
    
    def _load_excel(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file with sheet detection"""
        sheet_name = kwargs.pop('sheet_name', 0)
        
        # Try to load with openpyxl first, then xlrd
        try:
            return pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl', **kwargs)
        except Exception:
            try:
                return pd.read_excel(path, sheet_name=sheet_name, engine='xlrd', **kwargs)
            except Exception:
                return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    
    def _load_json(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON with automatic orientation detection"""
        try:
            return pd.read_json(path, **kwargs)
        except ValueError:
            # Try different orientations
            for orient in ['records', 'columns', 'index', 'split']:
                try:
                    return pd.read_json(path, orient=orient, **kwargs)
                except Exception:
                    continue
        
        # Last resort: load as dict and convert
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                return pd.DataFrame(data['data'])
            return pd.DataFrame([data])
        
        raise ValueError("Could not parse JSON file")
    
    def _load_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file"""
        try:
            return pd.read_parquet(path, engine='pyarrow', **kwargs)
        except Exception:
            return pd.read_parquet(path, engine='fastparquet', **kwargs)
    
    def _extract_metadata(self, path: Path):
        """Extract metadata from loaded data"""
        self.metadata = {
            'file_name': path.name,
            'file_path': str(path),
            'file_size': path.stat().st_size,
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': self.data.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
    
    def load_sample_data(self, domain: str) -> pd.DataFrame:
        """Load sample data for a specific domain"""
        sample_dir = Path(__file__).parent.parent / "sample_data"
        
        sample_files = {
            'hr': 'hr_employees.csv',
            'finance': 'finance_transactions.csv',
            'healthcare': 'healthcare_patients.csv',
            'retail': 'retail_sales.csv',
            'marketing': 'marketing_campaigns.csv',
            'education': 'education_students.csv'
        }
        
        if domain not in sample_files:
            raise ValueError(f"No sample data for domain: {domain}")
        
        file_path = sample_dir / sample_files[domain]
        
        if file_path.exists():
            return self.load(file_path)
        else:
            # Generate sample data if file doesn't exist
            return self._generate_sample_data(domain)
    
    def _generate_sample_data(self, domain: str) -> pd.DataFrame:
        """Generate sample data for a domain"""
        np.random.seed(42)
        n_rows = 500
        
        if domain == 'hr':
            data = {
                'employee_id': range(1, n_rows + 1),
                'name': [f'Employee_{i}' for i in range(1, n_rows + 1)],
                'department': np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing', 'Finance'], n_rows),
                'salary': np.random.randint(30000, 150000, n_rows),
                'age': np.random.randint(22, 65, n_rows),
                'tenure_years': np.random.uniform(0, 30, n_rows).round(1),
                'performance_score': np.random.uniform(1, 5, n_rows).round(2),
                'satisfaction_score': np.random.uniform(1, 5, n_rows).round(2),
                'is_active': np.random.choice([True, False], n_rows, p=[0.9, 0.1])
            }
        elif domain == 'finance':
            data = {
                'transaction_id': range(1, n_rows + 1),
                'date': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
                'amount': np.random.uniform(10, 10000, n_rows).round(2),
                'category': np.random.choice(['Sales', 'Expense', 'Investment', 'Refund'], n_rows),
                'account': np.random.choice(['Checking', 'Savings', 'Credit'], n_rows),
                'customer_id': np.random.randint(1, 100, n_rows)
            }
        elif domain == 'healthcare':
            data = {
                'patient_id': range(1, n_rows + 1),
                'age': np.random.randint(1, 100, n_rows),
                'gender': np.random.choice(['Male', 'Female'], n_rows),
                'diagnosis': np.random.choice(['Flu', 'Cold', 'Injury', 'Chronic', 'Other'], n_rows),
                'length_of_stay': np.random.randint(1, 30, n_rows),
                'total_charges': np.random.uniform(500, 50000, n_rows).round(2),
                'readmission': np.random.choice([True, False], n_rows, p=[0.1, 0.9])
            }
        elif domain == 'retail':
            data = {
                'order_id': range(1, n_rows + 1),
                'date': pd.date_range('2023-01-01', periods=n_rows, freq='H'),
                'product': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home', 'Sports'], n_rows),
                'quantity': np.random.randint(1, 10, n_rows),
                'price': np.random.uniform(5, 500, n_rows).round(2),
                'customer_segment': np.random.choice(['Regular', 'Premium', 'New'], n_rows)
            }
            data['total'] = np.array(data['quantity']) * np.array(data['price'])
        elif domain == 'marketing':
            data = {
                'campaign_id': range(1, n_rows + 1),
                'channel': np.random.choice(['Email', 'Social', 'Search', 'Display', 'TV'], n_rows),
                'spend': np.random.uniform(100, 10000, n_rows).round(2),
                'impressions': np.random.randint(1000, 1000000, n_rows),
                'clicks': np.random.randint(10, 10000, n_rows),
                'conversions': np.random.randint(0, 500, n_rows),
                'revenue': np.random.uniform(0, 50000, n_rows).round(2)
            }
        elif domain == 'education':
            data = {
                'student_id': range(1, n_rows + 1),
                'course': np.random.choice(['Math', 'Science', 'English', 'History', 'Art'], n_rows),
                'grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], n_rows, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
                'attendance': np.random.uniform(0.5, 1.0, n_rows).round(2),
                'score': np.random.randint(50, 100, n_rows),
                'passed': np.random.choice([True, False], n_rows, p=[0.8, 0.2])
            }
        else:
            # Generic data
            data = {
                'id': range(1, n_rows + 1),
                'value1': np.random.randn(n_rows),
                'value2': np.random.randn(n_rows),
                'category': np.random.choice(['A', 'B', 'C'], n_rows)
            }
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about loaded data"""
        if self.data is None:
            return {"error": "No data loaded"}
        return self.metadata


# Convenience function
def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from file"""
    loader = DataLoader()
    return loader.load(file_path, **kwargs)
