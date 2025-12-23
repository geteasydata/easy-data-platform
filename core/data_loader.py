"""
Data Loader - Universal Data Loading Module
Supports: CSV, Excel, JSON, Parquet, SQL, and more
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
import json


class DataLoader:
    """Universal data loader supporting multiple file formats."""
    
    SUPPORTED_FORMATS = {
        '.csv': 'CSV',
        '.xlsx': 'Excel',
        '.xls': 'Excel (Legacy)',
        '.json': 'JSON',
        '.parquet': 'Parquet',
        '.feather': 'Feather',
        '.pkl': 'Pickle',
        '.tsv': 'TSV',
        '.txt': 'Text',
        '.sql': 'SQL Database', # Virtual extension
        '.db': 'SQLite Database'
    }
    
    def __init__(self):
        self.loaded_data = {}
        self.errors = []
    
    @property
    def supported_formats(self) -> List[str]:
        """Return list of supported file extensions."""
        return list(self.SUPPORTED_FORMATS.keys())
        
    def load_sql(self, connection_string: str, query: str) -> Optional[pd.DataFrame]:
        """Load data from a SQL database."""
        try:
            from sqlalchemy import create_engine
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)
            return df
        except ImportError:
            self.errors.append("SQLAlchemy not installed")
            return None
        except Exception as e:
            self.errors.append(f"SQL Error: {str(e)}")
            return None

    def load_url(self, url: str) -> Optional[pd.DataFrame]:
        """
        Load data directly from a public URL.
        Supports CSV and Parquet.
        """
        try:
            import requests
            
            # 1. Quick check for HTML pages (common user mistake: pasting doc link)
            try:
                # Use GET with stream=True to only check headers first
                response = requests.get(url, stream=True, timeout=10, allow_redirects=True)
                content_type = response.headers.get('Content-Type', '').lower()
                
                if 'text/html' in content_type:
                    self.errors.append("⚠️ هذا الرابط يؤدي إلى صفحة ويب (HTML) وليس ملف بيانات خام.\n\n"
                                     "يرجى استخدام 'رابط التحميل المباشر' (Direct Download Link) للملف (بصيغة .csv أو .parquet).")
                    return None
            except:
                pass # Fallback to pandas directly if requests fails

            # 2. Try loading
            if 'parquet' in url.lower():
                return pd.read_parquet(url)
            else:
                # Try CSV with multiple common delimiters
                # We start with ',' then try others if it fails
                for sep in [',', ';', '\t']:
                    try:
                        # Use a small sample to check if it works
                        df = pd.read_csv(url, sep=sep, nrows=5)
                        # If successful, load full data
                        return pd.read_csv(url, sep=sep)
                    except:
                        continue
                
                # If all attempts failed, try once more to get a clean error message
                return pd.read_csv(url)
                
        except Exception as e:
            error_msg = str(e)
            if "tokenizing data" in error_msg:
                 self.errors.append("❌ خطأ في شكل الملف: النظام واجه مشكلة في تفكيك البيانات.\n"
                                  "تأكد أن الرابط ليس لصفحة ويب (HTML) وأنه ملف CSV أو Parquet صحيح.")
            else:
                 self.errors.append(f"URL Loading Error: {error_msg}")
            return None

    def load_file(self, file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
        """
        Load a single file into a DataFrame.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for pandas readers
            
        Returns:
            DataFrame or None if loading fails
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        try:
            if extension == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1256', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(path, encoding=encoding, **kwargs)
                        return df
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV with any known encoding")
                
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(path, **kwargs)
                return df
                
            elif extension == '.json':
                # Try different JSON structures
                try:
                    df = pd.read_json(path, **kwargs)
                except ValueError:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                    else:
                        raise ValueError("Unsupported JSON structure")
                return df
                
            elif extension == '.parquet':
                df = pd.read_parquet(path, **kwargs)
                return df
                
            elif extension == '.feather':
                df = pd.read_feather(path, **kwargs)
                return df
                
            elif extension == '.pkl':
                df = pd.read_pickle(path, **kwargs)
                return df
                
            elif extension in ['.tsv', '.txt']:
                df = pd.read_csv(path, sep='\t', **kwargs)
                return df
                
            elif extension in ['.db', '.sqlite', '.sqlite3']:
                 # Basic SQLite support via file path
                 return self.load_sql(f"sqlite:///{path}", "SELECT * FROM sqlite_master") # Just a test, user needs query. 
                 # Better: Just treat as file but return note
                 self.errors.append("For SQLite files, please use the SQL Connection tab")
                 return None

            else:
                self.errors.append(f"Unsupported format: {extension}")
                return None
                
        except Exception as e:
            self.errors.append(f"Error loading {path.name}: {str(e)}")
            return None
    
    def load_multiple_files(self, file_paths: List[Union[str, Path]], 
                           merge: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load multiple files.
        
        Args:
            file_paths: List of file paths
            merge: If True, concatenate all DataFrames
            
        Returns:
            Single DataFrame if merge=True, else list of DataFrames
        """
        dataframes = []
        
        for path in file_paths:
            df = self.load_file(path)
            if df is not None:
                dataframes.append(df)
        
        if not dataframes:
            return pd.DataFrame()
        
        if merge and len(dataframes) > 1:
            try:
                return pd.concat(dataframes, ignore_index=True)
            except Exception as e:
                self.errors.append(f"Could not merge files: {str(e)}")
                return dataframes[0]
        
        return dataframes[0] if len(dataframes) == 1 else dataframes
    
    def get_file_info(self, file_path: Union[str, Path]) -> dict:
        """Get basic info about a file without fully loading it."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        info = {
            'name': path.name,
            'extension': extension,
            'format': self.SUPPORTED_FORMATS.get(extension, 'Unknown'),
            'size_mb': path.stat().st_size / (1024 * 1024) if path.exists() else 0
        }
        
        return info


def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load a Streamlit uploaded file object.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        DataFrame or None
    """
    try:
        name = uploaded_file.name.lower()
        
        if name.endswith('.csv'):
            for encoding in ['utf-8', 'latin-1', 'cp1256']:
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=encoding)
                except:
                    continue
                    
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
            
        elif name.endswith('.json'):
            return pd.read_json(uploaded_file)
            
        elif name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
            
        return None
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
