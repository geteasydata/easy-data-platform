"""
DuckDB Engine - High Performance Data Processing Layer
Enables out-of-core processing for large datasets without memory crashes.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DuckDBEngine:
    """
    Wrapper around DuckDB for efficient data processing.
    Handles connection management and query execution on files.
    """
    
    def __init__(self, memory_limit: str = '2GB'):
        """
        Initialize DuckDB connection.
        Args:
            memory_limit: Max memory DuckDB allowed to use (e.g. '2GB')
        """
        self.con = duckdb.connect(database=':memory:') # In-memory db, but can read files from disk
        
        # Configure robustness
        try:
            self.con.execute(f"SET memory_limit='{memory_limit}'")
            self.con.execute("SET threads to 4") # Reasonable default
        except Exception as e:
            logger.warning(f"Failed to set DuckDB config: {e}")
            
    def register_file(self, file_path: str, table_name: str = 'data') -> Tuple[bool, str]:
        """
        Register a file (CSV/Parquet) as a table/view in DuckDB.
        Returns: (success, error_message)
        """
        file_path = str(Path(file_path).resolve()).replace('\\', '/')
        self.last_error = ""
        
        try:
            # Handle different formats
            if file_path.endswith('.parquet'):
                self.con.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{file_path}')")
                return True, ""
                
            elif file_path.endswith('.csv'):
                try:
                    # Try DuckDB auto-detect first (fastest)
                    self.con.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
                    return True, ""
                except Exception as csv_err:
                    logger.warning(f"DuckDB native CSV load failed, trying Pandas fallback: {csv_err}")
                    # Fallback to Pandas for problematic CSVs (encodings, etc.)
                    try:
                        # Try common encodings
                        for encoding in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                df_tmp = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                                self.con.register(table_name, df_tmp)
                                return True, ""
                            except:
                                continue
                        raise Exception("All common encodings failed for Pandas fallback.")
                    except Exception as pd_err:
                        self.last_error = f"DuckDB: {csv_err} | Pandas Fallback: {pd_err}"
                        logger.error(f"Registration failed completely: {self.last_error}")
                        return False, self.last_error
                
            elif file_path.endswith('.json'):
                 self.con.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_json_auto('{file_path}')")
                 return True, ""
                 
            else:
                self.last_error = f"Unsupported format: {file_path}"
                logger.error(self.last_error)
                return False, self.last_error
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"DuckDB Registration Error: {e}")
            return False, self.last_error

    def query(self, sql: str) -> Optional[pd.DataFrame]:
        """Execute arbirtary SQL and return Pandas DataFrame."""
        try:
            return self.con.execute(sql).df()
        except Exception as e:
            logger.error(f"DuckDB Query Error: {e}")
            return None

    def get_summary_stats(self, table_name: str = 'data') -> Dict[str, Any]:
        """Get summary statistics efficiently."""
        try:
            # 1. Total Rows
            total_rows = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            # 2. Get Columns
            columns_info = self.con.execute(f"DESCRIBE {table_name}").fetchall()
            # columns_info format: (column_name, column_type, null, key, default, extra)
            
            columns = [c[0] for c in columns_info]
            dtypes = {c[0]: c[1] for c in columns_info}
            
            # 3. Quick Missing Count (Optional optimization: do logical columns at once)
            # Construct a single query for all counts
            missing_aggregates = ", ".join([f"COUNT(*) - COUNT(\"{col}\") as \"{col}_missing\"" for col in columns])
            missing_df = self.con.execute(f"SELECT {missing_aggregates} FROM {table_name}").df()
            
            missing_counts = missing_df.iloc[0].to_dict()
            total_missing = sum(missing_counts.values())

            return {
                "total_rows": total_rows,
                "total_columns": len(columns),
                "columns": columns,
                "dtypes": dtypes,
                "missing_counts": missing_counts,
                "total_missing": total_missing
            }
        except Exception as e:
            logger.error(f"DuckDB Stats Error: {e}")
            return {}

    def get_sample(self, table_name: str = 'data', limit: int = 1000) -> pd.DataFrame:
        """Get a sample of data"""
        return self.con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").df()
