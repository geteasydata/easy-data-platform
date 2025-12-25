"""
Database Connectors for Data Science Master System.

Provides unified database connectivity:
- SQL: PostgreSQL, MySQL, SQLite, SQL Server, Oracle
- NoSQL: MongoDB, Redis, Elasticsearch, Cassandra

Example:
    >>> conn = SQLConnector.from_url("postgresql://user:pass@localhost/db")
    >>> df = conn.query("SELECT * FROM users")
    >>> conn.write(df, "processed_users")
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
import urllib.parse

from data_science_master_system.core.base_classes import BaseDataSource
from data_science_master_system.core.exceptions import DataIngestionError
from data_science_master_system.core.logger import get_logger
from data_science_master_system.utils.decorators import retry

logger = get_logger(__name__)


class DatabaseConnector(BaseDataSource):
    """
    Abstract base class for database connectors.
    
    Provides common interface for all database types.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize database connector.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Username
            password: Password
            config: Additional configuration
        """
        super().__init__(config)
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self._connection = None
    
    @classmethod
    @abstractmethod
    def from_url(cls, url: str) -> "DatabaseConnector":
        """Create connector from connection URL."""
        pass
    
    @abstractmethod
    def query(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute query and return results."""
        pass
    
    @abstractmethod
    def execute(self, query: str, params: Optional[Dict] = None) -> None:
        """Execute query without returning results."""
        pass
    
    @abstractmethod
    def write(
        self,
        data: Any,
        table: str,
        if_exists: str = "append",
        **kwargs: Any,
    ) -> None:
        """Write DataFrame to database table."""
        pass
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield
            self._connection.commit()
        except Exception as e:
            self._connection.rollback()
            raise e


class SQLConnector(DatabaseConnector):
    """
    SQL database connector using SQLAlchemy.
    
    Supports: PostgreSQL, MySQL, SQLite, SQL Server, Oracle.
    
    Example:
        >>> conn = SQLConnector(
        ...     host="localhost",
        ...     port=5432,
        ...     database="mydb",
        ...     username="user",
        ...     password="pass",
        ...     dialect="postgresql",
        ... )
        >>> conn.connect()
        >>> df = conn.query("SELECT * FROM users WHERE active = :active", {"active": True})
    """
    
    DIALECTS = {
        "postgresql": {"port": 5432, "driver": "postgresql+psycopg2"},
        "postgres": {"port": 5432, "driver": "postgresql+psycopg2"},
        "mysql": {"port": 3306, "driver": "mysql+pymysql"},
        "sqlite": {"port": None, "driver": "sqlite"},
        "mssql": {"port": 1433, "driver": "mssql+pyodbc"},
        "oracle": {"port": 1521, "driver": "oracle+cx_oracle"},
    }
    
    def __init__(
        self,
        host: str = "localhost",
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        dialect: str = "postgresql",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(host, port, database, username, password, config)
        self.dialect = dialect.lower()
        
        if self.dialect not in self.DIALECTS:
            raise DataIngestionError(
                f"Unsupported SQL dialect: {dialect}",
                context={"supported": list(self.DIALECTS.keys())},
            )
        
        # Set default port
        if self.port is None:
            self.port = self.DIALECTS[self.dialect]["port"]
        
        self._engine = None
    
    @classmethod
    def from_url(cls, url: str) -> "SQLConnector":
        """
        Create connector from database URL.
        
        Args:
            url: SQLAlchemy-style connection URL
            
        Returns:
            SQLConnector instance
            
        Example:
            >>> conn = SQLConnector.from_url("postgresql://user:pass@localhost:5432/mydb")
        """
        parsed = urllib.parse.urlparse(url)
        
        # Extract dialect from scheme
        dialect = parsed.scheme.split("+")[0]
        
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port,
            database=parsed.path.lstrip("/") if parsed.path else None,
            username=parsed.username,
            password=parsed.password,
            dialect=dialect,
        )
    
    @property
    def connection_url(self) -> str:
        """Build SQLAlchemy connection URL."""
        driver = self.DIALECTS[self.dialect]["driver"]
        
        if self.dialect == "sqlite":
            return f"{driver}:///{self.database}"
        
        url = f"{driver}://"
        if self.username:
            url += f"{self.username}"
            if self.password:
                url += f":{urllib.parse.quote_plus(self.password)}"
            url += "@"
        
        url += f"{self.host}"
        if self.port:
            url += f":{self.port}"
        if self.database:
            url += f"/{self.database}"
        
        return url
    
    @retry(max_attempts=3, delay=1.0)
    def connect(self) -> None:
        """Establish database connection."""
        try:
            from sqlalchemy import create_engine
            
            self._engine = create_engine(
                self.connection_url,
                pool_pre_ping=True,
                pool_size=self.config.get("pool_size", 5),
                max_overflow=self.config.get("max_overflow", 10),
                echo=self.config.get("echo", False),
            )
            
            # Test connection
            with self._engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self._connected = True
            logger.info(f"Connected to {self.dialect} database", database=self.database)
            
        except ImportError:
            raise DataIngestionError(
                "SQLAlchemy not installed. Install with: pip install sqlalchemy",
            )
        except Exception as e:
            raise DataIngestionError(
                f"Failed to connect to database",
                context={"host": self.host, "database": self.database, "error": str(e)},
            )
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._connected = False
            logger.info("Disconnected from database")
    
    def read(self, query: str = None, table: str = None, **kwargs: Any) -> Any:
        """
        Read data from database.
        
        Args:
            query: SQL query string
            table: Table name (if no query)
            **kwargs: Additional pandas read_sql options
            
        Returns:
            pandas DataFrame
        """
        if query:
            return self.query(query, **kwargs)
        elif table:
            return self.query(f"SELECT * FROM {table}", **kwargs)
        else:
            raise DataIngestionError("Must provide either query or table name")
    
    def query(
        self,
        query: str,
        params: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            **kwargs: Additional options
            
        Returns:
            pandas DataFrame
        """
        if not self._connected:
            self.connect()
        
        try:
            import pandas as pd
            from sqlalchemy import text
            
            with self._engine.connect() as conn:
                result = pd.read_sql(
                    text(query),
                    conn,
                    params=params,
                    **kwargs,
                )
            
            logger.debug(f"Query executed", rows=len(result))
            return result
            
        except Exception as e:
            raise DataIngestionError(
                "Query execution failed",
                context={"query": query[:200], "error": str(e)},
            )
    
    def execute(
        self,
        query: str,
        params: Optional[Dict] = None,
    ) -> None:
        """
        Execute SQL statement without returning results.
        
        Args:
            query: SQL statement
            params: Statement parameters
        """
        if not self._connected:
            self.connect()
        
        try:
            from sqlalchemy import text
            
            with self._engine.begin() as conn:
                conn.execute(text(query), params or {})
            
            logger.debug("Statement executed")
            
        except Exception as e:
            raise DataIngestionError(
                "Statement execution failed",
                context={"query": query[:200], "error": str(e)},
            )
    
    def write(
        self,
        data: Any,
        table: str,
        if_exists: str = "append",
        index: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Write DataFrame to database table.
        
        Args:
            data: pandas DataFrame
            table: Target table name
            if_exists: How to handle existing table ("fail", "replace", "append")
            index: Whether to write index
            **kwargs: Additional to_sql options
        """
        if not self._connected:
            self.connect()
        
        try:
            data.to_sql(
                table,
                self._engine,
                if_exists=if_exists,
                index=index,
                **kwargs,
            )
            
            logger.info(f"Wrote data to table", table=table, rows=len(data))
            
        except Exception as e:
            raise DataIngestionError(
                f"Failed to write to table",
                context={"table": table, "error": str(e)},
            )
    
    def get_tables(self) -> List[str]:
        """Get list of tables in database."""
        from sqlalchemy import inspect
        
        if not self._connected:
            self.connect()
        
        inspector = inspect(self._engine)
        return inspector.get_table_names()
    
    def get_columns(self, table: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        from sqlalchemy import inspect
        
        if not self._connected:
            self.connect()
        
        inspector = inspect(self._engine)
        return inspector.get_columns(table)


class MongoDBConnector(DatabaseConnector):
    """
    MongoDB connector.
    
    Example:
        >>> conn = MongoDBConnector(host="localhost", database="mydb")
        >>> conn.connect()
        >>> docs = conn.find("users", {"active": True})
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(host, port, database, username, password, config)
        self._client = None
        self._db = None
    
    @classmethod
    def from_url(cls, url: str) -> "MongoDBConnector":
        """Create connector from MongoDB URL."""
        parsed = urllib.parse.urlparse(url)
        database = parsed.path.lstrip("/") if parsed.path else None
        
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 27017,
            database=database,
            username=parsed.username,
            password=parsed.password,
        )
    
    def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            from pymongo import MongoClient
            
            uri = f"mongodb://"
            if self.username and self.password:
                uri += f"{self.username}:{self.password}@"
            uri += f"{self.host}:{self.port}"
            
            self._client = MongoClient(uri)
            if self.database:
                self._db = self._client[self.database]
            
            # Test connection
            self._client.admin.command("ping")
            
            self._connected = True
            logger.info(f"Connected to MongoDB", database=self.database)
            
        except ImportError:
            raise DataIngestionError("pymongo not installed")
        except Exception as e:
            raise DataIngestionError(f"MongoDB connection failed: {e}")
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._connected = False
    
    def read(self, collection: str = None, query: Dict = None, **kwargs: Any) -> Any:
        """Read documents from collection."""
        return self.find(collection, query or {}, **kwargs)
    
    def query(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute query (uses aggregation pipeline for MongoDB)."""
        raise NotImplementedError("Use find() or aggregate() for MongoDB queries")
    
    def execute(self, query: str, params: Optional[Dict] = None) -> None:
        """Execute command."""
        if not self._connected:
            self.connect()
        self._db.command(query)
    
    def find(
        self,
        collection: str,
        query: Dict = None,
        projection: Optional[Dict] = None,
        limit: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Find documents in collection.
        
        Args:
            collection: Collection name
            query: MongoDB query filter
            projection: Fields to include/exclude
            limit: Maximum documents to return
            
        Returns:
            pandas DataFrame
        """
        if not self._connected:
            self.connect()
        
        import pandas as pd
        
        cursor = self._db[collection].find(
            query or {},
            projection=projection,
            limit=limit,
            **kwargs,
        )
        
        docs = list(cursor)
        if docs:
            return pd.DataFrame(docs)
        return pd.DataFrame()
    
    def insert(
        self,
        collection: str,
        documents: Union[Dict, List[Dict]],
    ) -> List[str]:
        """Insert documents into collection."""
        if not self._connected:
            self.connect()
        
        if isinstance(documents, dict):
            result = self._db[collection].insert_one(documents)
            return [str(result.inserted_id)]
        else:
            result = self._db[collection].insert_many(documents)
            return [str(id) for id in result.inserted_ids]
    
    def write(
        self,
        data: Any,
        table: str,
        if_exists: str = "append",
        **kwargs: Any,
    ) -> None:
        """Write DataFrame to MongoDB collection."""
        if not self._connected:
            self.connect()
        
        if if_exists == "replace":
            self._db[table].drop()
        
        records = data.to_dict("records")
        if records:
            self._db[table].insert_many(records)
            logger.info(f"Inserted documents", collection=table, count=len(records))


class RedisConnector(DatabaseConnector):
    """
    Redis connector for caching and key-value operations.
    
    Example:
        >>> conn = RedisConnector(host="localhost")
        >>> conn.connect()
        >>> conn.set("key", "value")
        >>> value = conn.get("key")
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        database: int = 0,
        password: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(host, port, str(database), None, password, config)
        self.db_number = database
        self._redis = None
    
    @classmethod
    def from_url(cls, url: str) -> "RedisConnector":
        """Create connector from Redis URL."""
        parsed = urllib.parse.urlparse(url)
        db = int(parsed.path.lstrip("/")) if parsed.path.lstrip("/") else 0
        
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            database=db,
            password=parsed.password,
        )
    
    def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis
            
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db_number,
                password=self.password,
                decode_responses=True,
            )
            
            self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis", host=self.host, db=self.db_number)
            
        except ImportError:
            raise DataIngestionError("redis not installed")
        except Exception as e:
            raise DataIngestionError(f"Redis connection failed: {e}")
    
    def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            self._connected = False
    
    def read(self, key: str = None, **kwargs: Any) -> Any:
        """Read value from Redis."""
        return self.get(key)
    
    def query(self, query: str, params: Optional[Dict] = None) -> Any:
        """Not applicable for Redis."""
        raise NotImplementedError("Redis is key-value store, use get/set methods")
    
    def execute(self, query: str, params: Optional[Dict] = None) -> None:
        """Execute Redis command."""
        if not self._connected:
            self.connect()
        self._redis.execute_command(query)
    
    def write(self, data: Any, table: str, **kwargs: Any) -> None:
        """Write to Redis (stores entire DataFrame as JSON)."""
        import json
        if not self._connected:
            self.connect()
        
        if hasattr(data, "to_json"):
            value = data.to_json()
        else:
            value = json.dumps(data)
        
        self._redis.set(table, value)
    
    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if not self._connected:
            self.connect()
        return self._redis.get(key)
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set key-value pair."""
        if not self._connected:
            self.connect()
        return self._redis.set(key, value, ex=ex)
    
    def delete(self, key: str) -> int:
        """Delete key."""
        if not self._connected:
            self.connect()
        return self._redis.delete(key)
