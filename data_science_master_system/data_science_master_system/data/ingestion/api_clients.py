"""
API Clients for Data Science Master System.

Provides clients for:
- REST APIs with various authentication methods
- GraphQL APIs
- OAuth2 authentication flows

Example:
    >>> client = RESTClient("https://api.example.com", auth=("user", "pass"))
    >>> data = client.get("/users")
    >>> client.post("/users", json={"name": "John"})
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin
import time

from data_science_master_system.core.base_classes import BaseDataSource
from data_science_master_system.core.exceptions import DataIngestionError
from data_science_master_system.core.logger import get_logger
from data_science_master_system.utils.decorators import retry, rate_limit

logger = get_logger(__name__)


class APIClient(BaseDataSource):
    """
    Abstract base class for API clients.
    """
    
    def __init__(
        self,
        base_url: str,
        auth: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize API client.
        
        Args:
            base_url: Base URL for API
            auth: Authentication (tuple for basic, string for bearer)
            headers: Default headers
            timeout: Request timeout in seconds
            config: Additional configuration
        """
        super().__init__(config)
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.headers = headers or {}
        self.timeout = timeout
        self._session = None
    
    @abstractmethod
    def request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        """Make HTTP request."""
        pass


class RESTClient(APIClient):
    """
    REST API client with comprehensive features.
    
    Supports:
    - Basic, Bearer, and API key authentication
    - Automatic retries with backoff
    - Rate limiting
    - Pagination handling
    
    Example:
        >>> client = RESTClient(
        ...     base_url="https://api.example.com",
        ...     auth=("username", "password"),
        ... )
        >>> 
        >>> # GET request
        >>> users = client.get("/users", params={"active": True})
        >>> 
        >>> # POST request
        >>> new_user = client.post("/users", json={"name": "John"})
        >>> 
        >>> # Paginated request
        >>> all_users = client.get_paginated("/users", page_param="page")
    """
    
    def __init__(
        self,
        base_url: str,
        auth: Optional[Union[tuple, str]] = None,
        api_key: Optional[str] = None,
        api_key_header: str = "X-API-Key",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(base_url, auth, headers, timeout, config)
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.verify_ssl = verify_ssl
    
    def connect(self) -> None:
        """Initialize requests session."""
        try:
            import requests
            
            self._session = requests.Session()
            
            # Set default headers
            self._session.headers.update(self.headers)
            self._session.headers["User-Agent"] = "DSMS-APIClient/1.0"
            
            # Set authentication
            if isinstance(self.auth, tuple):
                self._session.auth = self.auth
            elif isinstance(self.auth, str):
                self._session.headers["Authorization"] = f"Bearer {self.auth}"
            
            if self.api_key:
                self._session.headers[self.api_key_header] = self.api_key
            
            self._connected = True
            logger.info(f"API client initialized", base_url=self.base_url)
            
        except ImportError:
            raise DataIngestionError("requests not installed")
    
    def disconnect(self) -> None:
        """Close session."""
        if self._session:
            self._session.close()
            self._connected = False
    
    def read(self, endpoint: str = "/", **kwargs: Any) -> Any:
        """Read data from API endpoint."""
        return self.get(endpoint, **kwargs)
    
    def write(self, data: Any, endpoint: str = "/", **kwargs: Any) -> None:
        """Write data to API endpoint."""
        self.post(endpoint, json=data, **kwargs)
    
    @retry(max_attempts=3, delay=1.0)
    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Make HTTP request.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint
            params: Query parameters
            json: JSON body
            data: Form data
            headers: Additional headers
            **kwargs: Additional request options
            
        Returns:
            Response data (parsed JSON or text)
        """
        if not self._connected:
            self.connect()
        
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        
        try:
            response = self._session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json,
                data=data,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
                **kwargs,
            )
            
            response.raise_for_status()
            
            # Parse response
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text
            
        except Exception as e:
            raise DataIngestionError(
                f"API request failed",
                context={"url": url, "method": method, "error": str(e)},
            )
    
    def get(self, endpoint: str, **kwargs: Any) -> Any:
        """Make GET request."""
        return self.request("GET", endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs: Any) -> Any:
        """Make POST request."""
        return self.request("POST", endpoint, **kwargs)
    
    def put(self, endpoint: str, **kwargs: Any) -> Any:
        """Make PUT request."""
        return self.request("PUT", endpoint, **kwargs)
    
    def patch(self, endpoint: str, **kwargs: Any) -> Any:
        """Make PATCH request."""
        return self.request("PATCH", endpoint, **kwargs)
    
    def delete(self, endpoint: str, **kwargs: Any) -> Any:
        """Make DELETE request."""
        return self.request("DELETE", endpoint, **kwargs)
    
    def get_paginated(
        self,
        endpoint: str,
        page_param: str = "page",
        limit_param: str = "limit",
        limit: int = 100,
        max_pages: Optional[int] = None,
        data_key: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Get all pages from paginated endpoint.
        
        Args:
            endpoint: API endpoint
            page_param: Name of page parameter
            limit_param: Name of limit parameter
            limit: Items per page
            max_pages: Maximum pages to fetch
            data_key: Key containing data in response
            **kwargs: Additional request options
            
        Returns:
            List of all items
        """
        all_items = []
        page = 1
        
        while True:
            params = kwargs.get("params", {})
            params[page_param] = page
            params[limit_param] = limit
            kwargs["params"] = params
            
            response = self.get(endpoint, **kwargs)
            
            # Extract data
            if data_key and isinstance(response, dict):
                items = response.get(data_key, [])
            elif isinstance(response, list):
                items = response
            else:
                break
            
            if not items:
                break
            
            all_items.extend(items)
            
            # Check if we've reached the end
            if len(items) < limit:
                break
            
            if max_pages and page >= max_pages:
                break
            
            page += 1
        
        logger.info(f"Fetched paginated data", pages=page, items=len(all_items))
        return all_items


class GraphQLClient(APIClient):
    """
    GraphQL API client.
    
    Example:
        >>> client = GraphQLClient("https://api.example.com/graphql")
        >>> 
        >>> query = '''
        ...     query GetUsers($active: Boolean) {
        ...         users(active: $active) {
        ...             id
        ...             name
        ...             email
        ...         }
        ...     }
        ... '''
        >>> result = client.query(query, variables={"active": True})
    """
    
    def connect(self) -> None:
        """Initialize session."""
        try:
            import requests
            
            self._session = requests.Session()
            self._session.headers.update(self.headers)
            self._session.headers["Content-Type"] = "application/json"
            
            if isinstance(self.auth, str):
                self._session.headers["Authorization"] = f"Bearer {self.auth}"
            elif isinstance(self.auth, tuple):
                self._session.auth = self.auth
            
            self._connected = True
            
        except ImportError:
            raise DataIngestionError("requests not installed")
    
    def disconnect(self) -> None:
        """Close session."""
        if self._session:
            self._session.close()
            self._connected = False
    
    def read(self, query: str = None, **kwargs: Any) -> Any:
        """Execute GraphQL query."""
        return self.query(query, **kwargs)
    
    def write(self, data: Any, mutation: str = None, **kwargs: Any) -> None:
        """Execute GraphQL mutation."""
        self.mutate(mutation, variables=data, **kwargs)
    
    @retry(max_attempts=3, delay=1.0)
    def request(
        self,
        method: str,
        endpoint: str = "",
        **kwargs: Any,
    ) -> Any:
        """Make GraphQL request (always POST to endpoint)."""
        return self.execute(kwargs.get("query", ""), kwargs.get("variables"))
    
    def execute(
        self,
        query: str,
        variables: Optional[Dict] = None,
        operation_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query or mutation.
        
        Args:
            query: GraphQL query/mutation string
            variables: Query variables
            operation_name: Optional operation name
            
        Returns:
            Response data
        """
        if not self._connected:
            self.connect()
        
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        if operation_name:
            payload["operationName"] = operation_name
        
        try:
            response = self._session.post(
                self.base_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check for GraphQL errors
            if "errors" in result:
                raise DataIngestionError(
                    "GraphQL query returned errors",
                    context={"errors": result["errors"]},
                )
            
            return result.get("data", {})
            
        except Exception as e:
            if isinstance(e, DataIngestionError):
                raise
            raise DataIngestionError(
                "GraphQL request failed",
                context={"error": str(e)},
            )
    
    def query(
        self,
        query: str,
        variables: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query.
        
        Args:
            query: GraphQL query string
            variables: Query variables
            
        Returns:
            Query result data
        """
        return self.execute(query, variables, **kwargs)
    
    def mutate(
        self,
        mutation: str,
        variables: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute GraphQL mutation.
        
        Args:
            mutation: GraphQL mutation string
            variables: Mutation variables
            
        Returns:
            Mutation result data
        """
        return self.execute(mutation, variables, **kwargs)
