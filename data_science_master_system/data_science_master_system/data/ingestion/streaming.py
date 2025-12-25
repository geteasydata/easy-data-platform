"""
Streaming Data Consumers for Data Science Master System.

Provides consumers for:
- Apache Kafka
- RabbitMQ
- AWS Kinesis
- Google Pub/Sub

Example:
    >>> consumer = KafkaConsumer(
    ...     bootstrap_servers=["localhost:9092"],
    ...     topic="events",
    ...     group_id="my-consumer-group",
    ... )
    >>> for message in consumer.consume():
    ...     process(message)
"""

from abc import abstractmethod
from typing import Any, Callable, Dict, Generator, List, Optional
import json
import time

from data_science_master_system.core.base_classes import BaseDataSource
from data_science_master_system.core.exceptions import DataIngestionError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class StreamingConsumer(BaseDataSource):
    """
    Abstract base class for streaming consumers.
    """
    
    def __init__(
        self,
        topic: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize streaming consumer.
        
        Args:
            topic: Topic/queue to consume from
            config: Additional configuration
        """
        super().__init__(config)
        self.topic = topic
        self._running = False
    
    @abstractmethod
    def consume(
        self,
        callback: Optional[Callable[[Any], None]] = None,
        batch_size: int = 1,
        timeout: Optional[float] = None,
    ) -> Generator[Any, None, None]:
        """
        Consume messages from the stream.
        
        Args:
            callback: Optional callback for each message
            batch_size: Number of messages to batch
            timeout: Timeout in seconds
            
        Yields:
            Messages from the stream
        """
        pass
    
    @abstractmethod
    def commit(self) -> None:
        """Commit consumed messages."""
        pass
    
    def stop(self) -> None:
        """Stop consuming."""
        self._running = False


class KafkaConsumer(StreamingConsumer):
    """
    Apache Kafka consumer.
    
    Example:
        >>> consumer = KafkaConsumer(
        ...     bootstrap_servers=["localhost:9092"],
        ...     topic="events",
        ...     group_id="my-group",
        ... )
        >>> consumer.connect()
        >>> 
        >>> # Consume messages
        >>> for msg in consumer.consume(batch_size=100, timeout=5.0):
        ...     print(msg)
        >>> 
        >>> # With callback
        >>> def process(msg):
        ...     save_to_database(msg)
        >>> consumer.consume(callback=process)
    """
    
    def __init__(
        self,
        bootstrap_servers: List[str],
        topic: str,
        group_id: str,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize Kafka consumer.
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            topic: Topic to consume from
            group_id: Consumer group ID
            auto_offset_reset: Where to start ("earliest", "latest")
            enable_auto_commit: Whether to auto-commit offsets
            config: Additional Kafka consumer config
        """
        super().__init__(topic, config)
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self._consumer = None
    
    def connect(self) -> None:
        """Connect to Kafka."""
        try:
            from kafka import KafkaConsumer as KC
            
            consumer_config = {
                "bootstrap_servers": self.bootstrap_servers,
                "group_id": self.group_id,
                "auto_offset_reset": self.auto_offset_reset,
                "enable_auto_commit": self.enable_auto_commit,
                "value_deserializer": lambda x: json.loads(x.decode("utf-8")),
                **(self.config or {}),
            }
            
            self._consumer = KC(self.topic, **consumer_config)
            self._connected = True
            logger.info(
                f"Connected to Kafka",
                servers=self.bootstrap_servers,
                topic=self.topic,
            )
            
        except ImportError:
            raise DataIngestionError("kafka-python not installed")
        except Exception as e:
            raise DataIngestionError(f"Kafka connection failed: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._consumer:
            self._consumer.close()
            self._connected = False
    
    def read(self, **kwargs: Any) -> Any:
        """Read messages (returns generator)."""
        return list(self.consume(**kwargs))
    
    def write(self, data: Any, **kwargs: Any) -> None:
        """Not applicable for consumer."""
        raise NotImplementedError("Use KafkaProducer for writing")
    
    def consume(
        self,
        callback: Optional[Callable[[Any], None]] = None,
        batch_size: int = 1,
        timeout: Optional[float] = None,
    ) -> Generator[Any, None, None]:
        """
        Consume messages from Kafka.
        
        Args:
            callback: Optional callback for each message
            batch_size: Number of messages to yield as batch
            timeout: Poll timeout in seconds
            
        Yields:
            Message values or batches
        """
        if not self._connected:
            self.connect()
        
        self._running = True
        batch = []
        start_time = time.time()
        
        while self._running:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Poll for messages
            records = self._consumer.poll(
                timeout_ms=1000,
                max_records=batch_size,
            )
            
            for topic_partition, messages in records.items():
                for message in messages:
                    value = message.value
                    
                    if callback:
                        callback(value)
                    
                    if batch_size == 1:
                        yield value
                    else:
                        batch.append(value)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
        
        # Yield remaining batch
        if batch:
            yield batch
    
    def commit(self) -> None:
        """Commit consumed offsets."""
        if self._consumer:
            self._consumer.commit()


class KafkaProducer:
    """
    Apache Kafka producer.
    
    Example:
        >>> producer = KafkaProducer(bootstrap_servers=["localhost:9092"])
        >>> producer.connect()
        >>> producer.send("my-topic", {"event": "user_signup", "user_id": 123})
    """
    
    def __init__(
        self,
        bootstrap_servers: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.config = config or {}
        self._producer = None
    
    def connect(self) -> None:
        """Connect to Kafka."""
        try:
            from kafka import KafkaProducer as KP
            
            producer_config = {
                "bootstrap_servers": self.bootstrap_servers,
                "value_serializer": lambda x: json.dumps(x).encode("utf-8"),
                **self.config,
            }
            
            self._producer = KP(**producer_config)
            logger.info(f"Kafka producer connected", servers=self.bootstrap_servers)
            
        except ImportError:
            raise DataIngestionError("kafka-python not installed")
        except Exception as e:
            raise DataIngestionError(f"Kafka producer connection failed: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._producer:
            self._producer.close()
    
    def send(
        self,
        topic: str,
        value: Any,
        key: Optional[str] = None,
        partition: Optional[int] = None,
    ) -> None:
        """
        Send message to Kafka topic.
        
        Args:
            topic: Topic name
            value: Message value
            key: Optional message key
            partition: Optional partition number
        """
        if not self._producer:
            self.connect()
        
        self._producer.send(
            topic,
            value=value,
            key=key.encode("utf-8") if key else None,
            partition=partition,
        )
    
    def flush(self) -> None:
        """Flush pending messages."""
        if self._producer:
            self._producer.flush()


class RabbitMQConsumer(StreamingConsumer):
    """
    RabbitMQ consumer.
    
    Example:
        >>> consumer = RabbitMQConsumer(
        ...     host="localhost",
        ...     queue="my-queue",
        ... )
        >>> consumer.connect()
        >>> for message in consumer.consume():
        ...     process(message)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        queue: str = "",
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(queue, config)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self._connection = None
        self._channel = None
    
    def connect(self) -> None:
        """Connect to RabbitMQ."""
        try:
            import pika
            
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=credentials,
            )
            
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            
            # Declare queue
            self._channel.queue_declare(queue=self.topic, durable=True)
            
            self._connected = True
            logger.info(f"Connected to RabbitMQ", host=self.host, queue=self.topic)
            
        except ImportError:
            raise DataIngestionError("pika not installed")
        except Exception as e:
            raise DataIngestionError(f"RabbitMQ connection failed: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self._connection:
            self._connection.close()
            self._connected = False
    
    def read(self, **kwargs: Any) -> Any:
        """Read messages."""
        return list(self.consume(**kwargs))
    
    def write(self, data: Any, **kwargs: Any) -> None:
        """Publish message to queue."""
        if not self._connected:
            self.connect()
        
        import pika
        
        body = json.dumps(data) if isinstance(data, dict) else str(data)
        self._channel.basic_publish(
            exchange="",
            routing_key=self.topic,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )
    
    def consume(
        self,
        callback: Optional[Callable[[Any], None]] = None,
        batch_size: int = 1,
        timeout: Optional[float] = None,
    ) -> Generator[Any, None, None]:
        """Consume messages from RabbitMQ."""
        if not self._connected:
            self.connect()
        
        self._running = True
        batch = []
        start_time = time.time()
        
        while self._running:
            if timeout and (time.time() - start_time) > timeout:
                break
            
            method, properties, body = self._channel.basic_get(
                queue=self.topic,
                auto_ack=True,
            )
            
            if body:
                try:
                    value = json.loads(body.decode("utf-8"))
                except:
                    value = body.decode("utf-8")
                
                if callback:
                    callback(value)
                
                if batch_size == 1:
                    yield value
                else:
                    batch.append(value)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            else:
                time.sleep(0.1)
        
        if batch:
            yield batch
    
    def commit(self) -> None:
        """Commit not needed for auto_ack."""
        pass
