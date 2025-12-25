"""
Streaming Module for Real-time Data Processing.

Provides real-time data processing capabilities:
- Kafka streaming
- Event processing
- Micro-batch processing
"""

from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import json
import time

from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class StreamProcessor:
    """Real-time stream processing engine."""
    
    def __init__(self, window_size: int = 100, window_time: float = 60.0):
        self.window_size = window_size
        self.window_time = window_time
        self.buffer: List[Dict] = []
        self.processors: List[Callable] = []
        self.window_start = datetime.now()
    
    def add_processor(self, processor: Callable):
        self.processors.append(processor)
    
    def process(self, event: Dict):
        self.buffer.append({'event': event, 'timestamp': datetime.now()})
        if len(self.buffer) >= self.window_size or self._window_expired():
            self._flush()
    
    def _window_expired(self) -> bool:
        return (datetime.now() - self.window_start).seconds >= self.window_time
    
    def _flush(self):
        if not self.buffer:
            return
        events = [e['event'] for e in self.buffer]
        for processor in self.processors:
            events = processor(events)
        self.buffer.clear()
        self.window_start = datetime.now()
        logger.info(f"Processed {len(events)} events")


class FeatureAggregator:
    """Aggregate features from streaming data."""
    
    def __init__(self):
        self.aggregations: Dict[str, Dict] = {}
    
    def update(self, entity_id: str, features: Dict):
        if entity_id not in self.aggregations:
            self.aggregations[entity_id] = {'count': 0, 'sum': {}, 'last_updated': None}
        agg = self.aggregations[entity_id]
        agg['count'] += 1
        for k, v in features.items():
            if isinstance(v, (int, float)):
                agg['sum'][k] = agg['sum'].get(k, 0) + v
        agg['last_updated'] = datetime.now()
    
    def get_features(self, entity_id: str) -> Dict:
        if entity_id not in self.aggregations:
            return {}
        agg = self.aggregations[entity_id]
        return {
            'count': agg['count'],
            **{f'{k}_sum': v for k, v in agg['sum'].items()},
            **{f'{k}_avg': v / agg['count'] for k, v in agg['sum'].items()},
        }
