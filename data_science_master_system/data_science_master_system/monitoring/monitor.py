"""
Monitoring Module for Production ML Systems.

Provides observability for ML models:
- Model performance monitoring
- Data drift detection
- Prediction logging
- Alerting
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
from collections import deque

from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class ModelMonitor:
    """Monitor ML model performance in production."""
    
    def __init__(self, baseline_metrics: Dict, alert_threshold: float = 0.1):
        self.baseline = baseline_metrics
        self.alert_threshold = alert_threshold
        self.predictions = deque(maxlen=10000)
        self.alerts: List[Dict] = []
    
    def log_prediction(self, prediction, actual=None, features=None):
        self.predictions.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual,
            'features': features
        })
    
    def check_performance(self, window_size: int = 100) -> Dict:
        outcomes = [p for p in list(self.predictions)[-window_size:] if p['actual'] is not None]
        if len(outcomes) < 10:
            return {'status': 'insufficient_data'}
        
        correct = sum(1 for p in outcomes if p['prediction'] == p['actual'])
        accuracy = correct / len(outcomes)
        drift = self.baseline.get('accuracy', 1.0) - accuracy
        
        status = 'alert' if drift > self.alert_threshold else 'ok'
        if status == 'alert':
            self.alerts.append({'time': datetime.now(), 'drift': drift})
        
        return {'status': status, 'accuracy': accuracy, 'drift': drift}


class DriftDetector:
    """Detect data drift using statistical tests."""
    
    def __init__(self, reference_data: np.ndarray):
        self.reference = reference_data
        self.reference_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, data: np.ndarray) -> Dict:
        return {'mean': np.mean(data, axis=0), 'std': np.std(data, axis=0)}
    
    def detect_drift(self, current_data: np.ndarray, threshold: float = 2.0) -> Dict:
        current_stats = self._compute_stats(current_data)
        z_scores = np.abs(current_stats['mean'] - self.reference_stats['mean']) / (self.reference_stats['std'] + 1e-8)
        drifted = z_scores > threshold
        return {
            'has_drift': bool(np.any(drifted)),
            'drifted_features': np.where(drifted)[0].tolist(),
            'z_scores': z_scores.tolist()
        }


class AlertManager:
    """Manage and send alerts for ML systems."""
    
    def __init__(self, channels: List[str] = None):
        self.channels = channels or ['log']
        self.alerts: List[Dict] = []
    
    def send_alert(self, severity: str, message: str, context: Dict = None):
        alert = {
            'timestamp': datetime.now(),
            'severity': severity,
            'message': message,
            'context': context or {}
        }
        self.alerts.append(alert)
        
        if 'log' in self.channels:
            logger.warning(f"[ALERT:{severity}] {message}")
    
    def get_recent_alerts(self, limit: int = 100) -> List[Dict]:
        return self.alerts[-limit:]
