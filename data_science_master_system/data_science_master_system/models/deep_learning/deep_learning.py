"""
Deep Learning Module - PyTorch, TensorFlow, Transformers support.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path

from data_science_master_system.core.base_classes import BaseModel
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

PYTORCH_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass


class DeepLearningModel(BaseModel):
    """Unified deep learning model wrapper."""
    
    def __init__(
        self,
        framework: str = 'pytorch',
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = 'auto',
    ) -> None:
        super().__init__()
        self.framework = framework.lower()
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = 'cuda' if device == 'auto' and PYTORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
    
    def _build_pytorch_mlp(self, input_dim: int, output_dim: int):
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def fit(self, X, y, validation_split: float = 0.1, **kwargs):
        X, y = np.asarray(X), np.asarray(y)
        input_dim, output_dim = X.shape[1], len(np.unique(y))
        
        if self.framework == 'pytorch':
            self.model = self._build_pytorch_mlp(input_dim, output_dim).to(self.device)
            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.LongTensor(y).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            for epoch in range(self.epochs):
                self.model.train()
                optimizer.zero_grad()
                loss = criterion(self.model(X_t), y_t)
                loss.backward()
                optimizer.step()
                self.history['train_loss'].append(loss.item())
        return self
    
    def predict(self, X) -> np.ndarray:
        X = np.asarray(X)
        if self.framework == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.FloatTensor(X).to(self.device)).argmax(1).cpu().numpy()
        return np.array([])
    
    def save(self, path: str):
        if self.framework == 'pytorch':
            torch.save(self.model.state_dict(), path)
