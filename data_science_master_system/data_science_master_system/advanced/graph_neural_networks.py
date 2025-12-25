"""
Graph Neural Networks Module.

GNN implementations for node classification, link prediction, and graph classification.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

TORCH_GEOMETRIC_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    pass


class GCN(nn.Module):
    """Graph Convolutional Network for node classification."""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """Graph Attention Network."""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGE(nn.Module):
    """GraphSAGE for inductive learning."""
    
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GNNTrainer:
    """Trainer for GNN models."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.01, weight_decay: float = 5e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def train_epoch(self, data: Data, train_mask: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, data: Data, mask: torch.Tensor) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out[mask].argmax(dim=1)
            correct = (pred == data.y[mask]).sum().item()
            accuracy = correct / mask.sum().item()
            loss = F.nll_loss(out[mask], data.y[mask]).item()
        return accuracy, loss
    
    def fit(self, data: Data, train_mask: torch.Tensor, val_mask: torch.Tensor, epochs: int = 200) -> Dict:
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_acc = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(data, train_mask)
            val_acc, val_loss = self.evaluate(data, val_mask)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
            
            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")
        
        logger.info(f"Best Val Accuracy: {best_acc:.4f}")
        return history


def create_graph_from_edges(edge_list: List[Tuple[int, int]], node_features: np.ndarray, labels: np.ndarray = None) -> Data:
    """Create PyG Data object from edge list."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not installed")
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    
    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.long)
    
    return data
