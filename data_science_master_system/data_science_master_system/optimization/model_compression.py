"""
Model Compression Module.

Implements quantization, pruning, and distillation for edge deployment.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class ModelQuantizer:
    """Quantize models for faster inference and smaller size."""
    
    def __init__(self, bits: int = 8, mode: str = 'dynamic'):
        self.bits = bits
        self.mode = mode
    
    def quantize_pytorch(self, model, calibration_data=None):
        """Quantize PyTorch model."""
        try:
            import torch
            
            if self.mode == 'dynamic':
                quantized = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            elif self.mode == 'static':
                model.eval()
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                if calibration_data is not None:
                    with torch.no_grad():
                        for data in calibration_data:
                            model(data)
                quantized = torch.quantization.convert(model, inplace=False)
            
            orig_size = sum(p.numel() * 4 for p in model.parameters()) / 1e6
            quant_size = sum(p.numel() for p in quantized.parameters()) / 1e6
            
            logger.info(f"Quantized: {orig_size:.1f}MB → {quant_size:.1f}MB ({self.mode})")
            return quantized
        except ImportError:
            raise ImportError("PyTorch required for quantization")
    
    def export_to_onnx_quantized(self, model, input_shape, output_path):
        """Export quantized model to ONNX."""
        try:
            import torch
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            dummy_input = torch.randn(*input_shape)
            temp_path = output_path.replace('.onnx', '_fp32.onnx')
            
            torch.onnx.export(model, dummy_input, temp_path)
            quantize_dynamic(temp_path, output_path, weight_type=QuantType.QUInt8)
            
            logger.info(f"Exported quantized ONNX to {output_path}")
        except ImportError:
            raise ImportError("onnxruntime required")


class ModelPruner:
    """Prune model weights for compression."""
    
    def __init__(self, sparsity: float = 0.5, method: str = 'magnitude'):
        self.sparsity = sparsity
        self.method = method
    
    def prune_pytorch(self, model, layers_to_prune=None):
        """Prune PyTorch model."""
        try:
            import torch
            import torch.nn.utils.prune as prune
            
            params_before = sum(p.numel() for p in model.parameters())
            
            for name, module in model.named_modules():
                if layers_to_prune and name not in layers_to_prune:
                    continue
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    if self.method == 'magnitude':
                        prune.l1_unstructured(module, name='weight', amount=self.sparsity)
                    elif self.method == 'random':
                        prune.random_unstructured(module, name='weight', amount=self.sparsity)
            
            # Make pruning permanent
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    if hasattr(module, 'weight_mask'):
                        prune.remove(module, 'weight')
            
            non_zero = sum((p != 0).sum().item() for p in model.parameters())
            actual_sparsity = 1 - (non_zero / params_before)
            
            logger.info(f"Pruned to {actual_sparsity:.1%} sparsity")
            return model
        except ImportError:
            raise ImportError("PyTorch required")


class KnowledgeDistiller:
    """Distill knowledge from large to small models."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha
    
    def distill(self, teacher, student, train_loader, epochs: int = 10, lr: float = 0.001):
        """Train student model using teacher knowledge."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            device = next(teacher.parameters()).device
            student = student.to(device)
            teacher.eval()
            
            optimizer = torch.optim.Adam(student.parameters(), lr=lr)
            
            for epoch in range(epochs):
                student.train()
                total_loss = 0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    with torch.no_grad():
                        teacher_logits = teacher(inputs)
                    student_logits = student(inputs)
                    
                    # Distillation loss
                    soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
                    soft_pred = F.log_softmax(student_logits / self.temperature, dim=1)
                    distill_loss = F.kl_div(soft_pred, soft_targets, reduction='batchmean') * (self.temperature ** 2)
                    
                    # Hard label loss
                    hard_loss = F.cross_entropy(student_logits, labels)
                    
                    loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                logger.info(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")
            
            return student
        except ImportError:
            raise ImportError("PyTorch required")


def compare_model_sizes(models: Dict[str, Any]) -> Dict:
    """Compare sizes of multiple models."""
    results = {}
    for name, model in models.items():
        try:
            import torch
            params = sum(p.numel() for p in model.parameters())
            size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
            results[name] = {'parameters': params, 'size_mb': round(size_mb, 2)}
        except:
            results[name] = {'error': 'Cannot compute size'}
    return results
