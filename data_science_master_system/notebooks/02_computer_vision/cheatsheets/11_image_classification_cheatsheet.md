# 📋 Image Classification Cheatsheet

## 📌 Key Concepts
- **CNN**: Convolutional Neural Network - learns spatial features
- **Transfer Learning**: Use pretrained models, fine-tune classifier
- **Data Augmentation**: Artificially increase training data variety
- **Feature Extraction**: Freeze backbone, train only classifier
- **Fine-tuning**: Unfreeze layers gradually, lower learning rate

## 🛠️ Essential Code

### Data Augmentation
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Transfer Learning
```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # Freeze backbone
model.classifier[1] = nn.Linear(1280, num_classes)  # New head
```

### Training Loop
```python
for epoch in range(epochs):
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
```

## 📊 Model Comparison
| Model | Params | ImageNet Acc | Speed |
|-------|--------|--------------|-------|
| ResNet18 | 11M | 69.8% | Fast |
| ResNet50 | 25M | 76.1% | Medium |
| EfficientNet-B0 | 5M | 77.1% | Fast |
| EfficientNet-B4 | 19M | 82.6% | Medium |
| ViT-B/16 | 86M | 84.5% | Slow |

## ⚠️ Common Pitfalls
| Problem | Solution |
|---------|----------|
| Overfitting | More augmentation, dropout, early stopping |
| Low accuracy | Unfreeze more layers, lower LR |
| Memory errors | Reduce batch size, use gradient checkpointing |
| Slow training | Use mixed precision (fp16) |

## 🚀 Production Tips
- Export to ONNX: `torch.onnx.export(model, dummy, 'model.onnx')`
- Quantization: 4x smaller, 2x faster
- TensorRT: NVIDIA optimization
- Batch inference: Higher throughput
