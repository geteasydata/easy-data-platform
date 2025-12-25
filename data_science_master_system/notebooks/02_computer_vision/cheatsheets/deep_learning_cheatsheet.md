# 📋 Deep Learning (CV & NLP) Cheatsheet

## Computer Vision

### Image Classification
```python
import torch
import torchvision.models as models

# Pretrained model
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
model.classifier[-1] = torch.nn.Linear(1280, num_classes)  # Modify head

# Data augmentation
from torchvision import transforms
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Object Detection
```python
# YOLO
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.predict('image.jpg')
```

### Segmentation
```python
# U-Net structure
# Encoder → Bottleneck → Decoder (with skip connections)
# Loss: Dice Loss + BCE
```

## NLP & Transformers

### Text Classification
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
outputs = model(**inputs)
```

### Text Generation
```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
result = generator("The future of AI is", max_length=50)
```

### Question Answering
```python
qa = pipeline('question-answering')
result = qa(question="What is ML?", context="ML is machine learning...")
```

## Key Hyperparameters
| Domain | Parameter | Typical Values |
|--------|-----------|----------------|
| CV | learning_rate | 1e-4 to 1e-3 |
| CV | batch_size | 16, 32, 64 |
| NLP | learning_rate | 2e-5 to 5e-5 |
| NLP | epochs | 2-5 for fine-tuning |

## GPU Memory Tips
1. Use gradient checkpointing
2. Reduce batch size + gradient accumulation
3. Mixed precision (fp16)
4. Freeze early layers

## Common Pitfalls
1. Not unfreezing layers gradually
2. Learning rate too high for fine-tuning
3. Not using pretrained weights
4. Insufficient augmentation
