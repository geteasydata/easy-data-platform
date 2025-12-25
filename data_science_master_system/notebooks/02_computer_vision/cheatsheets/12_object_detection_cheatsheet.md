# 📋 Object Detection Cheatsheet

## 📌 Key Concepts
- **Object Detection**: Localize + classify objects in images
- **Bounding Box**: [x, y, width, height] or [x1, y1, x2, y2]
- **IoU**: Intersection over Union - overlap measure
- **NMS**: Non-Maximum Suppression - remove duplicate detections
- **Anchor Boxes**: Predefined box shapes for detection

## 🛠️ Essential Code

### YOLO (Ultralytics)
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pretrained
results = model.predict('image.jpg', conf=0.5)

# Results
for r in results:
    boxes = r.boxes.xyxy  # Bounding boxes
    classes = r.boxes.cls  # Class IDs
    confs = r.boxes.conf   # Confidence scores
```

### Faster R-CNN (TorchVision)
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

predictions = model([image_tensor])
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']
```

## 📊 Model Comparison
| Model | mAP | Speed (FPS) | Size |
|-------|-----|-------------|------|
| YOLOv8n | 37.3 | 100+ | 6MB |
| YOLOv8s | 44.9 | 60 | 22MB |
| YOLOv8m | 50.2 | 40 | 52MB |
| Faster R-CNN | 42.0 | 5-10 | 160MB |

## 📐 Key Formulas
```
IoU = Intersection Area / Union Area
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
mAP = Mean Average Precision across classes
```

## ⚠️ Common Pitfalls
| Problem | Solution |
|---------|----------|
| Too many false positives | Increase confidence threshold |
| Missing small objects | Use larger input size |
| Slow inference | Use smaller model variant |
| Poor accuracy | Fine-tune on domain data |

## 🚀 Production Tips
- Use TensorRT for NVIDIA GPUs
- Batch processing for throughput
- Quantize to INT8 for edge deployment
- DeepStream for video analytics
