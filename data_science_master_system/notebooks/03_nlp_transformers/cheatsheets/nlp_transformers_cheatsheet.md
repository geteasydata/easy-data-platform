# 📋 NLP Transformers Cheatsheet

## 📌 Key Concepts
- **Tokenization**: Text → Token IDs
- **Attention**: Weighing importance of different tokens
- **Embeddings**: Dense vector representations
- **Pre-training**: Train on large corpus (masked LM, next sentence)
- **Fine-tuning**: Adapt to specific task with labeled data

## 🛠️ Essential Code

### Text Classification
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
```

### Pipeline (Easy Mode)
```python
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.99}]

qa = pipeline('question-answering')
result = qa(question="What is ML?", context="ML is machine learning...")
```

### Training
```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16
)
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val)
trainer.train()
```

## 📊 Model Comparison
| Model | Params | GLUE Score | Speed |
|-------|--------|------------|-------|
| DistilBERT | 66M | 79.0 | 2x faster |
| BERT-base | 110M | 82.1 | 1x |
| RoBERTa | 125M | 86.4 | 1x |
| DeBERTa | 134M | 88.8 | 0.8x |

## 📐 Attention Formula
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

## ⚠️ Common Pitfalls
| Problem | Solution |
|---------|----------|
| OOM Error | Reduce batch size, use gradient accumulation |
| Slow training | Use DistilBERT, fp16 training |
| Tokenization mismatch | Use same tokenizer for train/inference |
| Poor results | More data, longer training, try RoBERTa |

## 🚀 Production Tips
- ONNX export for faster inference
- Distillation for smaller models
- Quantization (INT8) for edge
- Batch requests for throughput
