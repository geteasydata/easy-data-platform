# 📚 Data Science Master System - Notebook Curriculum

Complete learning path from beginner to expert across all data science domains.

## 🎓 Learning Path Overview

| Level | Notebooks | Focus |
|-------|-----------|-------|
| **Beginner** | 01-04 | Core ML fundamentals |
| **Intermediate** | 01-04 | Real-world pipelines |
| **Advanced** | 01 | MLOps & Production |
| **Specialized** | 13-25 | Domain expertise |

---

## 📓 Beginner Notebooks (Foundations)

### 01 - Getting Started
- **File**: `beginner/01_getting_started.ipynb`
- **Time**: 30 min | **Difficulty**: ⭐
- **Topics**: First ML pipeline in 3 lines
- **Prerequisites**: Python basics

### 02 - Data Loading
- **Time**: 45 min | **Difficulty**: ⭐
- **Topics**: DataLoader, EDA, validation

### 03 - Feature Engineering
- **Time**: 45 min | **Difficulty**: ⭐⭐
- **Topics**: FeatureFactory, transformers

### 04 - Model Comparison
- **Time**: 45 min | **Difficulty**: ⭐⭐
- **Topics**: ModelFactory, AutoSelector

---

## 📗 Intermediate Notebooks (Real-World)

### 01 - Classification Pipeline
- **Time**: 60 min | **Difficulty**: ⭐⭐⭐
- **Topics**: Customer churn end-to-end

### 02 - Regression Pipeline
- **Time**: 60 min | **Difficulty**: ⭐⭐⭐
- **Topics**: House price prediction

### 03 - Time Series
- **Time**: 60 min | **Difficulty**: ⭐⭐⭐
- **Topics**: Sales forecasting

### 04 - Text Classification
- **Time**: 60 min | **Difficulty**: ⭐⭐⭐
- **Topics**: Sentiment analysis with TF-IDF

---

## 📕 Advanced Notebooks (Production)

### 01 - Production MLOps
- **Time**: 90 min | **Difficulty**: ⭐⭐⭐⭐⭐
- **Topics**: Feature stores, model registry, A/B testing

---

## 🔬 Specialized Notebooks

### Computer Vision
| # | Notebook | Topics | GPU |
|---|----------|--------|-----|
| 13 | Image Classification | CNN, EfficientNet, Transfer Learning | ✅ |
| 14 | Object Detection | YOLO, Faster R-CNN | ✅ |
| 15 | Image Segmentation | U-Net, Mask R-CNN, Detectron2 | ✅ |
| 16 | GANs | VAE, StyleGAN, CycleGAN | ✅ |

### NLP & Transformers
| # | Notebook | Topics | GPU |
|---|----------|--------|-----|
| 17 | Transformer Classification | BERT, RoBERTa, DistilBERT | ✅ |
| 18 | Text Generation | GPT-2, T5, LLMs | ✅ |
| 19 | Question Answering | Extractive & Generative QA | ✅ |
| 20 | Advanced Sentiment | ABSA, Multilingual, Emotion | ✅ |

### Time Series & Analytics
| # | Notebook | Topics | GPU |
|---|----------|--------|-----|
| 21 | Advanced Time Series | Prophet, N-BEATS, TFT | Optional |
| 22 | Anomaly Detection | Online learning, Streaming | ❌ |
| 23 | Recommender Systems | NCF, Graph embeddings | Optional |

### Advanced Topics
| # | Notebook | Topics | GPU |
|---|----------|--------|-----|
| 24 | Graph Neural Networks | GCN, GAT, PyG | Optional |
| 25 | Reinforcement Learning | DQN, Policy Gradients, PPO | Optional |

---

## 💻 Hardware Requirements

### CPU Only
- Beginner & Intermediate notebooks
- Time series, anomaly detection
- Basic RL

### GPU Recommended (4GB+ VRAM)
- Computer Vision (13-16)
- NLP/Transformers (17-20)
- Deep Learning models

### GPU Required (8GB+ VRAM)
- Large transformer models
- StyleGAN training
- Full GNN datasets

---

## 🚀 Quick Start

```bash
# 1. Generate sample data
python examples/data/generate_sample_data.py

# 2. Start Jupyter
jupyter lab examples/notebooks/

# 3. Begin learning path
# Open: beginner/01_getting_started.ipynb
```

---

## 📊 Expected Outcomes

After completing all notebooks, you will be able to:

- ✅ Build end-to-end ML pipelines
- ✅ Deploy models to production
- ✅ Implement CV, NLP, Time Series solutions
- ✅ Design MLOps workflows
- ✅ Work with graph and streaming data
- ✅ Apply RL to real-world problems
