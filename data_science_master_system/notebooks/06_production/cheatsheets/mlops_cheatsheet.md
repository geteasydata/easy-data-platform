# 📋 MLOps & Deployment Cheatsheet

## Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

## FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(req: PredictRequest):
    return {"prediction": model.predict([req.features]).tolist()}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

## MLflow

```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

## Cloud Deployment Commands

### AWS
```bash
aws ecr create-repository --repository-name ml-api
docker push <ecr-uri>/ml-api:latest
aws ecs create-service --cluster ml --service-name api
```

### GCP
```bash
gcloud run deploy ml-api --source . --region us-central1
```

### Azure
```bash
az containerapp create --name ml-api --image <acr>/ml-api:latest
```

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    spec:
      containers:
      - name: api
        image: ml-api:latest
        ports:
        - containerPort: 8000
```

## Monitoring

```python
from prometheus_client import Counter, Histogram

predictions = Counter('predictions_total', 'Total predictions')
latency = Histogram('prediction_latency', 'Latency')

@latency.time()
def predict(x):
    predictions.inc()
    return model.predict(x)
```

## Cost Estimates (Monthly)

| Service | Low Traffic | High Traffic |
|---------|-------------|--------------|
| AWS Lambda | $5 | $50 |
| AWS ECS | $30 | $150 |
| GCP Cloud Run | $10 | $60 |
| Azure Container Apps | $20 | $100 |
