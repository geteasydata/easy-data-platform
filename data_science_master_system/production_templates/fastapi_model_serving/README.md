# FastAPI Model Serving Template

Production-ready ML model serving API with monitoring, scaling, and deployment configurations.

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --port 8000

# Access API docs
open http://localhost:8000/docs
```

### Docker
```bash
# Build
docker build -f docker/Dockerfile -t ml-api .

# Run
docker run -p 8000:8000 ml-api

# Or use Docker Compose
cd docker && docker-compose up
```

### Kubernetes
```bash
# Deploy
kubectl apply -f kubernetes/

# Check status
kubectl get pods -l app=ml-api
```

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/docs` | GET | Swagger UI |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch prediction |
| `/api/v1/models` | GET | List models |
| `/api/v1/models/{name}` | GET | Model info |
| `/metrics` | GET | Prometheus metrics |

## 📦 Project Structure

```
fastapi_model_serving/
├── app/
│   ├── main.py          # FastAPI application
│   ├── models.py        # Model manager
│   ├── schemas.py       # Pydantic schemas
│   ├── routes.py        # API routes
│   └── config.py        # Configuration
├── docker/
│   ├── Dockerfile       # Production image
│   └── docker-compose.yml
├── kubernetes/
│   ├── deployment.yaml  # K8s deployment + HPA
│   └── service.yaml     # Service + Ingress
├── monitoring/
│   └── prometheus.yml   # Metrics config
├── models/              # Model files (.joblib)
├── tests/               # API tests
└── requirements.txt
```

## 🔧 Configuration

Environment variables:
- `DEBUG`: Enable debug mode (default: false)
- `LOG_LEVEL`: Logging level (default: INFO)
- `MODELS_DIR`: Models directory path
- `API_KEY`: Optional API key for auth

## 📊 Monitoring

- Prometheus metrics at `/metrics`
- Grafana dashboard included
- Request latency, count, errors tracked

## 🔒 Security Features

- Input validation with Pydantic
- Rate limiting (configurable)
- CORS configuration
- Non-root Docker user
- TLS termination via Ingress

## 📈 Scaling

- HPA configured for CPU-based autoscaling
- 2-10 replicas by default
- Resource limits defined
