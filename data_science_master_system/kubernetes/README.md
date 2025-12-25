# Kubernetes Manifests

Kubernetes deployment configurations.

## Structure

```
kubernetes/
├── deployments/      # Deployment manifests
│   ├── api.yaml
│   └── worker.yaml
├── services/         # Service definitions
│   ├── api-service.yaml
│   └── redis-service.yaml
└── ingress/          # Ingress rules
    └── api-ingress.yaml
```

## Deploy

```bash
kubectl apply -f deployments/
kubectl apply -f services/
kubectl apply -f ingress/
```
