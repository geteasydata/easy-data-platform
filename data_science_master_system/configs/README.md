# Configuration Files

Environment and model configurations.

## Structure

```
configs/
├── environments/     # Dev/Test/Prod configs
│   ├── dev.yaml
│   ├── test.yaml
│   └── prod.yaml
├── models/           # Model hyperparameters
│   ├── classification.yaml
│   └── regression.yaml
└── deployments/      # Deployment configs
    ├── docker.yaml
    └── kubernetes.yaml
```
