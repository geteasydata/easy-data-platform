# Utility Scripts

Automation scripts for setup, deployment, and monitoring.

## Structure

```
scripts/
├── setup/            # Environment setup
│   ├── install.sh
│   └── install.ps1
├── deployment/       # Deployment automation
│   ├── deploy.sh
│   └── rollback.sh
└── monitoring/       # Monitoring utilities
    ├── health_check.py
    └── log_analyzer.py
```

## Usage

```bash
# Setup
./scripts/setup/install.sh

# Deploy
./scripts/deployment/deploy.sh production

# Check health
python scripts/monitoring/health_check.py
```
