# Azure Deployment Guide

Deploy the Data Science Master System on Microsoft Azure.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Microsoft Azure                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Azure DNS  │────▶│ App Gateway │────▶│  Container  │       │
│  │             │     │     /LB     │     │    Apps     │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                │                 │
│                                                ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Blob      │◀───▶│  Azure ML   │◀───▶│  Azure SQL  │       │
│  │  Storage    │     │             │     │   /Postgres │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ Azure Cache │     │   Azure     │     │  Container  │       │
│  │ for Redis   │     │   Monitor   │     │  Registry   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Set subscription
az account set --subscription <subscription-id>
```

## Step 1: Create Resource Group

```bash
# Create resource group
az group create \
    --name dsms-rg \
    --location eastus
```

## Step 2: Set Up Container Registry

```bash
# Create ACR
az acr create \
    --resource-group dsms-rg \
    --name dsmsregistry \
    --sku Basic \
    --admin-enabled true

# Get credentials
az acr credential show --name dsmsregistry

# Login to ACR
az acr login --name dsmsregistry

# Build and push image
az acr build \
    --registry dsmsregistry \
    --image dsms-api:latest \
    --file docker/Dockerfile.prod .
```

## Step 3: Deploy to Container Apps

```bash
# Create Container Apps environment
az containerapp env create \
    --name dsms-env \
    --resource-group dsms-rg \
    --location eastus

# Deploy container app
az containerapp create \
    --name dsms-api \
    --resource-group dsms-rg \
    --environment dsms-env \
    --image dsmsregistry.azurecr.io/dsms-api:latest \
    --registry-server dsmsregistry.azurecr.io \
    --registry-username <username> \
    --registry-password <password> \
    --target-port 8000 \
    --ingress external \
    --cpu 1.0 \
    --memory 2.0Gi \
    --min-replicas 1 \
    --max-replicas 10 \
    --env-vars LOG_LEVEL=INFO WORKERS=4

# Get app URL
az containerapp show \
    --name dsms-api \
    --resource-group dsms-rg \
    --query properties.configuration.ingress.fqdn
```

## Step 4: Set Up Azure Database for PostgreSQL

```bash
# Create server
az postgres flexible-server create \
    --resource-group dsms-rg \
    --name dsms-db-server \
    --location eastus \
    --admin-user dsms_admin \
    --admin-password <secure-password> \
    --sku-name Standard_B1ms \
    --tier Burstable \
    --storage-size 32 \
    --version 15

# Create database
az postgres flexible-server db create \
    --resource-group dsms-rg \
    --server-name dsms-db-server \
    --database-name dsms

# Allow Azure services
az postgres flexible-server firewall-rule create \
    --resource-group dsms-rg \
    --name dsms-db-server \
    --rule-name AllowAzure \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0
```

## Step 5: Set Up Blob Storage

```bash
# Create storage account
az storage account create \
    --name dsmsmodels \
    --resource-group dsms-rg \
    --location eastus \
    --sku Standard_LRS

# Create container
az storage container create \
    --name models \
    --account-name dsmsmodels

# Upload model
az storage blob upload \
    --account-name dsmsmodels \
    --container-name models \
    --name production/model.joblib \
    --file models/production/model.joblib
```

## Step 6: Configure Environment Variables

```bash
# Update container app with secrets
az containerapp secret set \
    --name dsms-api \
    --resource-group dsms-rg \
    --secrets db-password=<password>

# Update with environment variables
az containerapp update \
    --name dsms-api \
    --resource-group dsms-rg \
    --set-env-vars \
        "DB_HOST=dsms-db-server.postgres.database.azure.com" \
        "DB_NAME=dsms" \
        "DB_USER=dsms_admin" \
        "DB_PASSWORD=secretref:db-password" \
        "AZURE_STORAGE_ACCOUNT=dsmsmodels"
```

## Step 7: Monitoring with Azure Monitor

```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
    --resource-group dsms-rg \
    --workspace-name dsms-logs

# Enable diagnostics
az containerapp logs show \
    --name dsms-api \
    --resource-group dsms-rg \
    --type system

# Create alert rule
az monitor metrics alert create \
    --name dsms-high-error-rate \
    --resource-group dsms-rg \
    --scopes /subscriptions/<sub-id>/resourceGroups/dsms-rg/providers/Microsoft.App/containerApps/dsms-api \
    --condition "avg Requests where StatusCode >= 500 > 10" \
    --window-size 5m \
    --evaluation-frequency 1m \
    --action-group dsms-alerts
```

## Step 8: Auto-Scaling Rules

```bash
# Scale based on HTTP requests
az containerapp update \
    --name dsms-api \
    --resource-group dsms-rg \
    --scale-rule-name http-rule \
    --scale-rule-type http \
    --scale-rule-http-concurrency 50
```

## Azure ML Integration

For enterprise ML capabilities:

```bash
# Create Azure ML workspace
az ml workspace create \
    --name dsms-ml \
    --resource-group dsms-rg

# Register model
az ml model create \
    --name churn-model \
    --version 1 \
    --path models/production/model.joblib \
    --resource-group dsms-rg \
    --workspace-name dsms-ml

# Create online endpoint
az ml online-endpoint create \
    --name dsms-endpoint \
    --resource-group dsms-rg \
    --workspace-name dsms-ml
```

## Cost Estimation

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| Container Apps | 1 vCPU, 2GB | ~$50 |
| PostgreSQL | Standard_B1ms | ~$15 |
| Blob Storage | 10GB LRS | ~$1 |
| Azure Monitor | Basic | ~$5 |
| **Total** | | **~$71/month** |

## Best Practices

1. Use Azure Key Vault for secrets
2. Enable Azure AD authentication
3. Use Private Endpoints for database
4. Enable Azure DDoS Protection
5. Set up Azure DevOps pipelines for CI/CD
