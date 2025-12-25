# Google Cloud Platform Deployment Guide

Deploy the Data Science Master System on Google Cloud Platform.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ Cloud DNS   │────▶│ Cloud Load  │────▶│  Cloud Run  │       │
│  │             │     │  Balancer   │     │             │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                │                 │
│                                                ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │    GCS      │◀───▶│ Vertex AI   │◀───▶│  Cloud SQL  │       │
│  │  (Models)   │     │(Training)   │     │ (Postgres)  │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Memorystore│     │ Cloud       │     │  Artifact   │       │
│  │   (Redis)   │     │ Monitoring  │     │  Registry   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Set Up Artifact Registry

```bash
# Enable services
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create repository
gcloud artifacts repositories create dsms-repo \
    --repository-format=docker \
    --location=us-central1

# Configure Docker auth
gcloud auth configure-docker us-central1-docker.pkg.dev
```

## Step 2: Build and Push Image

```bash
# Build with Cloud Build
gcloud builds submit \
    --tag us-central1-docker.pkg.dev/PROJECT_ID/dsms-repo/dsms-api:latest \
    --file docker/Dockerfile.prod .

# Or build locally and push
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/dsms-repo/dsms-api:latest \
    -f docker/Dockerfile.prod .
docker push us-central1-docker.pkg.dev/PROJECT_ID/dsms-repo/dsms-api:latest
```

## Step 3: Deploy to Cloud Run

```bash
# Deploy service
gcloud run deploy dsms-api \
    --image us-central1-docker.pkg.dev/PROJECT_ID/dsms-repo/dsms-api:latest \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --port 8000 \
    --allow-unauthenticated \
    --set-env-vars "LOG_LEVEL=INFO,WORKERS=4"

# Get service URL
gcloud run services describe dsms-api --region us-central1 --format 'value(status.url)'
```

## Step 4: Set Up Cloud SQL

```bash
# Create instance
gcloud sql instances create dsms-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=us-central1 \
    --storage-size=10 \
    --storage-type=SSD

# Create database
gcloud sql databases create dsms --instance=dsms-db

# Create user
gcloud sql users create dsms_user \
    --instance=dsms-db \
    --password=<secure-password>

# Get connection name
gcloud sql instances describe dsms-db --format 'value(connectionName)'
```

## Step 5: Configure Cloud Run with Cloud SQL

```bash
gcloud run services update dsms-api \
    --region us-central1 \
    --add-cloudsql-instances PROJECT_ID:us-central1:dsms-db \
    --set-env-vars "DB_HOST=/cloudsql/PROJECT_ID:us-central1:dsms-db,DB_NAME=dsms,DB_USER=dsms_user"
```

## Step 6: Cloud Storage for Models

```bash
# Create bucket
gsutil mb -l us-central1 gs://dsms-models-PROJECT_ID

# Enable versioning
gsutil versioning set on gs://dsms-models-PROJECT_ID

# Upload model
gsutil cp models/production/model.joblib gs://dsms-models-PROJECT_ID/production/

# Set IAM for Cloud Run
gcloud storage buckets add-iam-policy-binding gs://dsms-models-PROJECT_ID \
    --member="serviceAccount:PROJECT_ID@appspot.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```

## Step 7: Monitoring and Logging

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=dsms-api" \
    --limit 50

# Create alert policy
gcloud alpha monitoring policies create \
    --policy-from-file=monitoring/gcp-alerts.yaml
```

**monitoring/gcp-alerts.yaml:**
```yaml
displayName: DSMS High Error Rate
conditions:
  - displayName: Error rate > 5%
    conditionThreshold:
      filter: resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count"
      comparison: COMPARISON_GT
      thresholdValue: 0.05
      duration: 300s
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_RATE
notificationChannels:
  - projects/PROJECT_ID/notificationChannels/CHANNEL_ID
```

## Step 8: Custom Domain (Optional)

```bash
# Map custom domain
gcloud beta run domain-mappings create \
    --service dsms-api \
    --domain api.yourdomain.com \
    --region us-central1

# Get DNS records to configure
gcloud beta run domain-mappings describe \
    --domain api.yourdomain.com \
    --region us-central1
```

## Vertex AI Integration

For ML training at scale:

```bash
# Create training job
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=dsms-training \
    --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/PROJECT_ID/dsms-repo/dsms-training:latest

# Create model endpoint
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name=dsms-endpoint
```

## Cost Estimation

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| Cloud Run | 2 vCPU, 2GB, min 1 instance | ~$50 |
| Cloud SQL | db-f1-micro | ~$10 |
| Cloud Storage | 10GB | ~$1 |
| Cloud Monitoring | Standard | ~$0 |
| **Total** | | **~$61/month** |

## Best Practices

1. Use Secret Manager for credentials
2. Enable VPC connector for private access
3. Use Cloud Armor for DDoS protection
4. Implement Cloud CDN for static assets
5. Set up Cloud Build triggers for CI/CD
