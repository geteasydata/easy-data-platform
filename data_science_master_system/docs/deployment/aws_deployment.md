# AWS Deployment Guide

Deploy the Data Science Master System on Amazon Web Services.

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Docker installed locally
- Terraform (optional for IaC)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Amazon Web Services                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Route 53  │────▶│     ALB     │────▶│    ECS      │       │
│  │   (DNS)     │     │             │     │  Fargate    │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                │                 │
│                                                ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │     S3      │◀───▶│  SageMaker  │◀───▶│    RDS      │       │
│  │  (Models)   │     │  (Training) │     │ (Postgres)  │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ ElastiCache │     │ CloudWatch  │     │    ECR      │       │
│  │   (Redis)   │     │ (Monitoring)│     │  (Images)   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Set Up ECR Repository

```bash
# Create ECR repository
aws ecr create-repository \
    --repository-name dsms-api \
    --image-scanning-configuration scanOnPush=true

# Get login credentials
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push image
docker build -t dsms-api -f docker/Dockerfile.prod .
docker tag dsms-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/dsms-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/dsms-api:latest
```

## Step 2: Create ECS Cluster

```bash
# Create cluster
aws ecs create-cluster --cluster-name dsms-cluster

# Create task definition (save as task-definition.json)
```

**task-definition.json:**
```json
{
    "family": "dsms-api",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "containerDefinitions": [
        {
            "name": "dsms-api",
            "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/dsms-api:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {"name": "LOG_LEVEL", "value": "INFO"},
                {"name": "WORKERS", "value": "4"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/dsms-api",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3
            }
        }
    ]
}
```

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
    --cluster dsms-cluster \
    --service-name dsms-api-service \
    --task-definition dsms-api \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

## Step 3: Set Up RDS PostgreSQL

```bash
aws rds create-db-instance \
    --db-instance-identifier dsms-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 15 \
    --master-username dsms_admin \
    --master-user-password <secure-password> \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-xxx
```

## Step 4: Set Up S3 for Models

```bash
# Create bucket for models
aws s3 mb s3://dsms-models-<account-id>

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket dsms-models-<account-id> \
    --versioning-configuration Status=Enabled

# Upload model
aws s3 cp models/production/model.joblib s3://dsms-models-<account-id>/production/
```

## Step 5: Configure Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
    --name dsms-alb \
    --subnets subnet-xxx subnet-yyy \
    --security-groups sg-xxx

# Create target group
aws elbv2 create-target-group \
    --name dsms-targets \
    --protocol HTTP \
    --port 8000 \
    --vpc-id vpc-xxx \
    --target-type ip \
    --health-check-path /health
```

## Step 6: CloudWatch Monitoring

```bash
# Create log group
aws logs create-log-group --log-group-name /ecs/dsms-api

# Create alarm for high error rate
aws cloudwatch put-metric-alarm \
    --alarm-name dsms-high-error-rate \
    --metric-name 5XXError \
    --namespace AWS/ApplicationELB \
    --statistic Sum \
    --period 300 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:us-east-1:<account-id>:alerts
```

## Step 7: Auto Scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --resource-id service/dsms-cluster/dsms-api-service \
    --scalable-dimension ecs:service:DesiredCount \
    --min-capacity 2 \
    --max-capacity 10

# Create scaling policy
aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --resource-id service/dsms-cluster/dsms-api-service \
    --scalable-dimension ecs:service:DesiredCount \
    --policy-name cpu-scaling \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60
    }'
```

## Terraform Alternative

See `terraform/aws/` for Infrastructure as Code version.

## Cost Estimation

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| ECS Fargate | 2 tasks, 1 vCPU, 2GB | ~$60 |
| RDS | db.t3.micro | ~$15 |
| ALB | 1 ALB + LCU | ~$25 |
| S3 | 10GB storage | ~$1 |
| CloudWatch | Basic monitoring | ~$5 |
| **Total** | | **~$106/month** |

## Security Best Practices

1. Use IAM roles, not access keys
2. Enable encryption at rest for RDS and S3
3. Use VPC private subnets for databases
4. Enable WAF on ALB
5. Rotate secrets with AWS Secrets Manager
