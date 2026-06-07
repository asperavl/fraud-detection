# Fraud Detection API

Trains an XGBoost model on the IEEE-CIS dataset and serves predictions via a REST API. Training data is stored in S3 and pulled at Docker build time. Deployed on AWS EC2. GitHub Actions rebuilds and redeploys the container on every push to main.

## Architecture

```
S3 (Training Data)
       │
       ▼
Docker Build (GitHub Actions)
       │
       ├── Pull data from S3
       ├── Train XGBoost model
       └── Package FastAPI app
       │
       ▼
Docker Hub
       │
       ▼
AWS EC2
       │
       ▼
POST /predict → fraud_probability
```

## Model Performance

Trained on 590,540 transactions with 3.5% fraud rate.

| Metric | Fraud Class |
|--------|------------|
| Precision | 0.42 |
| Recall | 0.82 |
| F1 | 0.56 |

Recall is prioritized over precision — missing fraud is more costly than a false alarm.

## API

`POST /predict`

**Request:**
```json
{
  "data": {
    "TransactionAmt": 100.0,
    "ProductCD": 1,
    "card1": 9500
  }
}
```

**Response:**
```json
{
  "fraud_probability": 0.87,
  "verdict": "Fraudulent"
}
```

## Stack
Python, XGBoost, FastAPI, Docker, AWS EC2, AWS S3, GitHub Actions

## Improvements
- Replace EC2 with SageMaker for managed scaling
- Threshold tuning to improve precision
- Request logging and input validation