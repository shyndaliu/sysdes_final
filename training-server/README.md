# Training Server

This container trains a collaborative filtering model using data from ClickHouse.

## Prerequisites
- Docker
- ClickHouse instance with an `training_set` table

## Running Locally

1. Add a `.env` file with the following:
```
CLICKHOUSE_HOST=default 
CLICKHOUSE_PORT=8123 
CLICKHOUSE_USER=default 
CLICKHOUSE_PASSWORD=uldanabanana
TRAINING_DATE=2025-05-05
```

2. Build and run:
```bash
docker build -t training-server .
docker run --env-file .env training-server
```

3. To track models, run an MLflow server:
```bash
mlflow ui
```

4. Visit `localhost:5000` to see model logs.