import pandas as pd
import mlflow
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from clickhouse_conn import get_clickhouse_data
from mlflow_utils import init_mlflow, log_model_metrics
import shutil
import os
import numpy as np
import json


def train_model(date_partition="2023-07-06"):
    print(f"Training using data before {date_partition}")
    df = get_clickhouse_data(date_partition=date_partition)


    if df.empty:
        print("No training data for this partition.")
        return
    
    df = df[df["liked"] > 0]

    # Create mappings
    user_ids = df["user_id"].unique()
    item_ids = df["song_id"].unique()
    user_mapping = {int(u): i for i, u in enumerate(user_ids)}
    item_mapping = {int(s): i for i, s in enumerate(item_ids)}


    # Encode matrix
    cols = df["user_id"].map(user_mapping)
    rows = df["song_id"].map(item_mapping)
    data = df["liked"].astype(float)
    matrix = coo_matrix((data, (rows, cols))).tocsr()

    # Train model
    model = AlternatingLeastSquares(
        factors=128,
        regularization=0.1,
        iterations=50,
        use_gpu=False
    )
    model.fit(matrix)

    # Prepare output dirs
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    # Save ALS model weights
    np.savez(f"{model_dir}/model.npz",
             user_factors=model.user_factors,
             item_factors=model.item_factors)

    # Save mappings
    with open(f"{model_dir}/user_mapping.json", "w") as f:
        json.dump(user_mapping, f)
    with open(f"{model_dir}/item_mapping.json", "w") as f:
        json.dump(item_mapping, f)

    # MLflow logging
    init_mlflow()
    with mlflow.start_run(run_name=f"training-{date_partition}") as run:
        mlflow.log_artifacts(model_dir, artifact_path="model")

        artifact_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(
            model_uri=artifact_uri,
            name="collab_model"
        )

        print(f"Model registered: {result.name} v{result.version}")

        # Placeholder recall metric for now
        log_model_metrics(model, run_name=f"training-{date_partition}", recall=0.85)

    # Clean up
    shutil.rmtree(model_dir)


if __name__ == "__main__":
    train_model()
