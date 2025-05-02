import mlflow
mlflow.set_tracking_uri("http://localhost:5001")
def load_latest_model(name="collab_model"):
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(name, stages=["None", "Staging"])
    if not versions:
        raise ValueError("No model found in MLflow.")
    model_uri = f"runs:/{versions[0].run_id}/model"
    return mlflow.artifacts.download_artifacts(model_uri)


def promote_model_to_production(name, version):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
