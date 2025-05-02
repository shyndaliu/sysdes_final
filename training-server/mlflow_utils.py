import mlflow

def init_mlflow():
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("collab_filtering")

def log_model_metrics(model, run_name, recall):
    mlflow.log_metric("recall", recall)
    mlflow.log_artifact("model/model.npz")
