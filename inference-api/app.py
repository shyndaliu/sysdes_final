import os
import json
import numpy as np
from flask import Flask, request, jsonify
import mlflow
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://host.docker.internal:5001")

MODEL_NAME = "collab_model"
mlflow.set_tracking_uri(MLFLOW_URI)

app = Flask(__name__)

class ModelLoader:
    def __init__(self):
        self.model = None
        self.user_mapping = None
        self.item_mapping = None
        self.load_model()

    def load_model(self):
        """Load production model using your proven approach"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get latest production version
            prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if not prod_versions:
                raise ValueError(f"No production model found for {MODEL_NAME}")
                
            version = prod_versions[0]
            print(f"Loading production model: {MODEL_NAME} version {version.version}")
            
            # Download model artifacts
            model_uri = f"runs:/{version.run_id}/model"
            model_dir = mlflow.artifacts.download_artifacts(model_uri)
            
            # Load model components
            factors = np.load(os.path.join(model_dir, "model.npz"))
            
            with open(os.path.join(model_dir, "user_mapping.json")) as f:
                raw = json.load(f)
                self.user_mapping = {int(k): v for k, v in raw.items()}
                
            with open(os.path.join(model_dir, "item_mapping.json")) as f:
                raw = json.load(f)
                self.item_mapping = {int(k): v for k, v in raw.items()}
            
            # Reconstruct ALS model
            self.model = AlternatingLeastSquares()
            self.model.user_factors = factors["user_factors"]
            self.model.item_factors = factors["item_factors"]
            
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise

# Initialize model loader
model_loader = ModelLoader()

@app.route("/health", methods=["GET"])
def health():
    status = {
        "status": "ok",
        "model_loaded": model_loader.model is not None,
        "user_mapping_size": len(model_loader.user_mapping) if model_loader.user_mapping else 0,
        "item_mapping_size": len(model_loader.item_mapping) if model_loader.item_mapping else 0
    }
    return jsonify(status), 200

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_id = data.get("user_id")
    top_k = data.get("top_k", 10)
    
    if user_id not in model_loader.user_mapping:
        return jsonify({"error": "User not found in training data"}), 404

    uidx = model_loader.user_mapping[user_id]
    user_interactions = csr_matrix((1, len(model_loader.item_mapping)))
    
    recs, scores = model_loader.model.recommend(
        uidx,
        user_interactions,
        N=top_k,
        filter_already_liked_items=False
    )

    # Invert item mapping for readability
    item_mapping_inv = {v: k for k, v in model_loader.item_mapping.items()}
    recommended_items = [item_mapping_inv[i] for i in recs]

    return jsonify({
        "user_id": user_id,
        "recommendations": recommended_items
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)