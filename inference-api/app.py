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

@app.route("/predict", methods=["POST"])
def predict():
    """
    Request format:
    {
        "user_id": "user123",
        "candidate_songs": ["song1", "song2", "song3"],
        "k": 10  # optional, number of recommendations to return
    }
    """
    try:
        data = request.get_json()
        user_id = data["user_id"]
        candidate_songs = data["candidate_songs"]
        k = data.get("k", 10)
        
        # Convert to internal indices
        user_idx = model_loader.user_mapping.get(int(user_id))
        if user_idx is None:
            return jsonify({"error": "User not found in model"}), 404
            
        song_indices = []
        valid_songs = []
        for song in candidate_songs:
            song_idx = model_loader.item_mapping.get(int(song))
            if song_idx is not None:
                song_indices.append(song_idx)
                valid_songs.append(song)
        
        if not song_indices:
            return jsonify({"error": "No valid candidate songs found"}), 400
        
        # Create dummy user-item matrix (all zeros)
        user_items = csr_matrix((1, len(model_loader.item_mapping)))
        
        # Get recommendations
        rec_indices, _ = model_loader.model.recommend(
            userid=user_idx,
            user_items=user_items,
            N=k,
            items=song_indices,
            filter_already_liked_items=False
        )
        
        # Convert back to song IDs
        recommendations = []
        for idx in rec_indices:
            song_id = list(model_loader.item_mapping.keys())[list(model_loader.item_mapping.values()).index(idx)]
            recommendations.append(str(song_id))
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations,
            "valid_candidates": valid_songs
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)