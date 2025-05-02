import numpy as np
import json
import os

from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from clickhouse_conn import get_clickhouse_data, get_clickhouse_train_data
from mlflow_utils import load_latest_model, promote_model_to_production
import mlflow


def recall_at_k(preds, truth, k=100):
    correct = 0
    total   = len(truth)
    for u, actual_items in truth.items():
        recs = preds.get(u, [])[:k]
        correct += len(set(recs) & set(actual_items))
    return correct / total if total else 0


def evaluate_model(date_partition="2023-07-06", threshold=0.7):
    print(f"Evaluating on data *at* {date_partition}\n")

    # —– Load your registered model artifacts —–
    model_dir = load_latest_model("collab_model")
    # saved .npz has the exact user_factors & item_factors you trained with
    factors      = np.load(os.path.join(model_dir, "model.npz"))
    user_factors = factors["user_factors"]
    item_factors = factors["item_factors"]

    with open(os.path.join(model_dir, "user_mapping.json")) as f:
        raw = json.load(f)
        user_mapping = {int(k): v for k, v in raw.items()}
    with open(os.path.join(model_dir, "item_mapping.json")) as f:
        raw = json.load(f)
        item_mapping = {int(k): v for k, v in raw.items()}

    # reconstruct the ALS model “in memory”
    model = AlternatingLeastSquares()
    model.user_factors = user_factors
    model.item_factors = item_factors

    # —– Build the *training* user×item matrix for filtering —–
    # (so we only block out items they already saw *in train*, not in test)
    df_train = get_clickhouse_train_data(date_partition=date_partition)  
    print("df_train rows:", len(df_train))
    df_train = df_train[df_train["liked"] > 0]  
    print("df_train rows:", len(df_train))                 
    df_train = df_train[
        df_train["user_id"].isin(user_mapping) &
        df_train["song_id"].isin(item_mapping)
    ]
    print("df_train rows:", len(df_train))
    df_train["uidx"] = df_train["user_id"].map(user_mapping).astype(int)
    df_train["iidx"] = df_train["song_id"].map(item_mapping).astype(int)


    n_users = len(user_mapping)
    n_items = len(item_mapping)
    train_mat = coo_matrix(
        (
            df_train["liked"].astype(float).values,
            (df_train["uidx"].values, df_train["iidx"].values)
        ),
        shape=(n_users, n_items)
    ).tocsr()

    print("train_mat nnz:", train_mat.nnz)
    print("num users w/ interactions:", (train_mat.sum(axis=1) > 0).sum())



    # —– Build the *test* truth set —–
    df_test = get_clickhouse_data(date_partition=date_partition) 
    df_test = df_test[df_test["liked"] > 0]
    # filter to only users/items seen in train
    df_test = df_test[
        df_test["user_id"].isin(user_mapping) &
        df_test["song_id"].isin(item_mapping)
    ]
    df_test["uidx"] = df_test["user_id"].map(user_mapping).astype(int)
    df_test["iidx"] = df_test["song_id"].map(item_mapping).astype(int)

    truths = {}
    for uidx, group in df_test.groupby("uidx"):
        truths[uidx] = group["iidx"].tolist()
    print("Total truth interactions:", sum(len(v) for v in truths.values()))



    # —– Do recommendations —–
    preds = {}
    for uidx in truths:
        rec_iidxs, _ = model.recommend(
            uidx,
            train_mat[uidx],
            N=100,
            filter_already_liked_items=False
        )
        preds[uidx] = rec_iidxs.tolist()


    # —– Score recall@10 on *index* space —–
    recall = recall_at_k(preds, truths, k=100)
    print(f"Recall@10 = {recall:.4f}")

    # —– Promotion logic —–
    if recall >= threshold:
        print("✅ Met threshold — promoting to PROD")
        client   = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("collab_model", stages=["None","Staging"])
        if versions:
            promote_model_to_production("collab_model", versions[0].version)
    else:
        print("❌ Did not meet threshold")


if __name__ == "__main__":
    evaluate_model()
