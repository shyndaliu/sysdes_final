from clickhouse_connect import get_client
import pandas as pd

def get_clickhouse_data(table="training_set", date_partition="2023-07-06"):
    client = get_client(host="localhost", port=8123, username="default", password="uldanabanana")
    query = f"""
        SELECT user_id, song_id, liked
        FROM {table}
        WHERE date = '{date_partition}'
    """
    result = client.query(query)
    df = pd.DataFrame(result.result_set, columns=["user_id", "song_id", "liked"])
    return df

def get_clickhouse_train_data(table="training_set", date_partition="2023-07-06"):
    client = get_client(host="localhost", port=8123, username="default", password="uldanabanana")
    query = f"""
        SELECT user_id, song_id, liked
        FROM {table}
        WHERE date < '{date_partition}'
    """
    result = client.query(query)
    df = pd.DataFrame(result.result_set, columns=["user_id", "song_id", "liked"])
    return df
