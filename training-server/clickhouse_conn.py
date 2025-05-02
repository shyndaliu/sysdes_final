from clickhouse_connect import get_client
import pandas as pd

def get_clickhouse_data(host="localhost", port=8123, user="default", password="uldanabanana", date_partition="2023-07-06"):
    client = get_client(host=host, port=port, username=user, password=password)
    query = f"""
            SELECT user_id, song_id, liked
            FROM training_set
            WHERE date < '{date_partition}'
        """
    result = client.query(query)
    df = pd.DataFrame(result.result_set, columns=["user_id", "song_id", "liked"])
    print(f"Returned rows: {len(df)}")
    print(df.head())

    return df
