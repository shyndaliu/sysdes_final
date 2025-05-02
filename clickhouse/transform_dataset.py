import pandas as pd

columns = [
    "id",
    "timestamp",
    "user_id",
    "song_id",
    "liked",
    "meta1",
    "meta2",
    "meta3"
]

df = pd.read_csv("piki_dataset.csv", names=columns, parse_dates=["timestamp"], header=None)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

df = df[["user_id", "song_id", "liked", "timestamp"]]

user_counts = df["user_id"].value_counts()
df = df[df["user_id"].isin(user_counts[user_counts >= 5].index)]

df = df.sort_values("timestamp").reset_index(drop=True)

# Calculate 80th percentile timestamp
split_index = int(len(df) * 0.8)
split_time = df.loc[split_index, "timestamp"]

# Assign fixed dates based on the 80/20 timestamp split
df["date"] = df["timestamp"].apply(lambda x: pd.to_datetime("2023-07-01") if x <= split_time else pd.to_datetime("2023-07-06"))

# Final format
df = df[["user_id", "song_id", "liked", "date"]]
df.to_csv("training_data.csv", index=False)
