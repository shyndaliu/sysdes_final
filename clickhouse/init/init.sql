CREATE TABLE IF NOT EXISTS training_set (
    user_id UInt32,
    song_id UInt32,
    liked UInt8,
    date Date
) ENGINE = MergeTree()
PARTITION BY date
ORDER BY (user_id, song_id);


INSERT INTO training_set
SELECT * FROM file('training_data.csv', 'CSV', 'user_id UInt32, song_id UInt32, liked UInt8, date Date');
