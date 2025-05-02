ATTACH TABLE _ UUID 'e93589d2-e17a-4df5-aba6-25820848f9f1'
(
    `user_id` UInt32,
    `song_id` UInt32,
    `liked` UInt8,
    `date` Date
)
ENGINE = MergeTree
PARTITION BY date
ORDER BY (user_id, song_id)
SETTINGS index_granularity = 8192
