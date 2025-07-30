import sqlite3

DB_PATH = "./logs/access_log.db"

sample_data = [
    ("node_001", "/config", "2025-07-30T12:00:00", "ALLOWED"),
    ("node_002", "/firmware", "2025-07-30T12:01:00", "DENIED"),
    ("node_003", "/data", "2025-07-30T12:02:00", "ALLOWED"),
]

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

for entry in sample_data:
    cursor.execute("INSERT INTO access_logs (node_id, resource, timestamp, status) VALUES (?, ?, ?, ?)", entry)

conn.commit()
conn.close()

print("Sample logs inserted.")
