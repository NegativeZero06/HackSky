import sqlite3

DB_PATH = "./logs/access_log.db"

sample_policies = [
    ("node_001", "/config", 1),     # Allowed
    ("node_002", "/firmware", 0),   # Denied
    ("node_003", "/data", 1),       # Allowed
]

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

for policy in sample_policies:
    cursor.execute("INSERT INTO access_policies (node_id, resource, allowed) VALUES (?, ?, ?)", policy)

conn.commit()
conn.close()

print("Sample access policies inserted.")
