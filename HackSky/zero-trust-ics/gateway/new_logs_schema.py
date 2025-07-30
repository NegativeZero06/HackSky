import sqlite3

DB_PATH = "./logs/access_log.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Backup data (optional)
cursor.execute("SELECT node_id, ip_address, timestamp, result FROM access_logs")
old_data = cursor.fetchall()

# Drop the old table
cursor.execute("DROP TABLE IF EXISTS access_logs")

# Create new schema
cursor.execute("""
    CREATE TABLE access_logs (
        node_id TEXT,
        resource TEXT,
        timestamp TEXT,
        status TEXT
    )
""")

# Migrate old data to new format
for row in old_data:
    node_id, ip_address, timestamp, result = row
    cursor.execute("INSERT INTO access_logs (node_id, resource, timestamp, status) VALUES (?, ?, ?, ?)",
                   (node_id, ip_address, timestamp, result))

conn.commit()
conn.close()

print("✅ access_logs table schema fixed and data migrated.")
