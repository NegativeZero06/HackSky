import sqlite3

DB_PATH = "./logs/access_log.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS access_policies (
        node_id TEXT,
        resource TEXT,
        allowed INTEGER
    )
''')

conn.commit()
conn.close()

print("✅ access_policies table created.")
