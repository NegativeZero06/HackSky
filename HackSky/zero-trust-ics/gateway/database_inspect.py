import sqlite3

conn = sqlite3.connect("./logs/access_log.db")
cursor = conn.cursor()

# Print all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# If access_logs exists, print its structure
if ('access_logs',) in tables:
    cursor.execute("PRAGMA table_info(access_logs);")
    print("access_logs schema:")
    for row in cursor.fetchall():
        print(row)
else:
    print("⚠️ Table 'access_logs' not found.")

conn.close()
