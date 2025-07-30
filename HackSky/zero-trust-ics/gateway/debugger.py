import sqlite3

conn = sqlite3.connect("./logs/access_log.db")
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(access_logs)")
print("access_logs schema:")
print(cursor.fetchall())

cursor.execute("SELECT * FROM access_logs LIMIT 5")
print("Sample logs:")
print(cursor.fetchall())
conn.close()
