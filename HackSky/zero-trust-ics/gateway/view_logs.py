import sqlite3
from tabulate import tabulate

conn = sqlite3.connect("../logs/access_log.db")
cursor = conn.cursor()
cursor.execute("SELECT node_id, ip_address, result, timestamp FROM access_logs ORDER BY id DESC")
rows = cursor.fetchall()
conn.close()

print(tabulate(rows, headers=["Node ID", "IP", "Result", "Time"]))