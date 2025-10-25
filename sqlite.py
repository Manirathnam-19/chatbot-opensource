import sqlite3
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())
cursor.execute("SELECT * FROM message_store")
rows = cursor.fetchall()
for row in rows:
    print(row)
conn.close()
