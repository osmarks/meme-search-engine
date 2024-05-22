import numpy as np
import sqlite3
import base64

#db = sqlite3.connect("/srv/mse/data.sqlite3")
db = sqlite3.connect("data.sqlite3")
db.row_factory = sqlite3.Row

name = input("Name: ")
url = input("Embedding search URL: ")
data = base64.urlsafe_b64decode(url.removeprefix("https://mse.osmarks.net/?e="))
arr = np.frombuffer(data, dtype=np.float16).copy()
db.execute("INSERT OR REPLACE INTO predefined_embeddings VALUES (?, ?)", (name, arr.tobytes()))
db.commit()