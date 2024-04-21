import sqlite3
import json
import sys

iteration = sys.argv[1]

db = sqlite3.connect("data.sqlite3")
db.row_factory = sqlite3.Row

with open("top.json", "r") as f:
    listing = json.load(f)

db.executemany("INSERT INTO queue VALUES (?, ?, ?)", [ (x[0], x[1], iteration) for x, v in listing ])
db.commit()