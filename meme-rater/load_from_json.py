import jsonlines
import sqlite3
import numpy as np

import shared

shared.db.executescript("""
CREATE TABLE IF NOT EXISTS files (
    filename TEXT NOT NULL,
    title TEXT NOT NULL,
    link TEXT NOT NULL,
    embedding BLOB NOT NULL,
    timestamp INTEGER NOT NULL,
    UNIQUE (filename)
);
""")

with jsonlines.open("sample.jsonl") as reader:
    for obj in reader:
        shared.db.execute("INSERT OR REPLACE INTO files (filename, title, link, embedding, timestamp) VALUES (?, ?, ?, ?, ?)", (obj["metadata"]["final_url"], obj["title"], f"https://reddit.com/r/{obj['subreddit']}/comments/{obj['id']}", sqlite3.Binary(np.array(obj["embedding"], dtype=np.float16).tobytes()), obj["timestamp"]))
shared.db.commit()
