import sqlite3
import hashlib
from collections import defaultdict
import numpy
import random
import numpy as np

db = sqlite3.connect("data.sqlite3")
db.row_factory = sqlite3.Row

val_fraction = 0.2
def is_val_set(meme1, meme2):
    def is_one_val(meme):
        return hashlib.sha256(meme.encode("utf-8")).digest()[0] / 255 < (val_fraction / 2) # not strictly correct but good enough
    return is_one_val(meme1) or is_one_val(meme2)

def fetch_embedding(filename):
    csr = db.execute("SELECT embedding FROM files WHERE filename = ?", (filename,))
    x = numpy.frombuffer(csr.fetchone()[0], dtype="float16")
    csr.close()
    return x.copy() # PyTorch complains otherwise due to bad

def map_rating(rating):
    def map_one(rating):
        match rating:
            case "1": # meme 1 is better
                return 0.9
            case "2":
                return 0.1
            case "2+" | "2p":
                return 0.3
            case "1+" | "1p":
                return 0.7
            case "eq":
                return 0.5
            case _: raise ValueError("invalid rating, please fix")

    return np.array([map_one(r) for r in rating.split(",")])

def fetch_ratings(sets):
    trains = defaultdict(list)
    validations = defaultdict(list)
    csr = db.execute("SELECT meme1, meme2, rating, iteration FROM ratings")
    its = set()
    for meme1, meme2, rating, iteration in csr.fetchall():
        if iteration not in its:
            print(iteration)
        its.add(iteration)
        (validations if is_val_set(meme1, meme2) else trains)[int((iteration and iteration.split("-")[0]) or "0")].append((fetch_embedding(meme1), fetch_embedding(meme2), map_rating(rating)))
    csr.close()
    return list(x[1] for x in sorted(trains.items()) if str(x[0]) in sets), list(x[1] for x in sorted(validations.items()) if str(x[0]) in sets)

def generate_random_permutations(x, n):
    out = []
    for _ in range(n):
        random.shuffle(x)
        out.append(x.copy())
    return out

def fetch_all_files():
    csr = db.execute("SELECT filename, embedding FROM files WHERE embedding IS NOT NULL")
    x = [ (row[0], numpy.frombuffer(row[1], dtype="float16").copy()) for row in csr.fetchall() ]
    csr.close()
    return x

def checkpoint_for(steps):
    return f"./ckpt/model-{steps}.pt", f"./ckpt/optim-{steps}.pt"
