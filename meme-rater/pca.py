import sklearn.decomposition
import numpy as np
import sqlite3
import asyncio
import aiohttp
import base64

meme_search_backend = "http://localhost:1707/"
memes_url = "https://i.osmarks.net/memes-or-something/"
meme_search_url = "https://mse.osmarks.net/?e="
db = sqlite3.connect("/srv/mse/data.sqlite3")
db.row_factory = sqlite3.Row

def fetch_all_files():
    csr = db.execute("SELECT embedding FROM files WHERE embedding IS NOT NULL")
    x = [ np.frombuffer(row[0], dtype="float16").copy() for row in csr.fetchall() ]
    csr.close()
    return np.array(x)

embeddings = fetch_all_files()

print("loaded")
pca = sklearn.decomposition.PCA()
pca.fit(embeddings)
print(pca.explained_variance_ratio_)
print(pca.components_)

def emb_url(embedding):
    return meme_search_url + base64.urlsafe_b64encode(embedding.astype(np.float16).tobytes()).decode("utf-8")

async def get_exemplars():
    with open("components.html", "w") as f:
        f.write("""<!DOCTYPE html>
<title>Embeddings PCA</title>
<style>
div img {
    width: 20%
}
</style>
<body><h1>Embeddings PCA</h1>""")
        async with aiohttp.ClientSession():
            async def lookup(embedding):
                async with aiohttp.request("POST", meme_search_backend, json={
                    "terms": [{ "embedding": list(float(x) for x in embedding) }], # sorry
                    "k": 10
                }) as res:
                    return (await res.json())["matches"]

            for i, (component, explained_variance_ratio) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
                f.write(f"""
<h2>Component {i}</h2>
<h3>Explained variance {explained_variance_ratio*100:0.2}%</h3>
<div>
<h4><a href="{emb_url(component)}">Max</a></h4>
""")
                for match in await lookup(component):
                    f.write(f'<img loading="lazy" src="{memes_url+match[1]}">')
                f.write(f'<h4><a href="{emb_url(-component)}">Min</a></h4>')
                for match in await lookup(-component):
                    f.write(f'<img loading="lazy" src="{memes_url+match[1]}">')
                f.write("</div>")

asyncio.run(get_exemplars())