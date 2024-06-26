from aiohttp import web
import aiohttp
import asyncio
import traceback
import umsgpack
from PIL import Image
import base64
import aiosqlite
import faiss
import numpy
import os
import aiohttp_cors
import json
import io
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

with open(sys.argv[1], "r") as config_file:
    CONFIG = json.load(config_file)

app = web.Application(client_max_size=32*1024**2)
routes = web.RouteTableDef()

async def clip_server(sess: aiohttp.ClientSession, query, unpack_buffer=True):
    async with sess.post(CONFIG["clip_server"], data=umsgpack.dumps(query)) as res:
        response = umsgpack.loads(await res.read())
        if res.status == 200:
            if unpack_buffer:
                response = [ numpy.frombuffer(x, dtype="float16") for x in response ]
            return response
        else:
            raise Exception(response if res.headers.get("content-type") == "application/msgpack" else (await res.text()))

@routes.post("/")
async def run_query(request):
    sess = app["session"]
    data = await request.json()
    embeddings = []
    if images := data.get("images", []):
        target_image_size = app["index"].inference_server_config["image_size"]
        embeddings.extend(await clip_server(sess, { "images": [ load_image(io.BytesIO(base64.b64decode(x)), target_image_size)[0] for x, w in images ] }))
    if text := data.get("text", []):
        embeddings.extend(await clip_server(sess, { "text": [ x for x, w in text ] }))
    weights = [ w for x, w in images ] + [ w for x, w in text ]
    weighted_embeddings = [ e * w for e, w in zip(embeddings, weights) ]
    weighted_embeddings.extend([ numpy.array(x) for x in data.get("embeddings", []) ])
    if not weighted_embeddings:
        return web.json_response([])
    return web.json_response(app["index"].search(sum(weighted_embeddings), top_k=data.get("top_k", 4000)))

@routes.get("/")
async def health_check(request):
    return web.Response(text="OK")

@routes.post("/reload_index")
async def reload_index_route(request):
    await request.app["index"].reload()
    return web.json_response(True)

def load_image(path, image_size):
    im = Image.open(path)
    im.draft("RGB", image_size)
    buf = io.BytesIO()
    im.resize(image_size).convert("RGB").save(buf, format="BMP")
    return buf.getvalue(), path

class Index:
    def __init__(self, inference_server_config, http_session):
        self.faiss_index = faiss.IndexFlatIP(inference_server_config["embedding_size"])
        self.associated_filenames = []
        self.inference_server_config = inference_server_config
        self.lock = asyncio.Lock()
        self.session = http_session

    def search(self, query, top_k):
        distances, indices = self.faiss_index.search(numpy.array([query]), top_k)
        distances = distances[0]
        indices = indices[0]
        try:
            indices = indices[:numpy.where(indices==-1)[0][0]]
        except IndexError: pass
        return [ { "score": float(distance), "file": self.associated_filenames[index] } for index, distance in zip(indices, distances) ]

    async def run_ocr(self):
        if not CONFIG.get("enable_ocr"): return

        import ocr

        print("Running OCR")

        conn = await aiosqlite.connect(CONFIG["db_path"])
        unocred = await conn.execute_fetchall("SELECT files.filename FROM files LEFT JOIN ocr ON files.filename = ocr.filename WHERE ocr.scan_time IS NULL OR ocr.scan_time < files.modtime")

        ocr_sem = asyncio.Semaphore(20) # Google has more concurrency than our internal CLIP backend. I am sure they will be fine.
        load_sem = threading.Semaphore(100) # provide backpressure in loading to avoid using 50GB of RAM (this happened)

        async def run_image(filename, chunks):
            try:
                text, regions = await ocr.scan_chunks(self.session, chunks)
                await conn.execute("INSERT OR REPLACE INTO ocr VALUES (?, ?, ?, ?)", (filename, time.time(), text, json.dumps(regions)))
                await conn.commit()
                sys.stdout.write(".")
                sys.stdout.flush()
            except:
                print("OCR failed on", filename)
            finally:
                ocr_sem.release()

        def load_and_chunk_image(filename):
            load_sem.acquire()
            im = Image.open(Path(CONFIG["files"]) / filename)
            return filename, ocr.chunk_image(im)

        async with asyncio.TaskGroup() as tg:
            with ThreadPoolExecutor(max_workers=CONFIG.get("n_workers", 1)) as executor:
                for task in asyncio.as_completed([ asyncio.get_running_loop().run_in_executor(executor, load_and_chunk_image, file[0]) for file in unocred ]):
                    filename, chunks = await task
                    await ocr_sem.acquire()
                    tg.create_task(run_image(filename, chunks))
                    load_sem.release()

    async def reload(self):
        async with self.lock:
            with ThreadPoolExecutor(max_workers=CONFIG.get("n_workers", 1)) as executor:
                print("Indexing")
                conn = await aiosqlite.connect(CONFIG["db_path"])
                conn.row_factory = aiosqlite.Row
                await conn.executescript("""
CREATE TABLE IF NOT EXISTS files (
    filename TEXT PRIMARY KEY,
    modtime REAL NOT NULL,
    embedding_vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS ocr (
    filename TEXT PRIMARY KEY REFERENCES files(filename),
    scan_time INTEGER NOT NULL,
    text TEXT NOT NULL,
    raw_segments TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5 (
    filename,
    text,
    tokenize='unicode61 remove_diacritics 2',
    content='ocr'
);

CREATE TRIGGER IF NOT EXISTS ocr_fts_ins AFTER INSERT ON ocr BEGIN
    INSERT INTO ocr_fts (rowid, filename, text) VALUES (new.rowid, new.filename, new.text);
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_del AFTER DELETE ON ocr BEGIN
    INSERT INTO ocr_fts (ocr_fts, rowid, filename, text) VALUES ('delete', old.rowid, old.filename, old.text);
END;
                """)
                try:
                    async with asyncio.TaskGroup() as tg:
                        batch_sem = asyncio.Semaphore(3)

                        modified = set()

                        async def do_batch(batch):
                            try:
                                query = { "images": [ arg[2] for arg in batch ] }
                                embeddings = await clip_server(self.session, query, False)
                                await conn.executemany("INSERT OR REPLACE INTO files VALUES (?, ?, ?)", [
                                    (filename, modtime, embedding) for (filename, modtime, _), embedding in zip(batch, embeddings)
                                ])
                                await conn.commit()
                                for filename, _, _ in batch:
                                    modified.add(filename)
                                sys.stdout.write(".")
                                sys.stdout.flush()
                            finally:
                                batch_sem.release()

                        async def dispatch_batch(batch):
                            await batch_sem.acquire()
                            tg.create_task(do_batch(batch))
                        
                        files = {}
                        for filename, modtime in await conn.execute_fetchall("SELECT filename, modtime FROM files"):
                            files[filename] = modtime
                        await conn.commit()
                        batch = []
                        seen_files = set()

                        failed = set()
                        for dirpath, _, filenames in os.walk(CONFIG["files"]):
                            paths = set()
                            done = set()
                            for file in filenames:
                                path = os.path.join(dirpath, file)
                                file = os.path.relpath(path, CONFIG["files"])
                                st = os.stat(path)
                                seen_files.add(file)
                                if st.st_mtime != files.get(file):
                                    paths.add(path)
                            for task in asyncio.as_completed([ asyncio.get_running_loop().run_in_executor(executor, load_image, path, self.inference_server_config["image_size"]) for path in paths ]):
                                try:
                                    b, path = await task
                                    st = os.stat(path)
                                    file = os.path.relpath(path, CONFIG["files"])
                                    done.add(path)
                                except Exception as e:
                                    # print(file, "failed", e) we can't have access to file when we need it, oops
                                    continue
                                batch.append((file, st.st_mtime, b))
                                if len(batch) == self.inference_server_config["batch"]:
                                    await dispatch_batch(batch)
                                    batch = []
                            failed |= paths - done
                        if batch:
                            await dispatch_batch(batch)

                        print()
                        for failed_ in failed:
                            print("Failed to load", failed_)
                
                    filenames_set = set(self.associated_filenames)
                    new_data = []
                    new_filenames = []
                    async with conn.execute("SELECT * FROM files") as csr:
                        while row := await csr.fetchone():
                            filename, modtime, embedding_vector = row
                            if filename not in filenames_set:
                                new_data.append(numpy.frombuffer(embedding_vector, dtype="float16"))
                                new_filenames.append(filename)
                    if not new_data: return
                    new_data = numpy.array(new_data)
                    self.associated_filenames.extend(new_filenames)
                    self.faiss_index.add(new_data)

                    remove_indices = []
                    for index, filename in enumerate(self.associated_filenames):
                        if filename not in seen_files or filename in modified:
                            remove_indices.append(index)
                            self.associated_filenames[index] = None
                        if filename not in seen_files:
                            await conn.execute("DELETE FROM files WHERE filename = ?", (filename,))
                            await conn.commit()
                    print("Deleting", len(remove_indices), "old entries")
                    # TODO concurrency
                    # TODO understand what that comment meant
                    if remove_indices:
                        self.faiss_index.remove_ids(numpy.array(remove_indices))
                        self.associated_filenames = [ x for x in self.associated_filenames if x is not None ]
                finally:
                    await conn.close()

            await self.run_ocr()

app.router.add_routes(routes)

cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=False,
        expose_headers="*",
        allow_headers="*",
    )
})
for route in list(app.router.routes()):
    cors.add(route)

async def main():
    sess = aiohttp.ClientSession()
    while True:
            try:
                async with await sess.get(CONFIG["clip_server"] + "config") as res:
                    inference_server_config = umsgpack.unpackb(await res.read())
                    print("Backend config:", inference_server_config)
                    break
            except:
                traceback.print_exc()
                await asyncio.sleep(1)
    index = Index(inference_server_config, sess)
    app["index"] = index
    app["session"] = sess
    await index.reload()
    print("Ready")
    if CONFIG.get("no_run_server", False): return
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "", CONFIG["port"])
    await site.start()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    if CONFIG.get("no_run_server", False) == False: loop.run_forever()
