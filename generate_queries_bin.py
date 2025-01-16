import sys, aiohttp, msgpack, numpy, pgvector.asyncpg, asyncio

async def use_emb_server(sess, query):
    async with sess.post("http://100.64.0.10:1708/", data=msgpack.dumps(query), timeout=aiohttp.ClientTimeout(connect=5, sock_connect=5, sock_read=None)) as res:
        response = msgpack.loads(await res.read())
        if res.status == 200:
            return response
        else:
            raise Exception(response if res.headers.get("content-type") == "application/msgpack" else (await res.text()))

BATCH_SIZE = 32

async def main():
    with open("query_data.bin", "wb") as f:
        with open("queries.txt", "r") as g:
            write_lock = asyncio.Lock()
            async with aiohttp.ClientSession() as sess:
                async with asyncio.TaskGroup() as tg:
                    sem = asyncio.Semaphore(3)

                    async def process_batch(batch):
                        while True:
                            try:
                                embs = await use_emb_server(sess, { "text": batch })
                                async with write_lock:
                                    f.write(b"".join(embs))
                                sys.stdout.write(".")
                                sys.stdout.flush()
                                break
                            except Exception as e:
                                print(e)
                                await asyncio.sleep(5)

                        sem.release()

                    async def dispatch(batch):
                        await sem.acquire()
                        tg.create_task(process_batch(batch))

                    batch = []
                    while line := g.readline():
                        if line.strip(): batch.append(line.strip())
                        if len(batch) == BATCH_SIZE:
                            await dispatch(batch)
                            batch = []
                    if len(batch) > 0:
                        await dispatch(batch)

if __name__ == "__main__":
    asyncio.run(main())
