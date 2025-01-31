import numpy as np
import aiohttp
import asyncio
import sys

queries = np.random.randn(1000, 1152)

async def main():
    async with aiohttp.ClientSession() as sess:
        async with asyncio.TaskGroup() as tg:
            sem = asyncio.Semaphore(100)
            async def lookup(embedding):
                async with sess.post("http://localhost:5601", json={
                    "terms": [{ "embedding": list(float(x) for x in embedding) }], # sorry
                    "k": 10
                }) as res:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    return (await res.json())["matches"]

            async def dispatch(i):
                await lookup(queries[i])
                sem.release()

            for i in range(1000):
                await sem.acquire()
                tg.create_task(dispatch(i))

asyncio.run(main())
