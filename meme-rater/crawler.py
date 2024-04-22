import aiohttp
import asyncio
import aiofiles
import os.path
import hashlib
import json
import time
import sys

async def fetch_list_seg(sess, list_url, query):
    async with sess.get(list_url + ".json", params=query) as res:
        return await res.json()

async def fetch_past(sess, list_url, n):
    after = None
    count = 0
    while count < n:
        args = { "count": 25 }
        if after is not None: args["after"] = after
        chunk = await fetch_list_seg(sess, list_url, args)
        if "data" not in chunk:
            print("\n", chunk)
            await asyncio.sleep(400)
            continue
        new_items = chunk["data"]["children"]
        yield [ i["data"] for i in new_items ]
        count += len(new_items)
        print("\nup to", count)
        after = new_items[-1]["data"]["name"]

SEEN_ITEMS_SIZE = 200
async def fetch_stream(sess, list_url):
    # dicts are ordered, so this is a very janky ordered set implementation
    seen = {}
    while True:
        list_items = (await fetch_list_seg(sess, list_url, {}))["data"]["children"]
        new = [ i["data"] for i in list_items if i["data"]["name"] not in seen ]
        # yield the new items
        for n in new: yield n
        # add new items to list of seen things
        seen.update(dict.fromkeys(i["name"] for i in new))
        # remove old seen items until it's a reasonable size
        while len(seen) > SEEN_ITEMS_SIZE: seen.pop(next(iter(seen.keys())))
        # compute average time between posts and wait that long for next fetch cycle
        times = [ i["data"]["created"] for i in list_items ]
        timediffs = list(map(lambda x: x[0] - x[1], zip(times, times[1:])))
        average = sum(timediffs) / len(timediffs)
        await asyncio.sleep(average)

def bucket(id): return hashlib.md5(id.encode("utf-8")).hexdigest()[:2]

filetypes = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/avif": "avif"
}

CHUNK_SIZE = 1<<18 # entirely arbitrary
async def download(sess, url, file):
    async with sess.get(url) as res:
        ctype = res.headers.get("content-type")
        if ctype not in filetypes: return
        if int(res.headers.get("content-length", 1e9)) > 8e6: return
        async with aiofiles.open(file + "." + filetypes[ctype], mode="wb") as fh:
            while chunk := await res.content.read(CHUNK_SIZE):
                await fh.write(chunk)
        return dict(res.headers)

async def main(time_threshold):
    sem = asyncio.Semaphore(16)
    
    async with aiohttp.ClientSession() as sess:
        async def download_item(item):
            #print("starting on", item["name"])
            print(".", end="")
            sys.stdout.flush()
            if item["over_18"] or not item["is_robot_indexable"]: return
            id = item["name"]
            bck = bucket(id)
            os.makedirs(os.path.join("images", bck), exist_ok=True)
            os.makedirs(os.path.join("meta", bck), exist_ok=True)
            if not item["url"].startswith("https://"): return
            meta_path = os.path.join("meta", bck, id + ".json")
            if not os.path.exists(meta_path): # sorry
                print("|", end="")
                sys.stdout.flush()
                try:
                    result = await download(sess, item["url"], os.path.join("images", bck, id))
                except Exception as e:
                    print("\nMeme acquisition failure:", e)
                    return
                if result:
                    item["headers"] = result
                    with open(meta_path, "w") as fh:
                        json.dump(item, fh)
                else:
                    print("!", end="")
                    sys.stdout.flush()
            #print("done on", id)

        async def dl_task(item):
            async with sem:
                try:
                    await asyncio.wait_for(download_item(item), timeout=30)
                except asyncio.TimeoutError: pass

        async for items in fetch_past(sess, "https://www.reddit.com/user/osmarks/m/memeharvesting/new", 20000):
            #print("got new chunk")
            await sem.acquire()
            sem.release()
            #print("downloading new set")
            async with asyncio.TaskGroup() as tg:
                for item in items:
                    if time_threshold and time_threshold > item["created"]:
                        return
                    tg.create_task(dl_task(item))

if __name__ == "__main__":
    threshold = None
    if len(sys.argv) > 1:
        print("thresholding at", sys.argv[1])
        threshold = float(sys.argv[1])
    asyncio.run(main(threshold))