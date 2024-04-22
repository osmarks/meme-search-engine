from aiohttp import web
import aiosqlite
import asyncio
import random
import sys
import json
import os
from pathlib import Path
import shutil

PORT, DATABASE, TARGET_DIR = sys.argv[1:]
with open("rater_mse_config.json", "r") as f:
    mse_config = json.load(f)
    basedir = Path(mse_config["files"])
TARGET_DIR = Path(TARGET_DIR)

app = web.Application(client_max_size=32*1024**2)
routes = web.RouteTableDef()

@routes.get("/")
async def index(request):
    csr = await request.app["db"].execute("SELECT filename FROM library_queue ORDER BY score DESC")
    filename, = await csr.fetchone()
    await csr.close()
    return web.Response(text=f"""
<!DOCTYPE html>
<html>
    <style>
.memes img {{
  width: 100%; 
}}

input {{
    width: 100%;
}}

.memes {{
    margin-top: 2em;
}}
    </style>
    <body>
        <h1>Meme Processing</h1>
        <form action="/" method="POST">
            <input type="text" name="filename" id="filename" autofocus>
            <input type="hidden" name="original_filename" value="{filename}">
            <input type="submit" value="Submit">
            <div class="memes">
                <img src="/memes/{filename}" id="meme1">
            </div>
        </form>
        <script>
            document.addEventListener("keypress", function(event) {{
                if (event.key === "Enter") {{
                    document.querySelector("input[name='rating'][value='1']").checked = true
                    document.querySelector("form").submit()
                }}
            }});
        </script>
    </body>
</html>
    """, content_type="text/html")

def find_new_path(basename, ext):
    ctr = 1
    while True:
        new = TARGET_DIR / (basename + ("" if ctr == 1 else "-" + str(ctr)) + ext)
        if not new.exists(): return new
        ctr += 1

@routes.post("/")
async def rate(request):
    db = request.app["db"]
    post = await request.post()
    filename = post["filename"]
    original_filename = post["original_filename"]
    real_path = basedir / original_filename
    assert real_path.is_file()
    if filename == "": # bad meme, discard
        real_path.unlink()
    else:
        new_path = find_new_path(filename.replace(" ", "-"), real_path.suffix)
        print(real_path, new_path, real_path.suffix)
        shutil.move(real_path, new_path)
    await db.execute("DELETE FROM library_queue WHERE filename = ?", (original_filename,))
    await db.commit()
    return web.HTTPFound("/")

async def main():
    app["db"] = await aiosqlite.connect(DATABASE)
    app.router.add_routes(routes)
    app.router.add_static("/memes/", "./images")
    print("Ready")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "", int(PORT))
    await site.start()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.run_forever()
