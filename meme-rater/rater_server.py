from aiohttp import web
import aiosqlite
import asyncio
import random
import sys

PORT, DATABASE = sys.argv[1:]

app = web.Application(client_max_size=32*1024**2)
routes = web.RouteTableDef()

async def get_pair(db):
    while True:
        filenames = [ x[0] for x in await db.execute_fetchall("SELECT filename FROM files", ()) ]
        m1, m2 = tuple(sorted(random.sample(filenames, 2)))
        csr = await db.execute("SELECT 1 FROM ratings WHERE meme1 = ? AND meme2 = ?", (m1, m2))
        if not await csr.fetchone():
            return m1, m2

@routes.get("/")
async def index(request):
    meme1, meme2 = await get_pair(request.app["db"])
    return web.Response(text=f"""
<!DOCTYPE html>
<html>
    <style>
.memes img {{
  width: 45%; 
}}

@media (max-width: 768px) {{
 .memes img {{
    width: 100%;
  }}
}}

.memes {{
    margin-top: 2em;
}}
    </style>
    <body>
        <h1>Meme Rating</h1>
        <form action="/rate" method="POST">
            <input type="radio" name="rating" value="1" id="rating1"> <label for="rating1">Meme 1 is better</label>
            <input type="radio" name="rating" value="2" id="rating2"> <label for="rating2">Meme 2 is better</label>

            <input type="hidden" name="meme1" value="{meme1}">
            <input type="hidden" name="meme2" value="{meme2}">
            <input type="submit" value="Submit">
            <div class="memes">
                <img src="/memes/{meme1}" id="meme1">
                <img src="/memes/{meme2}" id="meme2">
            </div>
        </form>
        <script>
            document.addEventListener("keypress", function(event) {{
                if (event.key === "1") {{
                    document.querySelector("input[name='rating'][value='1']").checked = true
                    document.querySelector("form").submit()
                }} else if (event.key === "2") {{
                    document.querySelector("input[name='rating'][value='2']").checked = true
                    document.querySelector("form").submit()
                }}
            }});
            document.querySelector("#meme1").addEventListener("click", function(event) {{
                document.querySelector("input[name='rating'][value='1']").checked = true
                document.querySelector("form").submit()
            }})
            document.querySelector("#meme2").addEventListener("click", function(event) {{
                document.querySelector("input[name='rating'][value='2']").checked = true
                document.querySelector("form").submit()
            }})
        </script>
    </body>
</html>
    """, content_type="text/html")

@routes.post("/rate")
async def rate(request):
    db = request.app["db"]
    post = await request.post()
    meme1 = post["meme1"]
    meme2 = post["meme2"]
    rating = post["rating"]
    await db.execute("INSERT INTO ratings (meme1, meme2, rating) VALUES (?, ?, ?)", (meme1, meme2, rating))
    await db.commit()
    return web.HTTPFound("/")

async def main():
    app["db"] = await aiosqlite.connect(DATABASE)
    await app["db"].executescript("""
CREATE TABLE IF NOT EXISTS ratings (
    meme1 TEXT NOT NULL,
    meme2 TEXT NOT NULL,
    rating TEXT NOT NULL,
    UNIQUE (meme1, meme2)
);
""")
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
