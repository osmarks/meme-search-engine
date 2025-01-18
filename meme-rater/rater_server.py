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
        csr = await db.execute("SELECT * FROM queue")
        row = await csr.fetchone()
        await csr.close()
        iteration = None
        if row:
            m1, m2, iteration = row
        else:
            filenames = [ x[0] for x in await db.execute_fetchall("SELECT filename FROM files", ()) ]
            m1, m2 = tuple(sorted(random.sample(filenames, 2)))
        csr = await db.execute("SELECT 1 FROM ratings WHERE meme1 = ? AND meme2 = ?", (m1, m2))
        if not await csr.fetchone():
            return m1, m2, iteration

@routes.get("/")
async def index(request):
    meme1, meme2, iteration = await get_pair(request.app["db"])
    return web.Response(text=f"""
<!DOCTYPE html>
<html>
    <title>Data Labelling Frontend (Not Evil)</title>
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
        <h1>Data Labelling Frontend (Not Evil)</h1>
        <form action="/rate" method="POST">
            <table>
            <tr>
            <td><input type="radio" name="rating-useful" value="1" id="rq1"> <label for="rq1">LHS is better (useful)</label></td>
            <td><input type="radio" name="rating-useful" value="eq" id="rqe"> <label for="rqe">Tie</label></td>
            <td><input type="radio" name="rating-useful" value="2" id="rq2"> <label for="rq2">RHS is better (useful)</label></td>
            </tr>
            <tr>
            <td><input type="radio" name="rating-meme" value="1" id="rm1"> <label for="rm1">LHS is better (memetically)</label></td>
            <td><input type="radio" name="rating-meme" value="eq" id="rme"> <label for="rme">Tie</label></td>
            <td><input type="radio" name="rating-meme" value="2" id="rm2"> <label for="rm2">RHS is better (memetically)</label></td>
            </tr>
            <tr>
            <td><input type="radio" name="rating-aesthetic" value="1" id="ra1"> <label for="ra1">LHS is better (aesthetically)</label></td>
            <td><input type="radio" name="rating-aesthetic" value="eq" id="rae"> <label for="rae">Tie</label></td>
            <td><input type="radio" name="rating-aesthetic" value="2" id="ra2"> <label for="ra2">RHS is better (aesthetically)</label></td>
            </td>
            </table>

            <input type="hidden" name="meme1" value="{meme1}">
            <input type="hidden" name="meme2" value="{meme2}">
            <input type="hidden" name="iteration" value="{str(iteration or 0)}">
            <input type="submit" value="Submit">
            <div class="memes">
                <img src="{meme1}" id="meme1">
                <img src="{meme2}" id="meme2">
            </div>
        </form>
        <script>
            const commitIfReady = () => {{
                if (document.querySelector("input[name='rating-useful']:checked") && document.querySelector("input[name='rating-meme']:checked") && document.querySelector("input[name='rating-aesthetic']:checked")) {{
                    document.querySelector("form").submit()
                }}
            }}
            document.addEventListener("keypress", function(event) {{
                if (event.key === "q") {{
                    document.querySelector("input[name='rating-useful'][value='1']").checked = true
                    commitIfReady()
                }} else if (event.key === "w") {{
                    document.querySelector("input[name='rating-useful'][value='eq']").checked = true
                    commitIfReady()
                }} else if (event.key === "e") {{
                    document.querySelector("input[name='rating-useful'][value='2']").checked = true
                    commitIfReady()
                }} else if (event.key === "a") {{
                    document.querySelector("input[name='rating-meme'][value='1']").checked = true
                    commitIfReady()
                }} else if (event.key === "s") {{
                    document.querySelector("input[name='rating-meme'][value='eq']").checked = true
                    commitIfReady()
                }} else if (event.key === "d") {{
                    document.querySelector("input[name='rating-meme'][value='2']").checked = true
                    commitIfReady()
                }} else if (event.key === "z") {{
                    document.querySelector("input[name='rating-aesthetic'][value='1']").checked = true
                    commitIfReady()
                }} else if (event.key === "x") {{
                    document.querySelector("input[name='rating-aesthetic'][value='eq']").checked = true
                    commitIfReady()
                }} else if (event.key === "c") {{
                    document.querySelector("input[name='rating-aesthetic'][value='2']").checked = true
                    commitIfReady()
                }}
            }});
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
    iteration = post["iteration"]
    rating = post["rating-useful"] + "," + post["rating-meme"] + "," + post["rating-aesthetic"]
    await db.execute("INSERT INTO ratings (meme1, meme2, rating, iteration, ip) VALUES (?, ?, ?, ?, ?)", (meme1, meme2, rating, iteration, request.remote))
    await db.execute("DELETE FROM queue WHERE meme1 = ? AND meme2 = ?", (meme1, meme2))
    await db.commit()
    return web.HTTPFound("/")

async def main():
    app["db"] = await aiosqlite.connect(DATABASE)
    await app["db"].executescript("""
CREATE TABLE IF NOT EXISTS ratings (
    meme1 TEXT NOT NULL,
    meme2 TEXT NOT NULL,
    rating TEXT NOT NULL,
    iteration TEXT,
    ip TEXT,
    UNIQUE (meme1, meme2)
);
CREATE TABLE IF NOT EXISTS queue (
    meme1 TEXT NOT NULL,
    meme2 TEXT NOT NULL,
    iteration TEXT NOT NULL,
    UNIQUE (meme1, meme2, iteration)
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
