import pyjson5
import re
import asyncio
import aiohttp
from PIL import Image
import time
import io

CALLBACK_REGEX = re.compile(r">AF_initDataCallback\(({key: 'ds:1'.*?)\);</script>")

def encode_img(img):
    image_bytes = io.BytesIO()
    img.save(image_bytes, format="PNG", compress_level=6)
    return image_bytes.getvalue()

def rationalize_coords_format1(image_w, image_h, center_x_fraction, center_y_fraction, width_fraction, height_fraction, mysterious):
    return {
        "x": round((center_x_fraction - width_fraction / 2) * image_w),
        "y": round((center_y_fraction - height_fraction / 2) * image_h),
        "w": round(width_fraction * image_w),
        "h": round(height_fraction * image_h)
    }

async def scan_image_chunk(sess, image):
    timestamp = int(time.time() * 1000)
    url = f"https://lens.google.com/v3/upload?stcs={timestamp}"
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 13; RMX3771) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.144 Mobile Safari/537.36"}
    cookies = {"SOCS": "CAESEwgDEgk0ODE3Nzk3MjQaAmVuIAEaBgiA_LyaBg"}

    # send data to inscrutable undocumented Google service
    # https://github.com/AuroraWright/owocr/blob/master/owocr/ocr.py#L193
    async with aiohttp.ClientSession() as sess:
        data = aiohttp.FormData()
        data.add_field(
            "encoded_image",
            encode_img(image),
            filename="ocr" + str(timestamp) + ".png",
            content_type="image/png"
        )
        async with sess.post(url, headers=headers, cookies=cookies, data=data, timeout=10) as res:
            body = await res.text()

    # I really worry about Google sometimes. This is not a sensible format.
    match = CALLBACK_REGEX.search(body)
    if match == None:
        raise ValueError("Invalid callback")

    lens_object = pyjson5.loads(match.group(1))
    if "errorHasStatus" in lens_object:
        raise RuntimeError("Lens failed")

    text_segments = []
    text_regions = []

    root = lens_object["data"]

    # I don't know why Google did this.
    # Text segments are in one place and their locations are in another, using a very strange coordinate system.
    # At least I don't need whatever is contained in the base64 parts (which I assume are protobufs).
    # TODO: on a few images, this seems to not work for some reason.
    try:
        text_segments = root[3][4][0][0]
        text_regions = [ rationalize_coords_format1(image.width, image.height, *x[1]) for x in root[2][3][0] if x[11].startswith("text:") ]
    except (KeyError, IndexError):
        # https://github.com/dimdenGD/chrome-lens-ocr/blob/main/src/core.js#L316 has code for a fallback text segment read mode.
        # In testing, this proved unnecessary (quirks of the HTTP request? I don't know), and this only happens on textless images.
        return [], []

    return text_segments, text_regions

MAX_SCAN_DIM = 1000 # not actually true but close enough
def chunk_image(image: Image):
    chunks = []
    # Cut image down in X axis (I'm assuming images aren't too wide to scan in downscaled form because merging text horizontally would be annoying)
    if image.width > MAX_SCAN_DIM:
        image = image.resize((MAX_SCAN_DIM, round(image.height * (image.width / MAX_SCAN_DIM))), Image.LANCZOS)
    for y in range(0, image.height, MAX_SCAN_DIM):
        chunks.append(image.crop((0, y, image.width, min(y + MAX_SCAN_DIM, image.height))))
    return chunks

async def scan_chunks(sess: aiohttp.ClientSession, chunks: [Image]):
    # If text happens to be split across the cut line it won't get read.
    # This is because doing overlap read areas would be really annoying.
    text = ""
    regions = []
    for chunk in chunks:
        new_segments, new_regions = await scan_image_chunk(sess, chunk)
        for segment in new_segments:
            text += segment + "\n"
        for i, (segment, region) in enumerate(zip(new_segments, new_regions)):
            regions.append({ **region, "y": region["y"] + (MAX_SCAN_DIM * i), "text": segment })
    return text, regions

async def scan_image(sess: aiohttp.ClientSession, image: Image):
    return await scan_chunks(sess, chunk_image(image))

if __name__ == "__main__":
    async def main():
        async with aiohttp.ClientSession() as sess:
            print(await scan_image(sess, Image.open("/data/public/memes-or-something/linear-algebra-chess.png")))
    asyncio.run(main())