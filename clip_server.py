import torch
import time
import threading
from aiohttp import web
import aiohttp
import asyncio
import traceback
import umsgpack
import collections
import queue
import open_clip
from PIL import Image
from prometheus_client import Counter, Histogram, REGISTRY, generate_latest
import io
import json
import sys

with open(sys.argv[1], "r") as config_file:
    CONFIG = json.load(config_file)

device = torch.device(CONFIG["device"])
model, _, preprocess = open_clip.create_model_and_transforms(CONFIG["model"], device=device, pretrained=dict(open_clip.list_pretrained())[CONFIG["model"]], precision="fp16")
model.eval()
tokenizer = open_clip.get_tokenizer(CONFIG["model"])
print("Model loaded")

BS = CONFIG["max_batch_size"]
MODELNAME = CONFIG["model_name"]

InferenceParameters = collections.namedtuple("InferenceParameters", ["text", "images", "callback"])

items_ctr = Counter("modelserver_total_items", "Items run through model server", ["model", "modality"])
inference_time_hist = Histogram("modelserver_inftime", "Time running inference", ["model", "batch_size"])
batch_count_ctr = Counter("modelserver_batchcount", "Inference batches run", ["model"])

torch.set_grad_enabled(False)
def do_inference(params: InferenceParameters):
    with torch.no_grad():
        try:
            text, images, callback = params
            if text is not None:
                items_ctr.labels(MODELNAME, "text").inc(text.shape[0])
                with inference_time_hist.labels(MODELNAME + "-text", text.shape[0]).time():
                    features = model.encode_text(text)
            elif images is not None:
                with inference_time_hist.labels(MODELNAME + "-image", images.shape[0]).time():
                    items_ctr.labels(MODELNAME, "image").inc(images.shape[0])
                features = model.encode_image(images)
            batch_count_ctr.labels(MODELNAME).inc()
            features /= features.norm(dim=-1, keepdim=True)
            callback(True, features.cpu().numpy())
        except Exception as e:
            traceback.print_exc()
            callback(False, str(e))
        finally:
            torch.cuda.empty_cache()

iq = queue.Queue(10)
def infer_thread():
    while True:
        do_inference(iq.get())

pq = queue.Queue(10)
def preprocessing_thread():
    while True:
        text, images, callback = pq.get()
        try:
            if text:
                assert len(text) <= BS, f"max batch size is {BS}"
                text = tokenizer(text).to(device)
            elif images:
                assert len(images) <= BS, f"max batch size is {BS}"
                images = torch.stack([ preprocess(Image.open(io.BytesIO(im))).half() for im in images ]).to(device)
            else:
                assert False, "images or text required"
            iq.put(InferenceParameters(text, images, callback))
        except Exception as e:
            traceback.print_exc()
            callback(False, str(e))

app = web.Application(client_max_size=2**26)
routes = web.RouteTableDef()

@routes.post("/")
async def run_inference(request):
    loop = asyncio.get_event_loop()
    data = umsgpack.loads(await request.read())
    event = asyncio.Event()
    results = None
    def callback(*argv):
        loop.call_soon_threadsafe(lambda: event.set())
        nonlocal results
        results = argv
    pq.put_nowait(InferenceParameters(data.get("text"), data.get("images"), callback))
    await event.wait()
    body_data = results[1]
    if results[0]:
        status = 200
        body_data = [x.astype("float16").tobytes() for x in body_data]
    else:
        status = 500
        print(results[1])
    return web.Response(body=umsgpack.dumps(body_data), status=status, content_type="application/msgpack")

@routes.get("/config")
async def config(request):
    return web.Response(body=umsgpack.dumps({
        "model": CONFIG["model"],
        "batch": BS,
        "image_size": model.visual.image_size,
        "embedding_size": model.visual.output_dim
    }), status=200, content_type="application/msgpack")

@routes.get("/")
async def health(request):
    return web.Response(status=204)

@routes.get("/metrics")
async def metrics(request):
    return web.Response(body=generate_latest(REGISTRY))

app.router.add_routes(routes)

async def run_webserver():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "", CONFIG["port"])
    print("Ready")
    await site.start()

try:
    th = threading.Thread(target=infer_thread)
    th.start()
    th = threading.Thread(target=preprocessing_thread)
    th.start()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_webserver())
    loop.run_forever()
except KeyboardInterrupt:
    import sys
    sys.exit(0)