import os
import time
import threading
from aiohttp import web
import aiohttp
import asyncio
import traceback
import umsgpack
import collections
import queue
from PIL import Image
from prometheus_client import Counter, Histogram, REGISTRY, generate_latest
import io
import json
import sys
import torch
from transformers import SiglipImageProcessor, T5Tokenizer, SiglipModel, SiglipConfig
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors import safe_open
import numpy

with open(sys.argv[1], "r") as config_file:
    CONFIG = json.load(config_file)

DEVICE = CONFIG["device"]

# So400m/14@384
with init_empty_weights():
    model = SiglipModel(config=SiglipConfig.from_pretrained(CONFIG["model"])).half().eval()
with safe_open(os.path.join(CONFIG["model"], "model.safetensors"), framework="pt", device=DEVICE) as f:
    for key in f.keys():
        set_module_tensor_to_device(model, key, device=DEVICE, value=f.get_tensor(key))
model = model.to(DEVICE)
EMBDIM = model.config.vision_config.hidden_size # NOT projection_dim, why is that even there
RES = model.config.vision_config.image_size
tokenizer = T5Tokenizer(vocab_file=os.path.join(CONFIG["model"], "sentencepiece.model"), extra_ids=0, model_max_length=64, pad_token="</s>", legacy=False)
image_processor = SiglipImageProcessor(size={"height": RES, "width":RES})

BS = CONFIG["max_batch_size"]
MODELNAME = CONFIG["model_name"]

InferenceParameters = collections.namedtuple("InferenceParameters", ["text", "images", "callback"])

items_ctr = Counter("modelserver_total_items", "Items run through model server", ["model", "modality"])
inference_time_hist = Histogram("modelserver_inftime", "Time running inference", ["model", "batch_size"])
batch_count_ctr = Counter("modelserver_batchcount", "Inference batches run", ["model"])

def do_inference(params: InferenceParameters):
    with torch.no_grad():
        try:
            text, images, callback = params
            if text is not None:
                items_ctr.labels(MODELNAME, "text").inc(text.shape[0])
                with inference_time_hist.labels(MODELNAME + "-text", text.shape[0]).time():
                    features = model.text_model.forward(input_ids=torch.tensor(text, device=DEVICE)).pooler_output
            elif images is not None:
                items_ctr.labels(MODELNAME, "image").inc(images.shape[0])
                with inference_time_hist.labels(MODELNAME + "-image", images.shape[0]).time():
                    features = model.vision_model.forward(torch.tensor(images, device=DEVICE)).pooler_output
            features /= features.norm(dim=-1, keepdim=True)
            batch_count_ctr.labels(MODELNAME).inc()
            callback(True, features.cpu().numpy())
        except Exception as e:
            traceback.print_exc()
            callback(False, str(e))

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
                # I feel like this ought to be batchable but I can't see how to do that
                text = numpy.array(tokenizer(text, padding="max_length", truncation=True)["input_ids"])
            elif images:
                assert len(images) <= BS, f"max batch size is {BS}"
                images = numpy.array(image_processor([ Image.open(io.BytesIO(bs)) for bs in images ])["pixel_values"]).astype("float16")
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
        nonlocal results
        results = argv
        loop.call_soon_threadsafe(lambda: event.set())
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
        "model": MODELNAME,
        "batch": BS,
        "image_size": (RES, RES),
        "embedding_size": EMBDIM
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