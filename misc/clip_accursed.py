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
import numpy
import big_vision.models.proj.image_text.two_towers as model_mod
import jax
import jax.numpy as jnp
import ml_collections
import big_vision.pp.builder as pp_builder
import big_vision.pp.ops_general
import big_vision.pp.ops_image
import big_vision.pp.ops_text

with open(sys.argv[1], "r") as config_file:
    CONFIG = json.load(config_file)

# blatantly copypasted from colab
# https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb
VARIANT, RES = CONFIG["model"]
CKPT, TXTVARIANT, EMBDIM, SEQLEN, VOCAB = {
    ("So400m/14", 384): ("webli_en_so400m_384_58765454-fp16.safetensors", "So400m", 1152, 64, 32_000),
}[VARIANT, RES]

model_cfg = ml_collections.ConfigDict()
model_cfg.image_model = "vit"  # TODO(lbeyer): remove later, default
model_cfg.text_model = "proj.image_text.text_transformer"  # TODO(lbeyer): remove later, default
model_cfg.image = dict(variant=VARIANT, pool_type="map")
model_cfg.text = dict(variant=TXTVARIANT, vocab_size=VOCAB)
model_cfg.out_dim = (None, EMBDIM)  # (image_out_dim, text_out_dim)
model_cfg.bias_init = -10.0
model_cfg.temperature_init = 10.0

model = model_mod.Model(**model_cfg)

init_params = None  # sanity checks are a low-interest-rate phenomenon
model_params = model_mod.load(init_params, f"{CKPT}", model_cfg) # assume path

pp_img = pp_builder.get_preprocess_fn(f"resize({RES})|value_range(-1, 1)")
TOKENIZERS = {
    32_000: "c4_en",
    250_000: "mc4",
}
pp_txt = pp_builder.get_preprocess_fn(f'tokenize(max_len={SEQLEN}, model="{TOKENIZERS[VOCAB]}", eos="sticky", pad_value=1, inkey="text")')
print("Model loaded")

BS = CONFIG["max_batch_size"]
MODELNAME = CONFIG["model_name"]

InferenceParameters = collections.namedtuple("InferenceParameters", ["text", "images", "callback"])

items_ctr = Counter("modelserver_total_items", "Items run through model server", ["model", "modality"])
inference_time_hist = Histogram("modelserver_inftime", "Time running inference", ["model", "batch_size"])
batch_count_ctr = Counter("modelserver_batchcount", "Inference batches run", ["model"])

@jax.jit
def run_text_model(text_batch):
    _, features, out = model.apply({"params": model_params}, None, text_batch)
    return features

@jax.jit
def run_image_model(image_batch):
    features, _, out = model.apply({"params": model_params}, image_batch, None)
    return features

def round_down_to_power_of_two(x):
    return 1<<(x.bit_length()-1)

def minimize_jits(fn, batch):
    out = numpy.zeros((batch.shape[0], EMBDIM), dtype="float16")
    i = 0
    while True:
        batch_dim = batch.shape[0]
        s = round_down_to_power_of_two(batch_dim)
        fst = batch[:s,...]
        out[i:(i + s), ...] = fn(fst)
        i += s
        batch = batch[s:, ...]
        if batch.shape[0] == 0: break
    return out

def do_inference(params: InferenceParameters):
    try:
        text, images, callback = params
        if text is not None:
            items_ctr.labels(MODELNAME, "text").inc(text.shape[0])
            with inference_time_hist.labels(MODELNAME + "-text", text.shape[0]).time():
                features = run_text_model(text)
        elif images is not None:
            items_ctr.labels(MODELNAME, "image").inc(images.shape[0])
            with inference_time_hist.labels(MODELNAME + "-image", images.shape[0]).time():
                features = run_image_model(images)
        batch_count_ctr.labels(MODELNAME).inc()
        # TODO got to divide somewhere
        callback(True, numpy.asarray(features))
    except Exception as e:
        traceback.print_exc()
        callback(False, str(e))

iq = queue.Queue(100)
def infer_thread():
    while True:
        do_inference(iq.get())

pq = queue.Queue(100)
def preprocessing_thread():
    while True:
        text, images, callback = pq.get()
        try:
            if text:
                assert len(text) <= BS, f"max batch size is {BS}"
                # I feel like this ought to be batchable but I can't see how to do that
                text = numpy.array([pp_txt({"text": text})["labels"] for text in text])
            elif images:
                assert len(images) <= BS, f"max batch size is {BS}"
                images = numpy.array([pp_img({"image": numpy.array(Image.open(io.BytesIO(image)).convert("RGB"))})["image"] for image in images])
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
        "model": CONFIG["model"],
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