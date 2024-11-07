import torch
import time
import threading
from aiohttp import web
import aiohttp
import asyncio
import traceback
import msgpack
import collections
import queue
import open_clip
from PIL import Image
from prometheus_client import Counter, Histogram, REGISTRY, generate_latest
import io
import json
import torchvision.transforms.transforms as transforms
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

fast_image_fns = {}
# ugly hack, sorry
if CONFIG.get("aitemplate_image_models"):
    from aitemplate.compiler import Model
    from aitemplate.testing import detect_target

    USE_CUDA = detect_target().name() == "cuda"

    state = model.state_dict()
    conv_weights = state["visual.trunk.patch_embed.proj.weight"].permute((0, 2, 3, 1)).contiguous().cuda().half()

    def load_pretrained():
        params = {}
        for key, value in state.items():
            orig_key = key
            if key.startswith("visual."):
                key = key.removeprefix("visual.") \
                    .replace("trunk.patch_embed", "patch_embed") \
                    .replace("trunk.blocks", "encoder.layers") \
                    .replace(".attn.", ".mha.") \
                    .replace(".norm1.", ".ln1.") \
                    .replace(".norm2.", ".ln2.") \
                    .replace("trunk.pos_embed", "pos_emb_pos_emb") \
                    .replace("trunk.norm.", "encoder.ln.") \
                    .replace("trunk.attn_pool.latent", "pool.probe") \
                    .replace("trunk.attn_pool", "pool") \
                    .replace("pool.norm", "pool.ln")
                if "patch_embed.proj.weight" not in key:
                    params[key.replace(".", "_")] = value.cuda()
                    #print(orig_key, key.replace(".", "_"))

        params["patch_embed_proj_weight"] = conv_weights

        return params

    def generate_wrapper(path):
        ait_model = Model(path)
        ait_model.set_many_constants_with_tensors(load_pretrained())
        ait_model.fold_constants(sync=True)
        def wrapper(batch):
            xs = [batch.permute((0, 2, 3, 1)).contiguous()]
            ys = []
            for i in range(len(ait_model.get_output_name_to_index_map())):
                shape = ait_model.get_output_maximum_shape(i)
                ys.append(torch.empty(shape).cuda().half())
            ait_model.run_with_tensors(xs, ys)
            return ys[0][:, 0, :]
        return wrapper

    for batch_size, path in CONFIG["aitemplate_image_models"]:
        fast_image_fns[batch_size] = generate_wrapper(path)
        print("loaded", batch_size, path)

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
                    features /= features.norm(dim=-1, keepdim=True)
                    features = features.cpu().numpy()
            elif images is not None:
                with inference_time_hist.labels(MODELNAME + "-image", images.shape[0]).time():
                    items_ctr.labels(MODELNAME, "image").inc(images.shape[0])
                    batch = images.shape[0]
                    if fast_image_fns:
                        progress = 0
                        features = torch.zeros((batch, model.text.text_projection.out_features))
                        while progress < batch:
                            biggest_available = max(x for x in fast_image_fns.keys() if x <= (batch - progress))
                            chunk = fast_image_fns[biggest_available](images[progress:progress + biggest_available])
                            features[progress:progress + biggest_available] = chunk
                            progress += biggest_available
                    else:
                        features = model.encode_image(images)
                    features /= features.norm(dim=-1, keepdim=True)
                    features = features.cpu().numpy()
            batch_count_ctr.labels(MODELNAME).inc()
            callback(True, features)
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
    data = msgpack.loads(await request.read())
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
    return web.Response(body=msgpack.dumps(body_data), status=status, content_type="application/msgpack")

@routes.get("/config")
async def config(request):
    return web.Response(body=msgpack.dumps({
        "model": CONFIG["model"],
        "batch": BS,
        "image_size": [ t for t in preprocess.transforms if isinstance(t, transforms.Resize) ][0].size,
        "embedding_size": model.text.text_projection.out_features
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
