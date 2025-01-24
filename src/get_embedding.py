import json
import requests
import base64
import msgpack
import sys

with open("mse_config.json") as f:
    config = json.load(f)

def get_embedding(req):
    return msgpack.unpackb(requests.post(config["clip_server"], data=msgpack.packb(req)).content)

mode, output, input = sys.argv[1:]

with open(output, "wb") as f:
    if mode == "image":
        with open(input, "rb") as g:
            input_data = g.read()
        result = get_embedding({"images": [input_data]})[0]
    elif mode == "text":
        result = get_embedding({"text": input})[0]
    else:
        raise Exception("unknown mode")
    f.write(result)
    print(base64.urlsafe_b64encode(result).decode("ascii"))
