import json
import requests
import base64
import msgpack
import sys

with open("mse_config.json") as f:
    config = json.load(f)

def get_embedding(req):
    return msgpack.unpackb(requests.post(config["clip_server"], data=msgpack.packb(req)).content)

output, input, *xs = sys.argv[1:]

with open(output, "wb") as f:
    with open(input, "rb") as g:
        input_data = g.read()
    if not xs:
        result = get_embedding({"images": [input_data]})[0]
    else:
        result = get_embedding({"text": xs})[0]
    f.write(result)
    print(base64.urlsafe_b64encode(result).decode("ascii"))
