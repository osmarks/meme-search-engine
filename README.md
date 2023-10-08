# Meme Search Engine

Do you have a large folder of memes you want to search semantically? Do you have a Linux server with an Nvidia GPU? You do; this is now mandatory.

## Features

They say a picture is worth a thousand words. Unfortunately, many (most?) sets of words cannot be adequately described by pictures. Regardless, here is a picture. You can use a running instance [here](https://mse.osmarks.net/).

![Meme Search Engine's frontend.](/demo-image.png)

* Infinite-scroll masonry UI for dense meme viewing.
* Online reindexing (a good reason to use it over [clip-retrieval](https://github.com/rom1504/clip-retrieval)) - reload memes without a slow expensive rebuild step.
* Complex query support - query using text and images, including multiple terms at once, with weighting (including negative).
* Reasonably fast.

## Setup

This is untested. It might work.

* Serve your meme library from a static webserver.
    * I use nginx. If you're in a hurry, you can use `python -m http.server`.
* Install Python dependencies with `pip` from `requirements.txt` (the versions probably shouldn't need to match exactly if you need to change them; I just put in what I currently have installed).
    * You now need a [patched version](https://github.com/osmarks/transformers-patch-siglip) of `transformers` due to SigLIP support.
    * I have converted exactly one SigLIP model: [https://huggingface.co/gollark/siglip-so400m-14-384](https://huggingface.co/gollark/siglip-so400m-14-384). It's apparently the best one. If you don't like it, find out how to convert more. You need to download that repo.
* Run `clip_server.py` (as a background service).
    * It is configured with a JSON file given to it as its first argument. An example is in `clip_server_config.json`.
        * `device` should probably be `cuda` or `cpu`. The model will run on here.
        * `model` is ~~the [OpenCLIP](https://github.com/mlfoundations/open_clip) model to use~~ the path to the SigLIP model repository.
        * `model_name` is the name of the model for metrics purposes.
        * `max_batch_size` controls the maximum allowed batch size. Higher values generally result in somewhat better performance (the bottleneck in most cases is elsewhere right now though) at the cost of higher VRAM use.
        * `port` is the port to run the HTTP server on.
* Run `mse.py` (also as a background service).
    * This needs to be exposed somewhere the frontend can reach it. Configure your reverse proxy appropriately.
    * It has a JSON config file as well.
        * `clip_server` is the full URL for the backend server.
        * `db_path` is the path for the SQLite database of images and embedding vectors.
        * `files` is where meme files will be read from. Subdirectories are indexed.
        * `port` is the port to serve HTTP on.
* Build clipfront2, host on your favourite static webserver.
    * `npm install`, `node src/build.js`.
    * You will need to rebuild it whenever you edit `frontend_config.json`.
        * `image_path` is the base URL of your meme webserver (with trailing slash).
        * `backend_url` is the URL `mse.py` is exposed on (trailing slash probably optional).
* If you want, configure Prometheus to monitor `mse.py` and `clip_server.py`.

## Scaling

Meme Search Engine uses an in-memory FAISS index to hold its embedding vectors, because I was lazy and it works fine (~100MB total RAM used for my 8000 memes). If you want to store significantly more than that you will have to switch to a more efficient/compact index (see [here](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)). As vector indices are held exclusively in memory, you will need to either persist them to disk or use ones which are fast to build/remove from/add to (presumably PCA/PQ indices). At some point if you increase total traffic the CLIP model may also become a bottleneck, as I also have no batching strategy. Indexing appears to actually be CPU-bound (specifically, it's limited by single-threaded image decoding and serialization) - improving that would require a lot of redesigns so I haven't. You may also want to scale down displayed memes to cut bandwidth needs.