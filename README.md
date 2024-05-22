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

This is untested. It might work. The new Rust version simplifies some steps (it integrates its own thumbnailing).

* Serve your meme library from a static webserver.
    * I use nginx. If you're in a hurry, you can use `python -m http.server`.
* Install Python dependencies with `pip` from `requirements.txt` (the versions probably shouldn't need to match exactly if you need to change them; I just put in what I currently have installed).
    * ~~You now need a [patched version](https://github.com/osmarks/transformers-patch-siglip) of `transformers` due to SigLIP support.~~ OpenCLIP supports SigLIP. I am now using that.
    * ~~I have converted exactly one SigLIP model: [https://huggingface.co/gollark/siglip-so400m-14-384](https://huggingface.co/gollark/siglip-so400m-14-384). It's apparently the best one. If you don't like it, find out how to convert more. You need to download that repo.~~ You can use any OpenCLIP model which OpenCLIP supports.
* Run `thumbnailer.py` (periodically, at the same time as index reloads, ideally)
* Run `clip_server.py` (as a background service).
    * It is configured with a JSON file given to it as its first argument. An example is in `clip_server_config.json`.
        * `device` should probably be `cuda` or `cpu`. The model will run on here.
        * `model` is ~~the [OpenCLIP](https://github.com/mlfoundations/open_clip) model to use~~ the path to the SigLIP model repository.
        * `model_name` is the name of the model for metrics purposes.
        * `max_batch_size` controls the maximum allowed batch size. Higher values generally result in somewhat better performance (the bottleneck in most cases is elsewhere right now though) at the cost of higher VRAM use.
        * `port` is the port to run the HTTP server on.
* Build and run `meme-search-engine` (Rust) (also as a background service).
    * This needs to be exposed somewhere the frontend can reach it. Configure your reverse proxy appropriately.
    * It has a JSON config file as well.
        * `clip_server` is the full URL for the backend server.
        * `db_path` is the path for the SQLite database of images and embedding vectors.
        * `files` is where meme files will be read from. Subdirectories are indexed.
        * `port` is the port to serve HTTP on.
        * If you are deploying to the public set `enable_thumbs` to `true` to serve compressed images.
* Build clipfront2, host on your favourite static webserver.
    * `npm install`, `node src/build.js`.
    * You will need to rebuild it whenever you edit `frontend_config.json`.
        * `image_path` is the base URL of your meme webserver (with trailing slash).
        * `backend_url` is the URL `mse.py` is exposed on (trailing slash probably optional).
* If you want, configure Prometheus to monitor `clip_server.py`.

## MemeThresher

See [here](https://osmarks.net/memethresher/) for information on MemeThresher, the new automatic meme acquisition/rating system (under `meme-rater`). Deploying it yourself is anticipated to be somewhat tricky but should be roughly doable:

1. Edit `crawler.py` with your own source and run it to collect an initial dataset.
2. Run `mse.py` with a config file like the provided one to index it.
3. Use `rater_server.py` to collect an initial dataset of pairs.
4. Copy to a server with a GPU and use `train.py` to train a model. You might need to adjust hyperparameters since I have no idea which ones are good.
5. Use `active_learning.py` on the best available checkpoint to get new pairs to rate.
6. Use `copy_into_queue.py` to copy the new pairs into the `rater_server.py` queue.
7. Rate the resulting pairs.
8. Repeat 4 through 7 until you feel good enough about your model.
9. Deploy `library_processing_server.py` and schedule `meme_pipeline.py` to run periodically.

## Scaling

Meme Search Engine uses an in-memory FAISS index to hold its embedding vectors, because I was lazy and it works fine (~100MB total RAM used for my 8000 memes). If you want to store significantly more than that you will have to switch to a more efficient/compact index (see [here](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)). As vector indices are held exclusively in memory, you will need to either persist them to disk or use ones which are fast to build/remove from/add to (presumably PCA/PQ indices). At some point if you increase total traffic the CLIP model may also become a bottleneck, as I also have no batching strategy. Indexing is currently GPU-bound since the new model appears somewhat slower at high batch sizes and I improved the image loading pipeline. You may also want to scale down displayed memes to cut bandwidth needs.