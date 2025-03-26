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

New: somewhat tested Dockerized setup under /docker/. Note that you need a model like https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384 (for timm), not one packaged for other libraries.

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

Note: this has now been superseded by various changes made for the [scaled run](https://osmarks.net/memescale/). Use commit [512b776](https://github.com/osmarks/meme-search-engine/commit/512b776e10a2921b830cda478884d674ccdf1856) for old version.

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

This repository contains both the small-scale personal meme search system, which uses an in-memory FAISS index which scales to perhaps ~1e5 items (more with some minor tweaks) and the [larger-scale](https://osmarks.net/memescale/) version based on [DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf), which should work up to about ~1e9 and has been tested at hundred-million-item scale. The larger version (accessible [here](https://nooscope.osmarks.net)) is trickier to use. You will have to, roughly:

* Download a [Reddit scrape dataset](https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13). You can use other things, obviously, but then you would have to edit the code. Contact me if you're interested.
* Build and run `reddit-dump` to do the download. It will emit zstandard-compressed msgpack files of records containing metadata and embeddings. This may take some time. Configuration is done by editing `reddit_dump.rs`. I don't currently have a way to run the downloads across multiple machines, but you can put the CLIP backend behind a load balancer.
    * Use `genseahash.py` to generate hashes of files to discard.
* Build and run `dump-processor -s` on the resulting zstandard files to generate a sample of ~1 million embeddings to train the OPQ codec and clusters.
* Use `dump-processor -t` to generate a sample of titles and `generate_queries_bin.py` to generate text embeddings.
* Use `kmeans.py` with the appropriate `n_clusters` to generate centroids to cluster your dataset such that each of the clusters fits in your available RAM (with some spare space). Note that each vector goes to *two* clusters.
* Use `aopq_train.py` to train the OPQ codec on your embeddings sample and queries. A GPU is recommended, as I bruteforced some of the problems involved with computing time.
* Use `dump-processor -C [centroids] -S [shards folder]` to split your dataset into shards.
    * The shards will be about twice the size of the dump files.
    * You may want to do filtering at this point. I don't have a provision to do proper deduplication or aesthetic filtering at this stage but you *can* use `-E` to filter by similarity to embeddings. Make sure to use the same filtering settings here as you do later, so that the IDs match.
* Build `generate-index-shard` and run it on each shard with the queries file from earlier. This will generate small files containing the graph structure.
* Use `dump-processor -s [some small value] -j` to generate a sample of embeddings and metadata to train a quality model.
* Use `meme-rater/load_from_json.py` to initialize the rater database from the resulting JSON.
* Use `meme-rater/rater_server.py` to do data labelling. Its frontend has keyboard controls (QWERT for usefulness, ASDFG for memeness, ZXCVB for aesthetics). You will have to edit this file if you want to rate on other dimensions.
* Use `meme-rater/train.py` to configure and train a quality model. If you edit the model config here, edit the other scripts, since I never centralized this.
    * Use one of the `meme-rater/active_learning_*.py` scripts to select high-variance/high-gradient/high-quality samples to label and `copy_into_queue.py` to copy them into the queue.
    * Do this repeatedly until you like the model.
    * You can also use a pretrained checkpoint from [here](https://datasets.osmarks.net/projects-formerly-codenamed-radius-tyrian-phase-ii/).
* Use `meme-rater/ensemble_to_wide_model.py` to export the quality model to be useable by the Rust code.
* Use `meme-rater/compute_cdf.py` to compute the distribution of quality for packing by `dump-processor`.
* Use `dump-processor -S [shards folder] -i [index folder] -M [score model] --cdfs [cdfs file]` to generate an index file from the shards, model and CDFs.
* Build and run `query-disk-index` and fill in its configuration file to serve the index.
* Build and run the frontend in `clipfront2`.
