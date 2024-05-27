#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""benchmark for vit"""

import os

import numpy as np
import torch
from aitemplate.compiler import compile_model, Model
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
import open_clip

from PIL import Image
from model import VisionTransformer

model, _, preprocess = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli", precision="fp16", device="cuda")
model.eval()

torch.set_grad_enabled(False)

print(model.visual.trunk.patch_embed)

def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))

USE_CUDA = detect_target().name() == "cuda"

siglip_so400m_384_14 = {
    "img_size": 384,
    "emb_dim": 1152,
    "depth": 27,
    "num_heads": 16,
    "mlp_dim": 4304,
    "patch_size": 14,
    "in_chans": 3
}

batch_size = 32
image = preprocess(Image.open("/data/public/memes-or-something/0mg.jpg"))
input = torch.stack([image.cuda().half() for _ in range(batch_size)], dim=0)

def compile_vit(
    config,
    batch_size,
    use_fp16_acc=True,
):
    seq_len = (config["img_size"] // config["patch_size"]) ** 2
    ait_model = VisionTransformer(
        batch_size=batch_size,
        seq_len=seq_len,
        **config
    )
    ait_model.name_parameter_tensor()
    print(ait_model)
    inputs_ait = Tensor(
        [batch_size, config["img_size"], config["img_size"], config["in_chans"]], name="input0", is_input=True
    )
    Y = ait_model(inputs_ait)
    mark_output(Y)

    target = detect_target(use_fp16_acc=use_fp16_acc)
    exe_module = compile_model(
        Y, target, "./tmp", "vision_transformer_bs%d_seq%d" % (batch_size, seq_len)
    )
    return exe_module

def load_pretrained(config):
    params = {}
    st = model.state_dict()
    for key, value in st.items():
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
                print(orig_key, key.replace(".", "_"))
    if USE_CUDA:
        # horrors
        w_pad = torch.zeros((config["emb_dim"], config["patch_size"], config["patch_size"], 4)).cuda().half()
        w = st["visual.trunk.patch_embed.proj.weight"]#.permute((0, 2, 3, 1)).contiguous()
        params["patch_embed_proj_weight"] = w.permute((0, 2, 3, 1)).contiguous().cuda().half() # N H W C
    else:
        params["patch_embed_proj_weight"] = st["visual.trunk.patch_embed.proj.weight"].permute((0, 2, 3, 1)).contiguous().cuda().half()
    return params

def benchmark(name, config, batch_size, mod=None, graph_mode=True):
    seqlen = (config["img_size"] // config["patch_size"]) ** 2

    if mod is None:
        model_dir = f"vision_transformer_bs{batch_size}_seq{seqlen}"
        mod = Model(os.path.join("./tmp", model_dir, "test.so"))

    # prepare params
    params_ait = load_pretrained(config)

    s = set(mod.get_constant_names())
    d = []
    for k in params_ait:
        if k not in s:
            d.append(k)
    for x in d:
        del params_ait[x]

    mod.set_many_constants_with_tensors(params_ait)
    mod.fold_constants(sync=True)

    inputs = [torch.randn([batch_size, config["img_size"], config["img_size"], 3]).cuda().half()]
    ys = []
    num_outputs = len(mod.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = mod.get_output_maximum_shape(i)
        ys.append(torch.empty(shape).cuda().half())
    # warm up
    t, _, __ = mod.benchmark_with_tensors(
        inputs,
        ys,
        count=10,
        repeat=1,
        graph_mode=graph_mode,
    )
    #q = model.visual.trunk.attn_pool(model.visual.trunk.norm(model.visual.trunk.blocks(model.visual.trunk.patch_embed(input) + model.visual.trunk.pos_embed)))
    ## = #model.visual.trunk.attn_pool.q(model.visual.trunk.attn_pool.latent.expand(batch_size, -1, -1)).reshape(batch_size, 1, 16, 72).transpose(1, 2)
    #print("expected", q, q.shape)
    #print("actual", ys[0], ys[0].shape)
    """
    batch = ys[0][:, 0, :]
    batch = torch.nn.functional.normalize(batch, dim=-1)
    print(batch)
    print(f"batch_size: {batch_size}, latency: {t}")
    """
#for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256):
for bs in (1, 2, 4, 8, 16, 32):
    compile_vit(siglip_so400m_384_14, bs, use_fp16_acc=True)
    benchmark("siglip_so400m_384_14", siglip_so400m_384_14, bs, graph_mode=True)