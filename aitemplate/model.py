from functools import partial

from aitemplate.compiler import ops
from aitemplate.frontend import nn
from aitemplate.testing import detect_target

USE_CUDA = detect_target().name() == "cuda"

def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape

class MLPBlock(nn.Module):
    def __init__(self, emb_dim, mlp_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim
        self.fc1 = nn.Linear(emb_dim, mlp_dim, specialization="gelu")
        self.fc2 = nn.Linear(mlp_dim, emb_dim, specialization="add")

    def forward(self, x, res):
        x = self.fc1(x)
        x = self.fc2(x, res)
        return x

class Encoder1DBlock(nn.Module):
    def __init__(self, emb_dim, mlp_dim, num_heads, batch_size, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mha = nn.MultiheadAttention(
            emb_dim,
            batch_size,
            seq_len,
            num_heads,
            use_mem_eff=True
        )
        self.mlp = MLPBlock(emb_dim, mlp_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        #self_attention_input = self.ln1(x)
        x = self.mha(self.ln1(x), x)
        x = self.mlp(self.ln2(x), x)
        return x

class Encoder(nn.Module):
    def __init__(self, emb_dim, mlp_dim, num_heads, batch_size, seq_len, depth):
        super().__init__()
        self.layers = nn.ModuleList([ Encoder1DBlock(emb_dim, mlp_dim, num_heads, batch_size, seq_len) for i in range(depth) ])
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.ln(x)

class PositionalEmbeddings(nn.Module):
    def __init__(self, emb_dim, seq_len):
        super().__init__()
        self.pos_emb = nn.Parameter(shape=[1, seq_len, emb_dim], dtype="float16")

    def forward(self, x):
        return x + self.pos_emb.tensor()

class PatchEmbedder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, emb_dim):
        super().__init__()
        conv_op = nn.Conv2dBiasFewChannels if USE_CUDA else nn.Conv2dBias
        self.proj = conv_op(in_chans, emb_dim, kernel_size=patch_size, stride=patch_size, padding=0, auto_padding=False)
        self.flatten = True
        self.emb_dim = emb_dim
        self.proj_norm = nn.Identity()

    def forward(self, x):
        B, H, W, C = get_shape(x)
        x = self.proj(x)
        if self.flatten:
            x = ops.reshape()(x, [B, -1, self.emb_dim])
        x = self.proj_norm(x)
        return x

class MAPHead(nn.Module):
    def __init__(self, emb_dim, mlp_dim, num_heads, batch_size, seq_len):
        super().__init__()
        self.q = nn.Linear(emb_dim, emb_dim)
        self.kv = nn.Linear(emb_dim, emb_dim * 2)
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        #self.q_norm = nn.LayerNorm(self.head_dim)
        #self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.sdpa = nn.ScaledDotProductAttention()
        self.mlp = MLPBlock(emb_dim, mlp_dim)
        self.probe = nn.Parameter(shape=[1, 1, emb_dim], dtype="float16")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim

    def forward(self, x):
        ql = ops.expand()(self.probe.tensor(), [self.batch_size, -1, -1])
        q = ops.reshape()(self.q(ql), [self.batch_size, self.num_heads, 1, self.head_dim])
        kv = ops.permute()(ops.reshape()(self.kv(x), [self.batch_size, self.seq_len, 2, self.num_heads, self.head_dim]), (2, 0, 3, 1, 4))
        k, v = ops.split()(kv, [1, 1], dim=0)
        k, v = ops.squeeze(0)(k), ops.squeeze(0)(v)
        #q = self.q_norm(q)
        #k = self.k_norm(k)
        x = self.sdpa(q, k, v)
        x = ops.reshape()(ops.transpose()(x, 1, 2), (self.batch_size, 1, self.emb_dim))
        x = self.proj(x)
        return self.mlp(self.ln(x), x)

class VisionTransformer(nn.Module):
    def __init__(self, emb_dim, mlp_dim, num_heads, batch_size, seq_len, depth, img_size, patch_size, in_chans):
        super().__init__()
        self.patch_embed = PatchEmbedder(img_size, patch_size, in_chans, emb_dim)
        self.encoder = Encoder(emb_dim, mlp_dim, num_heads, batch_size, seq_len, depth)
        self.pool = MAPHead(emb_dim, mlp_dim, num_heads, batch_size, seq_len)
        self.pos_emb = PositionalEmbeddings(emb_dim, seq_len)

    def forward(self, image):
        x = self.pos_emb(self.patch_embed(image))
        return self.pool(self.encoder(x))