"""
Implementation of the Transformer models for dynamical systems. Derived from Karpathy's nanoGPT
https://github.com/karpathy/nanoGPT/
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    seq_len_ctx: int = 128
    seq_len_new: int = 128
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_u: int = 1
    n_y: int = 1
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):

    def __init__(self, config, causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.mha = nn.MultiheadAttention(config.n_embd, config.n_embd, dropout=config.dropout, batch_first=True)
        self.causal = causal
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        #if self.causal:
        #    self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(config.seq_len_ctx))

    def forward(self, x):
        if self.causal:
            seq_len = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            x = self.mha(x, x, x, attn_mask=mask, is_causal=True)[0]
        else:
            x = self.mha(x, x, x, is_causal=False)[0]
        y = self.resid_dropout(self.c_proj(x))
        return y


class CrossAttention(nn.Module):

    def __init__(self, config, causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # multi-head attention
        self.mha = nn.MultiheadAttention(config.n_embd, config.n_embd, dropout=config.dropout, batch_first=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.causal = causal

    def forward(self, x, mem):
        x = self.mha(x, mem, mem, is_causal=self.causal)[0]
        y = self.resid_dropout(self.c_proj(x))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.self_attn = SelfAttention(config, causal=False)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerDecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.self_attn = SelfAttention(config, causal=True)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mem):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), mem)
        x = x + self.mlp(self.ln_3(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.seq_len_ctx is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Linear(config.n_u + config.n_y, config.n_embd),  # we process continuous data
            wpe=nn.Embedding(config.seq_len_ctx, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerEncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_y, bias=True)  # False

    def forward(self, y_ctx, u_ctx):
        device = y_ctx.device
        b, t, ny = y_ctx.size()
        bb, tt, nu = u_ctx.size()
        assert (b == bb) and (t == tt)
        assert t <= self.config.seq_len_ctx, f"Cannot forward sequence of length {t}, block size is only {self.config.seq_len_ctx}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        batch_uy = torch.cat((u_ctx, y_ctx), dim=-1)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(batch_uy)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # final layer normalization

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.seq_len_new is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Linear(config.n_u, config.n_embd),  # we process continuous data
            wpe=nn.Embedding(config.seq_len_new, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerDecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_y, bias=True)  # False

    def forward(self, u_new, z_context):
        device = u_new.device
        b, t, nu = u_new.size()
        bb, tt, ny = z_context.size()
        assert (b == bb)
        assert t <= self.config.seq_len_ctx, f"Cannot forward sequence of length {t}, block size is only {self.config.seq_len_ctx}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(u_new)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, z_context)
        x = self.transformer.ln_f(x)

        y_sim = self.lm_head(x)
        return y_sim


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

    def forward(self, y_train, u_train, u_test):
        train_embd = self.encoder(y_train, u_train)
        y_test_hat = self.decoder(u_test, train_embd)
        return y_test_hat


def warmup_cosine_lr(iter, lr, min_lr, warmup_iters, lr_decay_iters):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return lr * iter / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)


if __name__ == "__main__":

    batch_size = 8
    seq_len_ctx = 256
    seq_len_new = 128
    n_u = 2
    n_y = 3

    model_cfg = Config(seq_len_ctx=seq_len_ctx, seq_len_new=seq_len_new, n_u=n_u, n_y=n_y)
    model = EncoderDecoderTransformer(model_cfg)

    batch_u = torch.randn((batch_size, seq_len_ctx, n_u))
    batch_y = torch.randn((batch_size, seq_len_ctx, n_y))
    batch_yu = torch.cat((batch_y, batch_u), dim=-1)
    batch_u_new = torch.randn((batch_size, seq_len_new, n_u))

    batch_z = model.encoder(batch_y, batch_u)
    batch_y_sim = model.decoder(batch_u_new, batch_z)
