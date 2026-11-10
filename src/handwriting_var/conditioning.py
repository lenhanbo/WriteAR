from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dimension=384, layers=4, heads=6, max_length=26, pad_id=0):
        super().__init__(); self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=pad_id)
        self.position = nn.Parameter(torch.randn(1, max_length, dimension) * .02)
        block = nn.TransformerEncoderLayer(dimension, heads, dimension * 4, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(dimension)
        self.null = nn.Parameter(torch.randn(1, 1, dimension) * .02)
    def forward(self, ids, drop: torch.Tensor | None = None):
        mask = ids.ne(self.pad_id); tokens = self.embedding(ids) + self.position[:, :ids.shape[1]]
        tokens = self.norm(self.encoder(tokens, src_key_padding_mask=~mask))
        if drop is not None:
            tokens = torch.where(drop[:, None, None], self.null.expand_as(tokens), tokens)
            mask = torch.where(drop[:, None], torch.ones_like(mask), mask)
        return tokens, mask


class StyleEncoder(nn.Module):
    """Four global mean/std tokens; raw reference spatial tokens never leave this module."""
    def __init__(self, dimension=384, layers=2, heads=6, channels=(48, 96, 192, 256)):
        super().__init__(); stages = []; previous = 1
        for channel in channels:
            stages.append(nn.Sequential(nn.Conv2d(previous, channel, 4, 2, 1), nn.GroupNorm(8, channel), nn.SiLU(),
                                        nn.Conv2d(channel, channel, 3, padding=1), nn.GroupNorm(8, channel), nn.SiLU()))
            previous = channel
        self.stages = nn.ModuleList(stages); self.projections = nn.ModuleList([nn.Linear(c * 2, dimension) for c in channels])
        self.stage_embedding = nn.Parameter(torch.randn(1, 4, dimension) * .02)
        block = nn.TransformerEncoderLayer(dimension, heads, dimension * 4, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(dimension)
        self.null = nn.Parameter(torch.randn(1, 4, dimension) * .02)
    def forward(self, reference, drop: torch.Tensor | None = None):
        if reference.ndim != 4 or reference.shape[1] != 1:
            raise ValueError("StyleEncoder requires exactly one reference image [B,1,H,W]")
        feature = reference; tokens = []
        for stage, projection in zip(self.stages, self.projections):
            feature = stage(feature); mean = feature.mean((2, 3)); std = feature.var((2, 3), unbiased=False).add(1e-6).sqrt()
            tokens.append(projection(torch.cat([mean, std.log()], 1)))
        output = self.norm(self.transformer(torch.stack(tokens, 1) + self.stage_embedding))
        if drop is not None: output = torch.where(drop[:, None, None], self.null.expand_as(output), output)
        return output


class FusionBlock(nn.Module):
    def __init__(self, dimension=384, heads=6):
        super().__init__(); self.qnorm = nn.LayerNorm(dimension); self.snorm = nn.LayerNorm(dimension)
        self.cross = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.mnorm = nn.LayerNorm(dimension); self.mlp = nn.Sequential(nn.Linear(dimension, dimension*4), nn.GELU(), nn.Linear(dimension*4, dimension))
    def forward(self, text, style):
        query = self.qnorm(text); context = self.snorm(style)
        text = text + self.cross(query, context, context, need_weights=False)[0]
        return text + self.mlp(self.mnorm(text))


class TextStyleConditioner(nn.Module):
    def __init__(self, vocab_size, dimension=384, heads=6, text_layers=4, style_layers=2, fusion_layers=2, pad_id=0):
        super().__init__(); self.text = TextEncoder(vocab_size, dimension, text_layers, heads, pad_id=pad_id)
        self.style = StyleEncoder(dimension, style_layers, heads)
        self.fusion = nn.ModuleList([FusionBlock(dimension, heads) for _ in range(fusion_layers)])
    def forward(self, ids, reference, drop_text=None, drop_style=None):
        text, mask = self.text(ids, drop_text); style = self.style(reference, drop_style)
        for block in self.fusion: text = block(text, style)
        return text, mask, style


def supervised_contrastive_loss(style_tokens: torch.Tensor, labels: torch.Tensor, temperature=.07):
    embedding = F.normalize(style_tokens.mean(1), dim=-1); logits = embedding @ embedding.t() / temperature
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device); positive = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positive.any(1)
    if not valid.any(): return embedding.sum() * 0
    logits = logits.masked_fill(eye, -torch.inf); log_prob = logits - torch.logsumexp(logits, 1, keepdim=True)
    return -(log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1))[valid].mean()

