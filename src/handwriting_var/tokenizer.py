from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


SCALES = ((1, 1), (1, 2), (1, 4), (2, 8), (3, 12), (4, 16))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))
    def forward(self, x): return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim=128, base=64):
        super().__init__()
        channels = [1, base, base * 2, base * 2, latent_dim]
        layers = []
        for a, b in zip(channels, channels[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.GroupNorm(8, b), nn.SiLU(), ResidualBlock(b)]
        self.net = nn.Sequential(*layers)
    def forward(self, image): return self.net(image)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, base=64):
        super().__init__()
        channels = [latent_dim, base * 2, base * 2, base, 1]
        layers = []
        for i, (a, b) in enumerate(zip(channels, channels[1:])):
            layers.append(ResidualBlock(a))
            layers.append(nn.ConvTranspose2d(a, b, 4, 2, 1))
            if i < 3: layers += [nn.GroupNorm(8, b), nn.SiLU()]
        layers.append(nn.Tanh()); self.net = nn.Sequential(*layers)
    def forward(self, latent): return self.net(latent)


class MultiScaleResidualQuantizer(nn.Module):
    def __init__(self, codebook_size=1024, dimension=128, scales=SCALES):
        super().__init__()
        self.scales = tuple(tuple(x) for x in scales)
        self.codebook = nn.Embedding(codebook_size, dimension)
        nn.init.uniform_(self.codebook.weight, -1 / codebook_size, 1 / codebook_size)
        self.residual_convs = nn.ModuleList([nn.Conv2d(dimension, dimension, 3, padding=1) for _ in self.scales])

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return self.codebook(ids).permute(0, 3, 1, 2).contiguous()

    def merge_embeddings(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        merged = 0
        for emb, conv in zip(embeddings, self.residual_convs):
            merged = merged + conv(F.interpolate(emb, (4, 16), mode="bilinear", align_corners=False))
        return merged

    def decode_indices(self, indices: list[torch.Tensor]) -> torch.Tensor:
        return self.merge_embeddings([self.lookup(ids) for ids in indices])

    def forward(self, latent: torch.Tensor):
        residual = latent; quantized_sum = torch.zeros_like(latent); indices = []; codebook_loss = latent.new_zeros(())
        commitment = latent.new_zeros(()); counts = torch.zeros(self.codebook.num_embeddings, device=latent.device)
        for size, conv in zip(self.scales, self.residual_convs):
            local = F.adaptive_avg_pool2d(residual, size)
            flat = local.permute(0, 2, 3, 1).reshape(-1, local.shape[1])
            distance = flat.square().sum(1, keepdim=True) + self.codebook.weight.square().sum(1) - 2 * flat @ self.codebook.weight.t()
            ids = distance.argmin(1).reshape(latent.shape[0], *size); indices.append(ids)
            counts.scatter_add_(0, ids.flatten(), torch.ones_like(ids.flatten(), dtype=counts.dtype))
            embedding = self.lookup(ids)
            codebook_loss = codebook_loss + F.mse_loss(embedding, local.detach())
            commitment = commitment + F.mse_loss(local, embedding.detach())
            st = local + (embedding - local).detach()
            contribution = conv(F.interpolate(st, latent.shape[-2:], mode="bilinear", align_corners=False))
            quantized_sum = quantized_sum + contribution
            residual = residual - contribution
        probability = counts / counts.sum().clamp_min(1)
        perplexity = torch.exp(-(probability * probability.clamp_min(1e-10).log()).sum())
        metrics = {"perplexity": perplexity, "active_code_ratio": counts.gt(0).float().mean(), "dead_codes": counts.eq(0).sum()}
        n = len(self.scales)
        return quantized_sum, indices, codebook_loss / n, commitment / n, metrics


class MultiScaleVQVAE(nn.Module):
    def __init__(self, latent_dim=128, codebook_size=1024, scales=SCALES):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.quantizer = MultiScaleResidualQuantizer(codebook_size, latent_dim, scales)
        self.decoder = Decoder(latent_dim)
    def encode(self, image): return self.quantizer(self.encoder(image))
    def decode_tokens(self, tokens): return self.decoder(self.quantizer.decode_indices(tokens))
    def forward(self, image):
        quantized, indices, codebook, commit, metrics = self.encode(image)
        return self.decoder(quantized), indices, codebook, commit, metrics
    def freeze(self):
        self.eval()
        for p in self.parameters(): p.requires_grad_(False)
        return self


class PatchDiscriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        def block(a, b, norm=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(a, b, 4, 2, 1))]
            if norm: layers.append(nn.GroupNorm(8, b))
            return layers + [nn.LeakyReLU(.2)]
        self.net = nn.Sequential(*block(1, base, False), *block(base, base*2), *block(base*2, base*4),
                                 nn.utils.spectral_norm(nn.Conv2d(base*4, 1, 3, padding=1)))
    def forward(self, image): return self.net(image)


def sobel(image: torch.Tensor) -> torch.Tensor:
    kernels = image.new_tensor([[[-1,0,1],[-2,0,2],[-1,0,1]], [[-1,-2,-1],[0,0,0],[1,2,1]]]).unsqueeze(1)
    edge = F.conv2d(image, kernels, padding=1)
    return edge.square().sum(1, keepdim=True).add(1e-8).sqrt()


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu_x, mu_y = F.avg_pool2d(x, 3, 1, 1), F.avg_pool2d(y, 3, 1, 1)
    var_x = F.avg_pool2d(x*x, 3, 1, 1) - mu_x.square(); var_y = F.avg_pool2d(y*y, 3, 1, 1) - mu_y.square()
    cov = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
    return (((2*mu_x*mu_y + .01**2)*(2*cov + .03**2)) / ((mu_x.square()+mu_y.square()+.01**2)*(var_x+var_y+.03**2))).mean()


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = ((1 - target) / 2).clamp(0, 1); weight = 1 + 4 * mask
    pixel = (weight * (target - prediction).abs()).sum() / weight.sum().clamp_min(1)
    return pixel + .2 * F.l1_loss(sobel(prediction), sobel(target)) + .5 * (1 - ssim(prediction, target))


def adversarial_weight(step: int) -> float:
    return max(0.0, min(0.1, (step - 10_000) / 10_000 * .1))
