from __future__ import annotations

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
    def forward(self, image):
        return self.net(image)


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


class ResidualQuantConv(nn.Module):
    """VAR's phi(x) = (1-r)x + r Conv3x3(x)."""
    def __init__(self, dimension: int, ratio=.5):
        super().__init__(); self.ratio=ratio; self.conv=nn.Conv2d(dimension,dimension,3,padding=1)
    def forward(self, value):
        return value.mul(1-self.ratio)+self.conv(value).mul(self.ratio)


class MultiScaleResidualQuantizer(nn.Module):
    def __init__(self, codebook_size=1024, dimension=128, scales=SCALES):
        super().__init__()
        self.scales = tuple(tuple(x) for x in scales)
        self.codebook = nn.Embedding(codebook_size, dimension)
        self.residual_convs = nn.ModuleList([ResidualQuantConv(dimension,.5) for _ in self.scales])
        self.register_buffer("initialized",torch.tensor(False))
        self.register_buffer("usage_ema",torch.zeros(codebook_size))
        self.register_buffer("update_count",torch.zeros((),dtype=torch.long))

    @torch.no_grad()
    def initialize_codebook(self, latent: torch.Tensor) -> None:
        """Data-dependent initialization gives every code a viable starting point."""
        samples=torch.cat([F.adaptive_avg_pool2d(latent,size).permute(0,2,3,1).reshape(-1,latent.shape[1])
                           for size in self.scales],0).float()
        count=self.codebook.num_embeddings
        if samples.shape[0]>=count:
            selected=samples[torch.randperm(samples.shape[0],device=samples.device)[:count]]
        else:
            selected=samples[torch.randint(samples.shape[0],(count,),device=samples.device)]
        scale=samples.std(0,unbiased=False).mean().clamp_min(1e-3)
        selected=selected+torch.randn_like(selected)*(scale*.01)
        self.codebook.weight.copy_(selected.to(self.codebook.weight.dtype))
        self.usage_ema.fill_(1/self.codebook.num_embeddings)
        self.initialized.fill_(True)

    @torch.no_grad()
    def update_usage_and_revive(self, counts: torch.Tensor, samples: torch.Tensor) -> None:
        probability=counts/counts.sum().clamp_min(1)
        self.usage_ema.mul_(.99).add_(probability.to(self.usage_ema.dtype),alpha=.01)
        self.update_count.add_(1)
        if not self.training or self.update_count.item()%100:
            return
        threshold=1/(self.codebook.num_embeddings*100)
        dead=torch.nonzero(self.usage_ema<threshold).flatten()
        if dead.numel()==0:
            return
        # Revive at most 5% at once to avoid a disruptive full reset.
        dead=dead[:max(1,self.codebook.num_embeddings//20)]
        replacements=samples[torch.randint(samples.shape[0],(dead.numel(),),device=samples.device)].float()
        noise=replacements.std(0,unbiased=False).mean().clamp_min(1e-3)*.01
        self.codebook.weight[dead].copy_((replacements+torch.randn_like(replacements)*noise).to(self.codebook.weight.dtype))
        self.usage_ema[dead]=1/self.codebook.num_embeddings

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
        # The official VAR quantizer performs assignment and residual updates
        # on a detached fp32 copy, while the cumulative f_hat objective trains
        # the codebook/projections and straight-through trains the encoder.
        feature=latent.float(); feature_detached=feature.detach()
        if not bool(self.initialized): self.initialize_codebook(feature_detached)
        residual=feature_detached.clone(); quantized_sum=torch.zeros_like(feature); indices=[]
        codebook_loss=feature.new_zeros(()); commitment=feature.new_zeros(())
        counts=torch.zeros(self.codebook.num_embeddings,device=latent.device); assignment_samples=[]
        for size, conv in zip(self.scales, self.residual_convs):
            local = F.adaptive_avg_pool2d(residual, size)
            flat = local.permute(0, 2, 3, 1).reshape(-1, local.shape[1])
            assignment_samples.append(flat.detach())
            distance = flat.square().sum(1, keepdim=True) + self.codebook.weight.square().sum(1) - 2 * flat @ self.codebook.weight.t()
            ids = distance.argmin(1).reshape(latent.shape[0], *size); indices.append(ids)
            counts.scatter_add_(0, ids.flatten(), torch.ones_like(ids.flatten(), dtype=counts.dtype))
            embedding = self.lookup(ids)
            contribution = conv(F.interpolate(embedding,feature.shape[-2:],mode="bicubic",align_corners=False))
            quantized_sum = quantized_sum + contribution
            codebook_loss = codebook_loss + F.mse_loss(quantized_sum,feature_detached)
            commitment = commitment + F.mse_loss(feature,quantized_sum.detach())
            residual = residual - contribution.detach()
        self.update_usage_and_revive(counts,torch.cat(assignment_samples,0))
        probability = counts / counts.sum().clamp_min(1)
        perplexity = torch.exp(-(probability * probability.clamp_min(1e-10).log()).sum())
        batch_active=counts.gt(0).float().mean()
        usage_active=self.usage_ema.gt(1/(self.codebook.num_embeddings*100)).float().mean()
        metrics = {"perplexity": perplexity, "active_code_ratio": batch_active,
                   "dead_codes": counts.eq(0).sum(), "usage_active_code_ratio": usage_active}
        n = len(self.scales)
        quantized_st=feature+(quantized_sum-feature).detach()
        return quantized_st, indices, codebook_loss / n, commitment / n, metrics


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
