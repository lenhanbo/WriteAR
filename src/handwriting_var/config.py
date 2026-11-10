from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json


@dataclass
class ModelConfig:
    image_height: int = 64
    image_width: int = 256
    latent_dim: int = 128
    codebook_size: int = 1024
    scales: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (1, 4), (2, 8), (3, 12), (4, 16))
    model_dim: int = 384
    heads: int = 6
    text_layers: int = 4
    style_layers: int = 2
    fusion_layers: int = 2
    var_layers: int = 8
    max_text_length: int = 24

    @property
    def num_tokens(self) -> int:
        return sum(h * w for h, w in self.scales)


@dataclass
class TrainConfig:
    seed: int = 42
    tokenizer_steps: int = 80_000
    var_steps: int = 120_000
    batch_size: int = 8
    accumulation: int = 4
    learning_rate: float = 2e-4
    amp: bool = True
    grad_clip: float = 1.0
    checkpoint_every: int = 1000
    log_every: int = 50
    preview_every: int = 1000
    preview_samples: int = 4
    auxiliary_every: int = 4
    text_dropout: float = 0.10
    style_dropout: float = 0.10
    model: ModelConfig = field(default_factory=ModelConfig)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

