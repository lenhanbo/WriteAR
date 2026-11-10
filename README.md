# Handwriting-VAR

Independent two-stage one-shot handwriting synthesis for IAM. It generates a
single grayscale word image at `64x256` from text and exactly one untranscribed
reference image. Conditioning is implemented only with Transformer
cross-attention (no AdaIN, FiLM, or AdaLN).

## Install and prepare

```bash
pip install -e .
handwriting-var prepare --iam-root data/iam_dataset \
  --output artifacts/manifests
```

`forms.txt` is discovered at `data/iam_dataset/forms.txt` (or under `ascii/`).
Preparation uses its official IAM form-to-writer mapping, normalizes labels,
and creates a deterministic 80/10/10 writer-disjoint split plus vocabulary and
training-writer mapping. VATr metadata is not required.

## Training order

```bash
handwriting-var train-tokenizer --config configs/kaggle.yaml --manifest artifacts/manifests/train.json --vocab artifacts/manifests/vocab.json --output checkpoints/tokenizer.pt
handwriting-var train-ocr --config configs/kaggle.yaml --manifest artifacts/manifests/train.json --vocab artifacts/manifests/vocab.json --output checkpoints/ocr.pt
handwriting-var train-writer --config configs/kaggle.yaml --manifest artifacts/manifests/train.json --vocab artifacts/manifests/vocab.json --writers artifacts/manifests/writers.json --output checkpoints/writer.pt
handwriting-var cache-tokens --config configs/kaggle.yaml --manifest artifacts/manifests/train.json --vocab artifacts/manifests/vocab.json --tokenizer checkpoints/tokenizer.pt --output artifacts/tokens
handwriting-var train-var --config configs/kaggle.yaml --manifest artifacts/manifests/train.json --vocab artifacts/manifests/vocab.json --writers artifacts/manifests/writers.json --tokenizer checkpoints/tokenizer.pt --token-cache artifacts/tokens --ocr checkpoints/ocr.pt --writer-checkpoint checkpoints/writer.pt --output checkpoints/var_best.pt
```

Checkpoints include optimizer, scheduler, AMP scaler, RNG state, config hash,
and manifest hash for resumable Kaggle sessions.

Training prints step, losses, learning rate, throughput, scale accuracies, and
peak CUDA memory. Preview grids are written next to the checkpoint under
`previews/<stage>/step_XXXXXXX.png`. The default Kaggle config logs every 50
optimizer steps and previews every 1,000; `configs/smoke.yaml` emits both at
every step.

## Inference

```python
from handwriting_var import HandwritingVARPipeline
pipeline = HandwritingVARPipeline.from_checkpoint("checkpoints/var_best.pt", device="cuda")
image = pipeline.generate("handwriting", "reference.png", seed=42)
```

```bash
handwriting-var sample --checkpoint checkpoints/var_best.pt --text handwriting \
  --style-image reference.png --output generated.png
```
"# WriteAR" 
