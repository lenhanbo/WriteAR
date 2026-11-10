# Handwriting-VAR

Independent two-stage one-shot handwriting synthesis for IAM. It generates a
single grayscale word image at `64x256` from text and exactly one untranscribed
reference image. Conditioning is implemented only with Transformer
cross-attention (no AdaIN, FiLM, or AdaLN).

## Install and prepare

```bash
pip install -e VAR
handwriting-var prepare --iam-root data/iam_dataset --vatr-root VATr \
  --output VAR/artifacts/manifests
```

Preparation recovers the trusted 500 VATr writer identities, normalizes labels,
and creates a deterministic 400/50/50 writer-disjoint split plus vocabulary and
training-writer mapping.

## Training order

```bash
handwriting-var train-tokenizer --config VAR/configs/kaggle.yaml --manifest VAR/artifacts/manifests/train.json --vocab VAR/artifacts/manifests/vocab.json --output checkpoints/tokenizer.pt
handwriting-var train-ocr --config VAR/configs/kaggle.yaml --manifest VAR/artifacts/manifests/train.json --vocab VAR/artifacts/manifests/vocab.json --output checkpoints/ocr.pt
handwriting-var train-writer --config VAR/configs/kaggle.yaml --manifest VAR/artifacts/manifests/train.json --vocab VAR/artifacts/manifests/vocab.json --writers VAR/artifacts/manifests/writers.json --output checkpoints/writer.pt
handwriting-var cache-tokens --config VAR/configs/kaggle.yaml --manifest VAR/artifacts/manifests/train.json --vocab VAR/artifacts/manifests/vocab.json --tokenizer checkpoints/tokenizer.pt --output VAR/artifacts/tokens
handwriting-var train-var --config VAR/configs/kaggle.yaml --manifest VAR/artifacts/manifests/train.json --vocab VAR/artifacts/manifests/vocab.json --writers VAR/artifacts/manifests/writers.json --tokenizer checkpoints/tokenizer.pt --token-cache VAR/artifacts/tokens --ocr checkpoints/ocr.pt --writer-checkpoint checkpoints/writer.pt --output checkpoints/var_best.pt
```

Checkpoints include optimizer, scheduler, AMP scaler, RNG state, config hash,
and manifest hash for resumable Kaggle sessions.

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
