from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path
import random
import unicodedata

import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageEnhance, ImageFilter

from .vocab import CharacterVocabulary


def normalize_label(value: str) -> str:
    return unicodedata.normalize("NFC", html.unescape(value)).strip()


def load_forms_writer_map(path: str | Path) -> dict[str, str]:
    """Load the official IAM ``form_id -> writer_id`` mapping."""
    mapping: dict[str, str] = {}
    source = Path(path)
    for line_number, raw in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed forms.txt row {line_number}: {raw!r}")
        form_id, writer = fields[:2]
        old = mapping.setdefault(form_id, writer)
        if old != writer:
            raise ValueError(f"Conflicting writer for IAM form {form_id}: {old}/{writer}")
    if not mapping:
        raise ValueError(f"No form-to-writer mappings found in {source}")
    return mapping


def find_forms_file(iam_root: str | Path, forms_path: str | Path | None = None) -> Path:
    if forms_path is not None:
        candidate = Path(forms_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"IAM forms.txt not found: {candidate}")
        return candidate
    root = Path(iam_root)
    for candidate in (root / "forms.txt", root / "ascii" / "forms.txt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Expected forms.txt under {root} or {root / 'ascii'}")


def parse_iam_tab_gt(path: str | Path, root: str | Path, form_writers: dict[str, str], max_length: int = 24) -> list[dict]:
    records = []
    root = Path(root)
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in raw:
            continue
        relative, value = raw.split("\t", 1)
        text = normalize_label(value)
        word_id = Path(relative).stem
        form_id = "-".join(word_id.split("-")[:2])
        writer = form_writers.get(form_id)
        image = root / relative
        if writer is not None and 0 < len(text) <= max_length and image.is_file():
            records.append({"sample_id": word_id, "path": str(image.resolve()), "text": text,
                            "writer_id": writer, "form_id": form_id})
    return records


def writer_disjoint_split(records: list[dict], seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> dict[str, list[dict]]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8 or any(value <= 0 for value in ratios):
        raise ValueError("Split ratios must contain three positive values summing to one")
    by_writer: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_writer[record["writer_id"]].append(record)
    by_writer = {writer: rows for writer, rows in by_writer.items() if len(rows) >= 2}
    writers = sorted(by_writer)
    if len(writers) < 3:
        raise ValueError(f"Need at least three eligible writers, got {len(writers)}")
    random.Random(seed).shuffle(writers)
    n_train = round(len(writers) * ratios[0])
    n_validation = round(len(writers) * ratios[1])
    groups = {"train": writers[:n_train], "validation": writers[n_train:n_train+n_validation],
              "test": writers[n_train+n_validation:]}
    return {name: [r for writer in ids for r in by_writer[writer]] for name, ids in groups.items()}


def prepare_manifests(iam_root: str | Path, output: str | Path, seed: int = 42,
                      forms_path: str | Path | None = None) -> dict:
    iam_root, output = Path(iam_root), Path(output)
    forms_file = find_forms_file(iam_root, forms_path)
    records = parse_iam_tab_gt(iam_root / "linux_gt.txt", iam_root, load_forms_writer_map(forms_file))
    splits = writer_disjoint_split(records, seed)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (output / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    vocab = CharacterVocabulary(sorted({ch for r in splits["train"] for ch in r["text"]}))
    vocab.save(output / "vocab.json")
    writers = {w: i for i, w in enumerate(sorted({r["writer_id"] for r in splits["train"]}))}
    (output / "writers.json").write_text(json.dumps(writers, indent=2), encoding="utf-8")
    summary = {k: {"samples": len(v), "writers": len({x["writer_id"] for x in v})} for k, v in splits.items()}
    summary["metadata"] = {"writer_source": str(forms_file.resolve()), "seed": seed,
                           "eligible_writers": sum(summary[name]["writers"] for name in ("train", "validation", "test"))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def canonical_image(image: Image.Image, canvas=(64, 256), region=(48, 240)) -> torch.Tensor:
    image = image.convert("L")
    scale = min(region[1] / image.width, region[0] / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)
    canvas_image = Image.new("L", (canvas[1], canvas[0]), 255)
    canvas_image.paste(image, ((canvas[1] - size[0]) // 2, (canvas[0] - size[1]) // 2))
    return TF.pil_to_tensor(canvas_image).float().div(127.5).sub(1)


def augment(image: torch.Tensor) -> torch.Tensor:
    angle, translate, scale, shear = random.uniform(-2, 2), [random.randint(-3, 3), random.randint(-2, 2)], random.uniform(.95, 1.05), random.uniform(-2, 2)
    out = TF.affine(image, angle, translate, scale, [shear, 0], InterpolationMode.BILINEAR, fill=1.0)
    pil = TF.to_pil_image(out.add(1).div(2).clamp(0, 1))
    pil = ImageEnhance.Contrast(pil).enhance(random.uniform(.85, 1.15))
    if random.random() < .25:
        pil = pil.filter(random.choice([ImageFilter.MinFilter(3), ImageFilter.MaxFilter(3)]))
    return TF.pil_to_tensor(pil).float().div(127.5).sub(1)


class IAMOneShotDataset(Dataset):
    def __init__(self, manifest: str | Path, vocabulary: CharacterVocabulary, training: bool = True):
        self.records = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.vocabulary, self.training = vocabulary, training
        self.by_writer: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(self.records): self.by_writer[row["writer_id"]].append(i)
        self.records = [r for r in self.records if len(self.by_writer[r["writer_id"]]) > 1]
        self.by_writer.clear()
        for i, row in enumerate(self.records): self.by_writer[row["writer_id"]].append(i)

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        choices = [i for i in self.by_writer[row["writer_id"]] if i != index]
        different = [i for i in choices if self.records[i]["text"] != row["text"]]
        ref = self.records[random.choice(different or choices)]
        with Image.open(row["path"]) as im: target = canonical_image(im)
        with Image.open(ref["path"]) as im: reference = canonical_image(im)
        if self.training:
            target, reference = augment(target), augment(reference)
        return {"target_image": target, "target_text": row["text"], "reference_image": reference,
                "writer_id": row["writer_id"], "sample_id": row["sample_id"], "reference_id": ref["sample_id"],
                "text_ids": torch.tensor(self.vocabulary.encode(row["text"]), dtype=torch.long)}


class WriterBalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: IAMOneShotDataset, writers_per_batch=8, samples_per_writer=4, seed=42):
        self.groups = list(dataset.by_writer.values()); self.wpb = writers_per_batch; self.spw = samples_per_writer
        self.seed = seed; self.epoch = 0
    def __len__(self): return max(1, len(self.groups) // self.wpb)
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch); self.epoch += 1
        groups = self.groups.copy(); rng.shuffle(groups)
        for offset in range(0, len(groups) - self.wpb + 1, self.wpb):
            yield [rng.choice(group) for group in groups[offset:offset+self.wpb] for _ in range(self.spw)]


def collate_batch(batch: list[dict], pad_id=0):
    length = max(x["text_ids"].numel() for x in batch)
    ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
    for i, x in enumerate(batch): ids[i, :x["text_ids"].numel()] = x["text_ids"]
    return {"target_image": torch.stack([x["target_image"] for x in batch]),
            "reference_image": torch.stack([x["reference_image"] for x in batch]), "text_ids": ids,
            "text_mask": ids.ne(pad_id), "target_text": [x["target_text"] for x in batch],
            "writer_id": [x["writer_id"] for x in batch], "sample_id": [x["sample_id"] for x in batch],
            "reference_id": [x["reference_id"] for x in batch]}
