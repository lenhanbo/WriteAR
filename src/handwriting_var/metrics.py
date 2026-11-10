from __future__ import annotations

from pathlib import Path
import torch


def edit_distance(a: str, b: str) -> int:
    previous=list(range(len(b)+1))
    for i,left in enumerate(a,1):
        current=[i]
        for j,right in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(left!=right)))
        previous=current
    return previous[-1]


def character_error_rate(predictions: list[str], targets: list[str]) -> float:
    return sum(edit_distance(a,b) for a,b in zip(predictions,targets))/max(1,sum(len(x) for x in targets))


def retrieval_top1(query: torch.Tensor, gallery: torch.Tensor, query_labels: torch.Tensor, gallery_labels: torch.Tensor) -> float:
    query=torch.nn.functional.normalize(query,dim=-1); gallery=torch.nn.functional.normalize(gallery,dim=-1)
    nearest=(query@gallery.t()).argmax(1); return gallery_labels[nearest].eq(query_labels).float().mean().item()


def greedy_ctc_decode(logits: torch.Tensor, characters: list[str], blank: int = 0) -> list[str]:
    output=[]
    for sequence in logits.argmax(-1).tolist():
        collapsed=[]; previous=None
        for value in sequence:
            if value != previous and value != blank and 0 < value <= len(characters): collapsed.append(characters[value-1])
            previous=value
        output.append("".join(collapsed))
    return output


def select_best_checkpoint(candidates: list[dict], nll_tolerance: float = 0.01):
    """NLL shortlist, then lowest CER, then highest writer retrieval."""
    if not candidates: raise ValueError("No checkpoint candidates")
    best_nll=min(x["nll"] for x in candidates); shortlist=[x for x in candidates if x["nll"]<=best_nll+nll_tolerance]
    return min(shortlist,key=lambda x:(x["cer"],-x["writer_retrieval"]))
