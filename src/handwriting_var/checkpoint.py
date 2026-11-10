from __future__ import annotations

import hashlib
import random
from pathlib import Path
import numpy as np
import torch


def file_hash(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1<<20),b""): digest.update(chunk)
    return digest.hexdigest()


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path,models,optimizer=None,scheduler=None,scaler=None,step=0,config_hash=None,manifest_hash=None,extra=None):
    payload={"models":{k:v.state_dict() for k,v in models.items()},"step":step,"config_hash":config_hash,
             "manifest_hash":manifest_hash,"rng":rng_state(),"extra":extra or {}}
    if optimizer is not None: payload["optimizer"]=optimizer.state_dict()
    if scheduler is not None: payload["scheduler"]=scheduler.state_dict()
    if scaler is not None: payload["scaler"]=scaler.state_dict()
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)


def load_checkpoint(path,models,optimizer=None,scheduler=None,scaler=None,map_location="cpu",strict=True,restore_random=True):
    payload=torch.load(path,map_location=map_location,weights_only=False)
    for key,model in models.items(): model.load_state_dict(payload["models"][key],strict=strict)
    if optimizer is not None and "optimizer" in payload: optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload: scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload: scaler.load_state_dict(payload["scaler"])
    if restore_random and "rng" in payload: restore_rng(payload["rng"])
    return payload

