from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def cache_tokens(tokenizer,dataset,output,device="cuda",batch_size=32,num_workers=0):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); tokenizer=tokenizer.to(device).eval(); index={}
    loader=DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,collate_fn=lambda batch: batch)
    for batch in loader:
        images=torch.stack([x["target_image"] for x in batch]).to(device); _,tokens,_,_,_=tokenizer.encode(images)
        for row,per_sample in zip(batch,zip(*[x.cpu() for x in tokens])):
            path=output/f'{row["sample_id"]}.pt'; torch.save(tuple(per_sample),path); index[row["sample_id"]]=path.name
    (output/"index.json").write_text(json.dumps(index),encoding="utf-8")
    return len(index)

