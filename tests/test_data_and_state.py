import json
import random
import numpy as np
import torch

from handwriting_var.checkpoint import load_checkpoint, save_checkpoint
from handwriting_var.data import load_forms_writer_map, normalize_label, writer_disjoint_split
from handwriting_var.metrics import character_error_rate, select_best_checkpoint


def test_normalize_and_writer_disjoint_split():
    assert normalize_label("  A&amp;B  ")=="A&B"
    records=[{"writer_id":str(w),"sample_id":f"{w}-{i}"} for w in range(657) for i in range(2)]
    split=writer_disjoint_split(records,42); sets=[{x["writer_id"] for x in rows} for rows in split.values()]
    assert [len(x) for x in sets]==[526,66,65]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])


def test_official_forms_mapping(tmp_path):
    forms=tmp_path/"forms.txt"
    forms.write_text("# comment\na01-000u 000 2 prt 7 5 52 36\na01-000x 001 2 all 6 6 52 52\n",encoding="utf-8")
    assert load_forms_writer_map(forms)=={"a01-000u":"000","a01-000x":"001"}


def test_checkpoint_restores_rng_optimizer_and_scheduler(tmp_path):
    model=torch.nn.Linear(2,2); optimizer=torch.optim.AdamW(model.parameters()); scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    path=tmp_path/"state.pt"; save_checkpoint(path,{"model":model},optimizer,scheduler,step=3)
    expected=(random.random(),np.random.rand(),torch.rand(1))
    load_checkpoint(path,{"model":model},optimizer,scheduler)
    actual=(random.random(),np.random.rand(),torch.rand(1))
    assert actual[0]==expected[0] and actual[1]==expected[1] and torch.equal(actual[2],expected[2])


def test_metrics_and_checkpoint_selection():
    assert character_error_rate(["cat"],["cut"])==1/3
    rows=[{"path":"a","nll":1.0,"cer":.2,"writer_retrieval":.9},{"path":"b","nll":1.005,"cer":.1,"writer_retrieval":.7}]
    assert select_best_checkpoint(rows)["path"]=="b"
