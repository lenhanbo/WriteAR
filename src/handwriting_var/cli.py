from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml

from .cache import cache_tokens
from .checkpoint import load_checkpoint
from .config import ModelConfig, TrainConfig
from .data import IAMOneShotDataset, prepare_manifests
from .pipeline import HandwritingVARPipeline
from .auxiliary import WordOCR, WriterStyleNetwork, freeze_auxiliary
from .metrics import character_error_rate, greedy_ctc_decode, retrieval_top1
from .tokenizer import MultiScaleVQVAE
from .train import train_ocr, train_tokenizer, train_var, train_writer
from .vocab import CharacterVocabulary


def load_config(path):
    if not path: return TrainConfig()
    raw=yaml.safe_load(Path(path).read_text()) or {}; model=ModelConfig(**raw.pop("model",{})); return TrainConfig(model=model,**raw)


def add_common(parser):
    parser.add_argument("--config",default=None); parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); parser.add_argument("--steps",type=int,default=None)


def main(argv=None):
    parser=argparse.ArgumentParser(prog="handwriting-var"); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare"); p.add_argument("--iam-root",default="data/iam_dataset"); p.add_argument("--forms",help="Optional forms.txt path; defaults to <iam-root>/forms.txt"); p.add_argument("--output",default="VAR/artifacts/manifests"); p.add_argument("--seed",type=int,default=42)
    p=sub.add_parser("train-tokenizer"); add_common(p); p.add_argument("--manifest",required=True); p.add_argument("--vocab",required=True); p.add_argument("--output",required=True); p.add_argument("--resume")
    p=sub.add_parser("train-ocr"); add_common(p); p.add_argument("--manifest",required=True); p.add_argument("--vocab",required=True); p.add_argument("--output",required=True)
    p=sub.add_parser("train-writer"); add_common(p); p.add_argument("--manifest",required=True); p.add_argument("--vocab",required=True); p.add_argument("--writers",required=True); p.add_argument("--output",required=True)
    p=sub.add_parser("cache-tokens"); add_common(p); p.add_argument("--manifest",required=True); p.add_argument("--vocab",required=True); p.add_argument("--tokenizer",required=True); p.add_argument("--output",required=True); p.add_argument("--batch-size",type=int,default=32)
    p=sub.add_parser("train-var"); add_common(p); p.add_argument("--manifest",required=True); p.add_argument("--vocab",required=True); p.add_argument("--writers",required=True); p.add_argument("--tokenizer",required=True); p.add_argument("--token-cache"); p.add_argument("--ocr",required=True); p.add_argument("--writer-checkpoint",required=True); p.add_argument("--output",required=True); p.add_argument("--resume")
    p=sub.add_parser("sample"); p.add_argument("--checkpoint",required=True); p.add_argument("--text",required=True); p.add_argument("--style-image",required=True); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--text-cfg",type=float,default=4); p.add_argument("--style-cfg",type=float,default=2); p.add_argument("--temperature",type=float,default=1); p.add_argument("--top-k",type=int,default=256); p.add_argument("--top-p",type=float,default=.95); p.add_argument("--seed",type=int,default=42)
    p=sub.add_parser("evaluate"); p.add_argument("--checkpoint",required=True); p.add_argument("--manifest",required=True); p.add_argument("--vocab",required=True); p.add_argument("--output",required=True); p.add_argument("--limit",type=int,default=100); p.add_argument("--ocr"); p.add_argument("--writer-checkpoint"); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args(argv)
    if args.command=="prepare": print(json.dumps(prepare_manifests(args.iam_root,args.output,args.seed,args.forms),indent=2)); return
    if args.command=="sample":
        image=HandwritingVARPipeline.from_checkpoint(args.checkpoint,args.device).generate(args.text,args.style_image,args.text_cfg,args.style_cfg,args.temperature,args.top_k,args.top_p,args.seed); Path(args.output).parent.mkdir(parents=True,exist_ok=True); image.save(args.output); return
    if args.command=="evaluate":
        pipeline=HandwritingVARPipeline.from_checkpoint(args.checkpoint,args.device); vocab=CharacterVocabulary.load(args.vocab); dataset=IAMOneShotDataset(args.manifest,vocab,False); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
        ocr=None; writer=None; characters=[]
        if args.ocr:
            state=torch.load(args.ocr,map_location=args.device,weights_only=False); characters=state.get("extra",{}).get("characters",[]); ocr=freeze_auxiliary(WordOCR(len(characters)+1).to(args.device)); ocr.load_state_dict(state["models"]["ocr"])
        if args.writer_checkpoint:
            state=torch.load(args.writer_checkpoint,map_location=args.device,weights_only=False); writers=state["models"]["writer"]["head.weight"].shape[0]; writer=freeze_auxiliary(WriterStyleNetwork(writers).to(args.device)); writer.load_state_dict(state["models"]["writer"])
        rows=[]; predictions=[]; targets=[]; generated_embeddings=[]; reference_embeddings=[]; labels=[]; writer_ids={}
        for i in range(min(args.limit,len(dataset))):
            item=dataset[i]; generated=pipeline.generate(item["target_text"],item["reference_image"],seed=i); generated.save(output/f"{item['sample_id']}.png")
            tensor=torch.from_numpy(__import__("numpy").array(generated)).float().div(127.5).sub(1)[None,None].to(args.device)
            prediction=None
            if ocr is not None: prediction=greedy_ctc_decode(ocr(tensor),characters)[0]; predictions.append(prediction); targets.append(item["target_text"])
            if writer is not None:
                generated_embeddings.append(writer(tensor)[0].cpu()); reference_embeddings.append(writer(item["reference_image"][None].to(args.device))[0].cpu()); labels.append(writer_ids.setdefault(item["writer_id"],len(writer_ids)))
            rows.append({"sample_id":item["sample_id"],"writer_id":item["writer_id"],"text":item["target_text"],"ocr":prediction})
        summary={"samples":len(rows)}
        if predictions: summary["cer"]=character_error_rate(predictions,targets)
        if generated_embeddings:
            label_tensor=torch.tensor(labels); summary["writer_retrieval_top1"]=retrieval_top1(torch.cat(generated_embeddings),torch.cat(reference_embeddings),label_tensor,label_tensor)
        (output/"predictions.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8"); (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); return
    config=load_config(args.config); steps=args.steps
    if args.command=="train-tokenizer": train_tokenizer(args.manifest,args.vocab,args.output,config,args.device,steps,args.resume)
    elif args.command=="train-ocr": train_ocr(args.manifest,args.vocab,args.output,config,args.device,steps or 20_000)
    elif args.command=="train-writer": train_writer(args.manifest,args.vocab,args.writers,args.output,config,args.device,steps or 20_000)
    elif args.command=="cache-tokens":
        vocab=CharacterVocabulary.load(args.vocab); dataset=IAMOneShotDataset(args.manifest,vocab,False); model=MultiScaleVQVAE(config.model.latent_dim,config.model.codebook_size,config.model.scales); load_checkpoint(args.tokenizer,{"tokenizer":model},restore_random=False); print(cache_tokens(model,dataset,args.output,args.device,args.batch_size))
    elif args.command=="train-var": train_var(args.manifest,args.vocab,args.writers,args.tokenizer,args.ocr,args.writer_checkpoint,args.output,config,args.device,steps,args.resume,token_cache=args.token_cache)


if __name__=="__main__": main()

