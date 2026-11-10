from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .auxiliary import WordOCR, WriterStyleNetwork, ctc_loss, freeze_auxiliary
from .checkpoint import file_hash, load_checkpoint, save_checkpoint
from .conditioning import TextStyleConditioner, supervised_contrastive_loss
from .config import TrainConfig
from .data import IAMOneShotDataset, WriterBalancedBatchSampler, collate_batch
from .metrics import greedy_ctc_decode
from .tokenizer import MultiScaleVQVAE, PatchDiscriminator, adversarial_weight, reconstruction_loss
from .var import ConditionalVAR, autoregressive_loss, categorical_straight_through
from .vocab import CharacterVocabulary, SPECIALS


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def infinite(loader):
    while True: yield from loader


def amp_context(device, enabled):
    return torch.autocast(device_type=device.type,dtype=torch.float16) if enabled and device.type=="cuda" else nullcontext()


def make_scaler(enabled):
    return torch.amp.GradScaler("cuda",enabled=enabled and torch.cuda.is_available())


def should_emit(step: int, every: int, final: int) -> bool:
    return step == 1 or step == final or (every > 0 and step % every == 0)


def preview_path(output, stage: str, step: int) -> Path:
    path = Path(output).parent / "previews" / stage / f"step_{step:07d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_preview(output, stage: str, step: int, groups: list[torch.Tensor], samples=4) -> Path:
    count = min(samples, *(value.shape[0] for value in groups))
    # Each row represents one sample; columns follow the order in ``groups``.
    grid = torch.stack([value for index in range(count) for group in groups for value in [group[index].detach().float().cpu()]])
    path = preview_path(output, stage, step)
    save_image(grid.add(1).div(2).clamp(0, 1), path, nrow=len(groups), padding=2, pad_value=1.0)
    return path


def log_progress(stage, step, final, start_step, started_at, optimizer, device, **metrics):
    elapsed = max(time.monotonic() - started_at, 1e-6)
    rate = max(0, step - start_step) / elapsed
    parts = [f"[{stage}] step={step}/{final}", f"rate={rate:.2f} step/s",
             f"lr={optimizer.param_groups[0]['lr']:.3e}"]
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor): value = value.detach().float().item()
        parts.append(f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}")
    if device.type == "cuda": parts.append(f"vram={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB")
    print(" ".join(parts), flush=True)


def train_tokenizer(manifest,vocab_path,output,config: TrainConfig,device="cuda",steps=None,resume=None,num_workers=0):
    device=torch.device(device); seed_everything(config.seed); vocab=CharacterVocabulary.load(vocab_path)
    dataset=IAMOneShotDataset(manifest,vocab,True); loader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=num_workers,collate_fn=lambda x:collate_batch(x,vocab.pad_id),drop_last=True)
    model=MultiScaleVQVAE(config.model.latent_dim,config.model.codebook_size,config.model.scales).to(device); discriminator=PatchDiscriminator().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate); d_optimizer=torch.optim.AdamW(discriminator.parameters(),lr=config.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config.tokenizer_steps); scaler=make_scaler(config.amp); start=0
    if resume: start=load_checkpoint(resume,{"tokenizer":model,"discriminator":discriminator},optimizer,scheduler,scaler,map_location=device)["step"]
    iterator=infinite(loader); final=steps or config.tokenizer_steps; optimizer.zero_grad(set_to_none=True); started_at=time.monotonic()
    print(f"[tokenizer] samples={len(dataset)} start={start} target={final} device={device} "
          f"batch={config.batch_size} accumulation={config.accumulation}", flush=True)
    for micro_step in range(start*config.accumulation,final*config.accumulation):
        step=micro_step//config.accumulation
        batch=next(iterator); image=batch["target_image"].to(device); weight=adversarial_weight(step)
        with amp_context(device,config.amp):
            reconstruction,_,codebook,commit,metrics=model(image); rec=reconstruction_loss(reconstruction,image)
            generator_adv=-discriminator(reconstruction).mean() if weight else rec.new_zeros(()); loss=rec+codebook+.25*commit+weight*generator_adv
        scaler.scale(loss/config.accumulation).backward()
        if (micro_step+1)%config.accumulation==0:
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        dloss=loss.new_zeros(())
        if step>=10_000:
            d_optimizer.zero_grad(set_to_none=True)
            with amp_context(device,config.amp): dloss=F.relu(1-discriminator(image)).mean()+F.relu(1+discriminator(reconstruction.detach())).mean()
            dloss.backward(); d_optimizer.step()
        update=step+1
        boundary=(micro_step+1)%config.accumulation==0
        if boundary and should_emit(update,config.log_every,final):
            log_progress("tokenizer",update,final,start,started_at,optimizer,device,
                         loss=float(loss.detach()),rec=float(rec.detach()),codebook=float(codebook.detach()),
                         commit=float(commit.detach()),adv=float(generator_adv.detach()),d_loss=float(dloss.detach()),
                         perplexity=float(metrics["perplexity"].detach()),active=float(metrics["active_code_ratio"].detach()),
                         dead=int(metrics["dead_codes"].detach()))
        if boundary and should_emit(update,config.preview_every,final):
            path=save_preview(output,"tokenizer",update,[image,reconstruction],config.preview_samples)
            print(f"[tokenizer] preview={path}",flush=True)
        if boundary and (update%config.checkpoint_every==0 or update==final):
            save_checkpoint(output,{"tokenizer":model,"discriminator":discriminator},optimizer,scheduler,scaler,update,config.digest(),file_hash(manifest),{"metrics":{k:float(v) for k,v in metrics.items()}})
            print(f"[tokenizer] checkpoint={output} step={update}",flush=True)
    return model


def train_ocr(manifest,vocab_path,output,config: TrainConfig,device="cuda",steps=20_000,num_workers=0):
    device=torch.device(device); vocab=CharacterVocabulary.load(vocab_path); dataset=IAMOneShotDataset(manifest,vocab,True)
    loader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=num_workers,collate_fn=lambda x:collate_batch(x,vocab.pad_id),drop_last=True); iterator=infinite(loader)
    chars=vocab.symbols[len(SPECIALS):]; mapping={ch:i+1 for i,ch in enumerate(chars)}; model=WordOCR(len(chars)+1).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate); started_at=time.monotonic()
    print(f"[ocr] samples={len(dataset)} target={steps} device={device} batch={config.batch_size}",flush=True)
    for step in range(steps):
        batch=next(iterator); image=batch["target_image"].to(device); optimizer.zero_grad(set_to_none=True); logits=model(image); loss=ctc_loss(logits,batch["target_text"],mapping); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); update=step+1
        if should_emit(update,config.log_every,steps):
            log_progress("ocr",update,steps,0,started_at,optimizer,device,ctc=float(loss.detach()))
        if should_emit(update,config.preview_every,steps):
            path=save_preview(output,"ocr",update,[image],config.preview_samples); predictions=greedy_ctc_decode(logits.detach(),chars)
            print(f"[ocr] preview={path} gt={batch['target_text'][0]!r} pred={predictions[0]!r}",flush=True)
    save_checkpoint(output,{"ocr":model},optimizer,step=steps,config_hash=config.digest(),manifest_hash=file_hash(manifest),extra={"characters":chars}); print(f"[ocr] checkpoint={output} step={steps}",flush=True); return model


def train_writer(manifest,vocab_path,writers_path,output,config: TrainConfig,device="cuda",steps=20_000,num_workers=0):
    device=torch.device(device); vocab=CharacterVocabulary.load(vocab_path); writers=json.loads(Path(writers_path).read_text()); dataset=IAMOneShotDataset(manifest,vocab,True)
    sampler=WriterBalancedBatchSampler(dataset); loader=DataLoader(dataset,batch_sampler=sampler,num_workers=num_workers,collate_fn=lambda x:collate_batch(x,vocab.pad_id)); iterator=infinite(loader)
    model=WriterStyleNetwork(len(writers)).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate); started_at=time.monotonic()
    print(f"[writer] samples={len(dataset)} writers={len(writers)} target={steps} device={device} batch=32",flush=True)
    for step in range(steps):
        batch=next(iterator); image=batch["target_image"].to(device); labels=torch.tensor([writers[x] for x in batch["writer_id"]],device=device); optimizer.zero_grad(set_to_none=True)
        embedding,logits=model(image); ce=F.cross_entropy(logits,labels); contrast=supervised_contrastive_loss(embedding[:,None],labels); loss=ce+.1*contrast; loss.backward(); optimizer.step(); update=step+1
        if should_emit(update,config.log_every,steps):
            accuracy=logits.argmax(1).eq(labels).float().mean()
            log_progress("writer",update,steps,0,started_at,optimizer,device,loss=float(loss.detach()),ce=float(ce.detach()),supcon=float(contrast.detach()),accuracy=float(accuracy.detach()))
        if should_emit(update,config.preview_every,steps):
            path=save_preview(output,"writer",update,[image],config.preview_samples); print(f"[writer] preview={path}",flush=True)
    save_checkpoint(output,{"writer":model},optimizer,step=steps,config_hash=config.digest(),manifest_hash=file_hash(manifest)); print(f"[writer] checkpoint={output} step={steps}",flush=True); return model


def auxiliary_weights(step):
    ramp=max(0,min(1,(step-10_000)/10_000)); return .2*ramp,.1*ramp


def train_var(manifest,vocab_path,writers_path,tokenizer_checkpoint,ocr_checkpoint,writer_checkpoint,output,config: TrainConfig,device="cuda",steps=None,resume=None,num_workers=0,token_cache=None):
    device=torch.device(device); seed_everything(config.seed); vocab=CharacterVocabulary.load(vocab_path); writers=json.loads(Path(writers_path).read_text())
    dataset=IAMOneShotDataset(manifest,vocab,True)
    samples_per_writer=4 if config.batch_size>=4 else 2
    writers_per_batch=max(1,config.batch_size//samples_per_writer)
    sampler=WriterBalancedBatchSampler(dataset,writers_per_batch,samples_per_writer,config.seed)
    micro_batch=writers_per_batch*samples_per_writer
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=num_workers,collate_fn=lambda x:collate_batch(x,vocab.pad_id)); iterator=infinite(loader)
    tokenizer=MultiScaleVQVAE(config.model.latent_dim,config.model.codebook_size,config.model.scales).to(device); load_checkpoint(tokenizer_checkpoint,{"tokenizer":tokenizer},map_location=device,strict=True,restore_random=False); tokenizer.freeze()
    conditioner=TextStyleConditioner(len(vocab),config.model.model_dim,config.model.heads,config.model.text_layers,config.model.style_layers,config.model.fusion_layers,vocab.pad_id).to(device)
    var=ConditionalVAR(config.model.codebook_size,config.model.latent_dim,config.model.model_dim,config.model.heads,config.model.var_layers,config.model.scales).to(device)
    ocr_payload=torch.load(ocr_checkpoint,map_location=device,weights_only=False); chars=ocr_payload.get("extra",{}).get("characters",vocab.symbols[len(SPECIALS):]); charmap={ch:i+1 for i,ch in enumerate(chars)}
    ocr=WordOCR(len(chars)+1).to(device); ocr.load_state_dict(ocr_payload["models"]["ocr"]); freeze_auxiliary(ocr)
    writer=WriterStyleNetwork(len(writers)).to(device); writer.load_state_dict(torch.load(writer_checkpoint,map_location=device,weights_only=False)["models"]["writer"]); freeze_auxiliary(writer)
    params=list(conditioner.parameters())+list(var.parameters()); optimizer=torch.optim.AdamW(params,lr=config.learning_rate); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config.var_steps); scaler=make_scaler(config.amp); start=0
    if resume: start=load_checkpoint(resume,{"tokenizer":tokenizer,"conditioner":conditioner,"var":var,"ocr":ocr,"writer":writer},optimizer,scheduler,scaler,map_location=device)["step"]
    final=steps or config.var_steps; optimizer.zero_grad(set_to_none=True); started_at=time.monotonic()
    print(f"[var] samples={len(dataset)} writers={len(writers)} start={start} target={final} device={device} "
          f"micro_batch={micro_batch} accumulation={config.accumulation} effective_batch={micro_batch*config.accumulation} "
          f"token_cache={bool(token_cache)}",flush=True)
    for micro_step in range(start*config.accumulation,final*config.accumulation):
        step=micro_step//config.accumulation
        batch=next(iterator); image=batch["target_image"].to(device); reference=batch["reference_image"].to(device); ids=batch["text_ids"].to(device); text_mask=batch["text_mask"].to(device)
        labels=torch.tensor([writers[x] for x in batch["writer_id"]],device=device); drop_t=torch.rand(len(ids),device=device)<config.text_dropout; drop_s=torch.rand(len(ids),device=device)<config.style_dropout
        if token_cache:
            cached=[torch.load(Path(token_cache)/f"{sample_id}.pt",map_location=device,weights_only=True) for sample_id in batch["sample_id"]]
            targets=[torch.stack([row[i] for row in cached]) for i in range(len(config.model.scales))]
        else:
            with torch.no_grad(): _,targets,_,_,_=tokenizer.encode(image)
        with amp_context(device,config.amp):
            raw_style=conditioner.style(reference); style_loss=supervised_contrastive_loss(raw_style,labels)
            text_cond,condition_mask,style=conditioner(ids,reference,drop_t,drop_s); logits=var(targets,text_cond,style,condition_mask); ar,scale_losses,accuracies=autoregressive_loss(logits,targets)
            total=ar+.1*style_loss; ctc=ar.new_zeros(()); writer_loss=ar.new_zeros(()); ctc_weight,writer_weight=auxiliary_weights(step)
            if (step+1)%config.auxiliary_every==0 and (ctc_weight or writer_weight):
                temperature=1-.5*min(1,step/config.var_steps); predicted=categorical_straight_through(logits,tokenizer,temperature)
                keep_text=~drop_t
                if keep_text.any(): ctc=ctc_loss(ocr(predicted[keep_text]),[t for t,k in zip(batch["target_text"],keep_text.tolist()) if k],charmap); total=total+ctc_weight*ctc
                keep_style=~drop_s
                if keep_style.any(): writer_loss=F.cross_entropy(writer(predicted[keep_style])[1],labels[keep_style]); total=total+writer_weight*writer_loss
        scaler.scale(total/config.accumulation).backward()
        if (micro_step+1)%config.accumulation==0:
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(params,config.grad_clip); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        update=step+1
        boundary=(micro_step+1)%config.accumulation==0
        if boundary and should_emit(update,config.log_every,final):
            values={"total":float(total.detach()),"ar":float(ar.detach()),"style":float(style_loss.detach()),
                    "ctc":float(ctc.detach()),"writer":float(writer_loss.detach())}
            values.update({f"acc_s{i+1}":float(value.detach()) for i,value in enumerate(accuracies)})
            log_progress("var",update,final,start,started_at,optimizer,device,**values)
        if boundary and should_emit(update,config.preview_every,final):
            with torch.no_grad():
                predicted_preview=tokenizer.decode_tokens([value.argmax(-1) for value in logits])
            path=save_preview(output,"var",update,[image,reference,predicted_preview],config.preview_samples)
            print(f"[var] preview={path} columns=target,reference,prediction",flush=True)
        if boundary and (update%config.checkpoint_every==0 or update==final):
            extra={"model_config":asdict(config.model),"vocabulary":vocab.symbols[len(SPECIALS):],"scale_loss":[float(x.detach()) for x in scale_losses],"scale_accuracy":[float(x.detach()) for x in accuracies]}
            save_checkpoint(output,{"tokenizer":tokenizer,"conditioner":conditioner,"var":var,"ocr":ocr,"writer":writer},optimizer,scheduler,scaler,update,config.digest(),file_hash(manifest),extra)
            print(f"[var] checkpoint={output} step={update}",flush=True)
    return var
