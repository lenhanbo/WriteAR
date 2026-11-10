from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .auxiliary import WordOCR, WriterStyleNetwork, ctc_loss, freeze_auxiliary
from .checkpoint import file_hash, load_checkpoint, save_checkpoint
from .conditioning import TextStyleConditioner, supervised_contrastive_loss
from .config import TrainConfig
from .data import IAMOneShotDataset, WriterBalancedBatchSampler, collate_batch
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


def train_tokenizer(manifest,vocab_path,output,config: TrainConfig,device="cuda",steps=None,resume=None,num_workers=0):
    device=torch.device(device); seed_everything(config.seed); vocab=CharacterVocabulary.load(vocab_path)
    dataset=IAMOneShotDataset(manifest,vocab,True); loader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=num_workers,collate_fn=lambda x:collate_batch(x,vocab.pad_id),drop_last=True)
    model=MultiScaleVQVAE(config.model.latent_dim,config.model.codebook_size,config.model.scales).to(device); discriminator=PatchDiscriminator().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate); d_optimizer=torch.optim.AdamW(discriminator.parameters(),lr=config.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config.tokenizer_steps); scaler=make_scaler(config.amp); start=0
    if resume: start=load_checkpoint(resume,{"tokenizer":model,"discriminator":discriminator},optimizer,scheduler,scaler,map_location=device)["step"]
    iterator=infinite(loader); final=steps or config.tokenizer_steps; optimizer.zero_grad(set_to_none=True)
    for micro_step in range(start*config.accumulation,final*config.accumulation):
        step=micro_step//config.accumulation
        batch=next(iterator); image=batch["target_image"].to(device); weight=adversarial_weight(step)
        with amp_context(device,config.amp):
            reconstruction,_,codebook,commit,metrics=model(image); rec=reconstruction_loss(reconstruction,image)
            generator_adv=-discriminator(reconstruction).mean() if weight else rec.new_zeros(()); loss=rec+codebook+.25*commit+weight*generator_adv
        scaler.scale(loss/config.accumulation).backward()
        if (micro_step+1)%config.accumulation==0:
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        if step>=10_000:
            d_optimizer.zero_grad(set_to_none=True)
            with amp_context(device,config.amp): dloss=F.relu(1-discriminator(image)).mean()+F.relu(1+discriminator(reconstruction.detach())).mean()
            dloss.backward(); d_optimizer.step()
        update=step+1
        if (micro_step+1)%config.accumulation==0 and (update%config.checkpoint_every==0 or update==final):
            save_checkpoint(output,{"tokenizer":model,"discriminator":discriminator},optimizer,scheduler,scaler,update,config.digest(),file_hash(manifest),{"metrics":{k:float(v) for k,v in metrics.items()}})
    return model


def train_ocr(manifest,vocab_path,output,config: TrainConfig,device="cuda",steps=20_000,num_workers=0):
    device=torch.device(device); vocab=CharacterVocabulary.load(vocab_path); dataset=IAMOneShotDataset(manifest,vocab,True)
    loader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=num_workers,collate_fn=lambda x:collate_batch(x,vocab.pad_id),drop_last=True); iterator=infinite(loader)
    chars=vocab.symbols[len(SPECIALS):]; mapping={ch:i+1 for i,ch in enumerate(chars)}; model=WordOCR(len(chars)+1).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    for step in range(steps):
        batch=next(iterator); optimizer.zero_grad(set_to_none=True); logits=model(batch["target_image"].to(device)); loss=ctc_loss(logits,batch["target_text"],mapping); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
    save_checkpoint(output,{"ocr":model},optimizer,step=steps,config_hash=config.digest(),manifest_hash=file_hash(manifest),extra={"characters":chars}); return model


def train_writer(manifest,vocab_path,writers_path,output,config: TrainConfig,device="cuda",steps=20_000,num_workers=0):
    device=torch.device(device); vocab=CharacterVocabulary.load(vocab_path); writers=json.loads(Path(writers_path).read_text()); dataset=IAMOneShotDataset(manifest,vocab,True)
    sampler=WriterBalancedBatchSampler(dataset); loader=DataLoader(dataset,batch_sampler=sampler,num_workers=num_workers,collate_fn=lambda x:collate_batch(x,vocab.pad_id)); iterator=infinite(loader)
    model=WriterStyleNetwork(len(writers)).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    for step in range(steps):
        batch=next(iterator); labels=torch.tensor([writers[x] for x in batch["writer_id"]],device=device); optimizer.zero_grad(set_to_none=True)
        embedding,logits=model(batch["target_image"].to(device)); loss=F.cross_entropy(logits,labels)+.1*supervised_contrastive_loss(embedding[:,None],labels); loss.backward(); optimizer.step()
    save_checkpoint(output,{"writer":model},optimizer,step=steps,config_hash=config.digest(),manifest_hash=file_hash(manifest)); return model


def auxiliary_weights(step):
    ramp=max(0,min(1,(step-10_000)/10_000)); return .2*ramp,.1*ramp


def train_var(manifest,vocab_path,writers_path,tokenizer_checkpoint,ocr_checkpoint,writer_checkpoint,output,config: TrainConfig,device="cuda",steps=None,resume=None,num_workers=0,token_cache=None):
    device=torch.device(device); seed_everything(config.seed); vocab=CharacterVocabulary.load(vocab_path); writers=json.loads(Path(writers_path).read_text())
    dataset=IAMOneShotDataset(manifest,vocab,True); sampler=WriterBalancedBatchSampler(dataset); loader=DataLoader(dataset,batch_sampler=sampler,num_workers=num_workers,collate_fn=lambda x:collate_batch(x,vocab.pad_id)); iterator=infinite(loader)
    tokenizer=MultiScaleVQVAE(config.model.latent_dim,config.model.codebook_size,config.model.scales).to(device); load_checkpoint(tokenizer_checkpoint,{"tokenizer":tokenizer},map_location=device,strict=True,restore_random=False); tokenizer.freeze()
    conditioner=TextStyleConditioner(len(vocab),config.model.model_dim,config.model.heads,config.model.text_layers,config.model.style_layers,config.model.fusion_layers,vocab.pad_id).to(device)
    var=ConditionalVAR(config.model.codebook_size,config.model.latent_dim,config.model.model_dim,config.model.heads,config.model.var_layers,config.model.scales).to(device)
    ocr_payload=torch.load(ocr_checkpoint,map_location=device,weights_only=False); chars=ocr_payload.get("extra",{}).get("characters",vocab.symbols[len(SPECIALS):]); charmap={ch:i+1 for i,ch in enumerate(chars)}
    ocr=WordOCR(len(chars)+1).to(device); ocr.load_state_dict(ocr_payload["models"]["ocr"]); freeze_auxiliary(ocr)
    writer=WriterStyleNetwork(len(writers)).to(device); writer.load_state_dict(torch.load(writer_checkpoint,map_location=device,weights_only=False)["models"]["writer"]); freeze_auxiliary(writer)
    params=list(conditioner.parameters())+list(var.parameters()); optimizer=torch.optim.AdamW(params,lr=config.learning_rate); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config.var_steps); scaler=make_scaler(config.amp); start=0
    if resume: start=load_checkpoint(resume,{"tokenizer":tokenizer,"conditioner":conditioner,"var":var,"ocr":ocr,"writer":writer},optimizer,scheduler,scaler,map_location=device)["step"]
    final=steps or config.var_steps; optimizer.zero_grad(set_to_none=True)
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
        if (micro_step+1)%config.accumulation==0 and (update%config.checkpoint_every==0 or update==final):
            extra={"model_config":asdict(config.model),"vocabulary":vocab.symbols[len(SPECIALS):],"scale_loss":[float(x.detach()) for x in scale_losses],"scale_accuracy":[float(x.detach()) for x in accuracies]}
            save_checkpoint(output,{"tokenizer":tokenizer,"conditioner":conditioner,"var":var,"ocr":ocr,"writer":writer},optimizer,scheduler,scaler,update,config.digest(),file_hash(manifest),extra)
    return var
