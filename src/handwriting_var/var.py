from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .tokenizer import SCALES, MultiScaleVQVAE


def scale_slices(scales=SCALES):
    result=[]; start=0
    for h,w in scales:
        end=start+h*w; result.append(slice(start,end)); start=end
    return result


def block_causal_mask(scales=SCALES, device=None):
    scale_ids = torch.cat([torch.full((h*w,), i) for i,(h,w) in enumerate(scales)]).to(device)
    return scale_ids[None, :] > scale_ids[:, None]


class SDPAttention(nn.Module):
    def __init__(self, dimension=384, heads=6, cross=False):
        super().__init__(); self.heads=heads; self.head_dim=dimension//heads
        self.q=nn.Linear(dimension,dimension); self.k=nn.Linear(dimension,dimension); self.v=nn.Linear(dimension,dimension); self.out=nn.Linear(dimension,dimension)
    def forward(self, query, context=None, mask=None, key_mask=None):
        context = query if context is None else context; b,n,d=query.shape; m=context.shape[1]
        q=self.q(query).view(b,n,self.heads,self.head_dim).transpose(1,2)
        k=self.k(context).view(b,m,self.heads,self.head_dim).transpose(1,2); v=self.v(context).view(b,m,self.heads,self.head_dim).transpose(1,2)
        attention_mask = None
        if mask is not None: attention_mask = ~mask[None,None]
        if key_mask is not None:
            km=key_mask[:,None,None,:]
            attention_mask = km if attention_mask is None else attention_mask & km
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask)
        return self.out(out.transpose(1,2).reshape(b,n,d))


class GatedVARBlock(nn.Module):
    def __init__(self, dimension=384, heads=6, mlp_ratio=4):
        super().__init__(); self.norms=nn.ModuleList([nn.LayerNorm(dimension) for _ in range(6)])
        self.self_attention=SDPAttention(dimension,heads); self.text_attention=SDPAttention(dimension,heads,True); self.style_attention=SDPAttention(dimension,heads,True)
        self.mlp=nn.Sequential(nn.Linear(dimension,dimension*mlp_ratio),nn.GELU(),nn.Linear(dimension*mlp_ratio,dimension))
        self.gates=nn.Parameter(torch.zeros(4))
    def forward(self,x,text,style,mask,text_mask):
        x=x+self.gates[0]*self.self_attention(self.norms[0](x),mask=mask)
        x=x+self.gates[1]*self.text_attention(self.norms[1](x),self.norms[2](text),key_mask=text_mask)
        x=x+self.gates[2]*self.style_attention(self.norms[3](x),self.norms[4](style))
        return x+self.gates[3]*self.mlp(self.norms[5](x))


class ConditionalVAR(nn.Module):
    def __init__(self, codebook_size=1024, latent_dim=128, dimension=384, heads=6, layers=8, scales=SCALES, gradient_checkpointing=True):
        super().__init__(); self.scales=tuple(scales); self.codebook_size=codebook_size; self.gradient_checkpointing=gradient_checkpointing
        self.token_embedding=nn.Embedding(codebook_size,dimension); self.input_projection=nn.Linear(latent_dim,dimension)
        self.position=nn.Parameter(torch.randn(1,sum(h*w for h,w in scales),dimension)*.02); self.scale_embedding=nn.Embedding(len(scales),dimension)
        self.start=nn.Parameter(torch.randn(1,1,dimension)*.02); self.blocks=nn.ModuleList([GatedVARBlock(dimension,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(dimension); self.heads=nn.ModuleList([nn.Linear(dimension,codebook_size) for _ in scales])
        ids=torch.cat([torch.full((h*w,),i) for i,(h,w) in enumerate(scales)]); self.register_buffer("scale_ids",ids,persistent=False)

    def teacher_inputs(self, tokens: list[torch.Tensor]):
        chunks=[]
        for i,(h,w) in enumerate(self.scales):
            if i==0: chunks.append(self.start.expand(tokens[0].shape[0],h*w,-1))
            else:
                previous=self.token_embedding(tokens[i-1]).permute(0,3,1,2)
                resized=F.interpolate(previous,(h,w),mode="nearest").flatten(2).transpose(1,2); chunks.append(resized)
        return torch.cat(chunks,1)

    def forward(self,tokens,text,style,text_mask):
        x=self.teacher_inputs(tokens)+self.position+self.scale_embedding(self.scale_ids)[None]
        mask=block_causal_mask(self.scales,x.device)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x=checkpoint(block,x,text,style,mask,text_mask,use_reentrant=False)
            else: x=block(x,text,style,mask,text_mask)
        x=self.norm(x); slices=scale_slices(self.scales)
        return [head(x[:,sl]).reshape(x.shape[0],h,w,self.codebook_size) for head,sl,(h,w) in zip(self.heads,slices,self.scales)]

    @torch.no_grad()
    def sample_tokens(self,text,style,text_mask,temperature=1.0,top_k=256,top_p=.95,generator=None):
        batch=text.shape[0]; tokens=[]
        for i,(h,w) in enumerate(self.scales):
            provisional=tokens+[torch.zeros(batch,*s,dtype=torch.long,device=text.device) for s in self.scales[i:]]
            logits=self(provisional,text,style,text_mask)[i]/max(temperature,1e-5)
            flat=logits.reshape(-1,self.codebook_size); flat=filter_logits(flat,top_k,top_p)
            sampled=torch.multinomial(flat.softmax(-1),1,generator=generator).reshape(batch,h,w); tokens.append(sampled)
        return tokens


def filter_logits(logits, top_k=0, top_p=1.0):
    if top_k: logits=logits.masked_fill(logits < logits.topk(min(top_k,logits.shape[-1]),dim=-1).values[:,-1:],-torch.inf)
    if top_p < 1:
        sorted_logits,indices=logits.sort(descending=True); probs=sorted_logits.softmax(-1); remove=probs.cumsum(-1)>top_p
        remove[:,1:]=remove[:,:-1].clone(); remove[:,0]=False
        logits=logits.scatter(1,indices,sorted_logits.masked_fill(remove,-torch.inf))
    return logits


def categorical_straight_through(logits: list[torch.Tensor], tokenizer: MultiScaleVQVAE, temperature=1.0):
    embeddings=[]
    for value in logits:
        soft=(value/temperature).softmax(-1); hard=F.one_hot(soft.argmax(-1),soft.shape[-1]).to(soft.dtype)
        probability=hard+soft-soft.detach(); embeddings.append((probability @ tokenizer.quantizer.codebook.weight).permute(0,3,1,2))
    return tokenizer.decoder(tokenizer.quantizer.merge_embeddings(embeddings))


def autoregressive_loss(logits, targets):
    losses=[]; accuracies=[]
    for prediction,target in zip(logits,targets):
        losses.append(F.cross_entropy(prediction.reshape(-1,prediction.shape[-1]),target.flatten()))
        accuracies.append((prediction.argmax(-1)==target).float().mean())
    weights=torch.tensor([t.numel()/t.shape[0] for t in targets],device=targets[0].device)
    return (torch.stack(losses)*weights).sum()/weights.sum(), losses, accuracies
