from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .config import ModelConfig
from .conditioning import TextStyleConditioner
from .data import canonical_image
from .tokenizer import MultiScaleVQVAE
from .var import ConditionalVAR, filter_logits
from .vocab import CharacterVocabulary


class HandwritingVARPipeline:
    def __init__(self,tokenizer,conditioner,var,vocabulary,device="cpu"):
        self.device=torch.device(device); self.tokenizer=tokenizer.to(self.device).freeze(); self.conditioner=conditioner.to(self.device).eval(); self.var=var.to(self.device).eval(); self.vocabulary=vocabulary

    @classmethod
    def from_checkpoint(cls,path,device="cpu"):
        payload=torch.load(path,map_location="cpu",weights_only=False); raw=payload.get("model_config",payload.get("extra",{}).get("model_config",{}))
        allowed={f.name for f in fields(ModelConfig)}; config=ModelConfig(**{k:v for k,v in raw.items() if k in allowed})
        vocabulary=CharacterVocabulary(payload.get("vocabulary",payload.get("extra",{}).get("vocabulary",[])))
        tokenizer=MultiScaleVQVAE(config.latent_dim,config.codebook_size,config.scales)
        conditioner=TextStyleConditioner(len(vocabulary),config.model_dim,config.heads,config.text_layers,config.style_layers,config.fusion_layers,vocabulary.pad_id)
        var=ConditionalVAR(config.codebook_size,config.latent_dim,config.model_dim,config.heads,config.var_layers,config.scales,False)
        states=payload.get("models",payload)
        tokenizer.load_state_dict(states["tokenizer"]); conditioner.load_state_dict(states["conditioner"]); var.load_state_dict(states["var"])
        return cls(tokenizer,conditioner,var,vocabulary,device)

    @torch.inference_mode()
    def generate(self,text,style_image,text_cfg=4.0,style_cfg=2.0,temperature=1.0,top_k=256,top_p=.95,seed=42):
        if not 0 < len(text) <= 24: raise ValueError("text must contain 1..24 characters")
        if isinstance(style_image,(str,Path)):
            with Image.open(style_image) as image: reference=canonical_image(image)
        elif isinstance(style_image,Image.Image): reference=canonical_image(style_image)
        else: reference=style_image
        reference=reference.to(self.device).unsqueeze(0) if reference.ndim==3 else reference.to(self.device)
        ids=torch.tensor([self.vocabulary.encode(text)],device=self.device); ids3=ids.repeat(3,1); refs3=reference.repeat(3,1,1,1)
        drop_text=torch.tensor([True,False,False],device=self.device); drop_style=torch.tensor([True,True,False],device=self.device)
        text_tokens,text_mask,style=self.conditioner(ids3,refs3,drop_text,drop_style)
        generator=torch.Generator(device=self.device).manual_seed(seed); tokens=[]; batch=1
        for i,(h,w) in enumerate(self.var.scales):
            expanded=[x.repeat(3,1,1) for x in tokens]
            expanded += [torch.zeros(3,*s,dtype=torch.long,device=self.device) for s in self.var.scales[i:]]
            branch=self.var(expanded,text_tokens,style,text_mask)[i]; unconditional,text_only,text_style=branch.chunk(3)
            logits=unconditional+text_cfg*(text_only-unconditional)+style_cfg*(text_style-text_only)
            flat=filter_logits((logits/temperature).reshape(-1,logits.shape[-1]),top_k,top_p)
            tokens.append(torch.multinomial(flat.softmax(-1),1,generator=generator).reshape(batch,h,w))
        image=self.tokenizer.decode_tokens(tokens).clamp(-1,1)
        return TF.to_pil_image(image[0].add(1).div(2).cpu())
