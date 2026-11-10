"""Auxiliary recognizers ported from CharWriteViT under the MIT license.

See THIRD_PARTY_NOTICES.md. This module deliberately has no runtime import from
CharWriteViT.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__(); self.conv1=nn.Conv2d(inputs,outputs,3,stride,1,bias=False); self.bn1=nn.BatchNorm2d(outputs)
        self.conv2=nn.Conv2d(outputs,outputs,3,1,1,bias=False); self.bn2=nn.BatchNorm2d(outputs)
        self.skip=nn.Identity() if stride==1 and inputs==outputs else nn.Sequential(nn.Conv2d(inputs,outputs,1,stride,bias=False),nn.BatchNorm2d(outputs))
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x))); return F.relu(self.bn2(self.conv2(out))+self.skip(x))


class OCRBackbone(nn.Module):
    def __init__(self,width=256):
        super().__init__(); base=width//4
        self.net=nn.Sequential(nn.Conv2d(1,base,3,(2,1),1,bias=False),nn.BatchNorm2d(base),nn.ReLU(),nn.MaxPool2d(3,(2,1),1),
            BasicBlock(base,base,(2,2)),BasicBlock(base,base),BasicBlock(base,width//2,2),BasicBlock(width//2,width//2),
            BasicBlock(width//2,width,2),BasicBlock(width,width),nn.MaxPool2d(3,1,1))
    def forward(self,x): return self.net(x)


class WordOCR(nn.Module):
    """Returns 32 CTC time steps for a 32x256 input."""
    def __init__(self, classes: int, width=256, depth=3, heads=8):
        super().__init__(); self.features=OCRBackbone(width)
        layer=nn.TransformerEncoderLayer(width,heads,width*4,batch_first=True,norm_first=True,activation="gelu")
        self.blocks=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False); self.norm=nn.LayerNorm(width); self.head=nn.Linear(width,classes)
    def forward(self,image):
        if image.shape[-2:] != (32,256): image=F.interpolate(image,(32,256),mode="bilinear",align_corners=False)
        feature=self.features(image); tokens=feature.flatten(2).transpose(1,2)
        return self.head(self.norm(self.blocks(tokens)))


class WriterStyleNetwork(nn.Module):
    def __init__(self,writers: int,embedding_dim=256):
        super().__init__(); channels=(64,128,256,embedding_dim); layers=[]; previous=1
        for channel in channels:
            layers += [nn.Conv2d(previous,channel,4,2,1),nn.GroupNorm(8,channel),nn.SiLU()]; previous=channel
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(embedding_dim,writers)
    def forward(self,image):
        embedding=F.normalize(self.encoder(image).flatten(1),dim=-1); return embedding,self.head(embedding)


def freeze_auxiliary(module: nn.Module):
    module.eval()
    for parameter in module.parameters(): parameter.requires_grad_(False)
    return module


def ctc_targets(texts: list[str], char_to_id: dict[str,int], device=None):
    values=[]; lengths=[]
    for text in texts:
        encoded=[char_to_id[ch] for ch in text if ch in char_to_id]; values.extend(encoded); lengths.append(len(encoded))
    return torch.tensor(values,dtype=torch.long,device=device),torch.tensor(lengths,dtype=torch.long,device=device)


def ctc_loss(logits,texts,char_to_id,blank=0):
    targets,target_lengths=ctc_targets(texts,char_to_id,logits.device)
    input_lengths=torch.full((logits.shape[0],),logits.shape[1],dtype=torch.long,device=logits.device)
    return F.ctc_loss(logits.log_softmax(-1).transpose(0,1),targets,input_lengths,target_lengths,blank=blank,zero_infinity=True)

