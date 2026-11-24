"""Isolated RGB/pose connection component. Not a trained sign recognizer."""
from pathlib import Path
import copy
import hashlib
import re
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


def require(ok, message):
    if not ok: raise ValueError(message)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


class FrozenRGBEncoder(nn.Module):
    """Explicit local weight file only. Features shared between the two hands."""
    feature_dim=576

    def __init__(self, path, expected_sha256):
        super().__init__()
        require(bool(re.fullmatch(r'[0-9a-f]{64}',expected_sha256)), 'Full SHA256 required')
        if not Path(path).is_file(): raise FileNotFoundError('RGB checkpoint missing')
        require(file_sha(path)==expected_sha256, 'RGB checkpoint hash mismatch')
        model=mobilenet_v3_small(weights=None)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.features=model.features
        self.pool=model.avgpool
        self.checkpoint_sha256=expected_sha256
        self.forward_images=0
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images, batch_size=16):
        require(images.device.type=='cpu' and images.dtype==torch.float32, 'CPU float32 images required')
        require(images.ndim==4 and tuple(images.shape[1:])==(3,224,224), 'Expected K,3,224,224')
        require(bool(torch.isfinite(images).all()), 'Nonfinite RGB input')
        require(isinstance(batch_size,int) and batch_size>0, 'Invalid batch size')
        if not len(images): return images.new_empty((0,self.feature_dim))
        result=[]
        for chunk in images.split(batch_size):
            result.append(self.pool(self.features(chunk)).flatten(1))
        out=torch.cat(result)
        require(out.shape==(len(images),self.feature_dim) and bool(torch.isfinite(out).all()), 'Invalid RGB features')
        self.forward_images+=len(images)
        return out


class PoseRGBBridge(nn.Module):
    """Zero-initialized additional input path before unchanged temporal blocks.

    Only the connection projection is trainable in this implementation check.
    A future training experiment must separately freeze its training contract.
    """
    status='CONNECTION_ONLY_NOT_TRAINED_FOR_SIGN_CLASSIFICATION'

    def __init__(self, parent, feature_dim=576):
        super().__init__()
        require(isinstance(parent.proj,nn.Conv1d) and parent.proj.kernel_size==(1,), 'Expected pointwise pose projection')
        require(isinstance(feature_dim,int) and feature_dim>0, 'Invalid feature dimension')
        self.base=copy.deepcopy(parent).cpu().eval()
        self.base.requires_grad_(False)
        self.feature_dim=feature_dim
        self.rgb_projection=nn.Linear(feature_dim*2+4,self.base.proj.out_channels,bias=False)
        nn.init.zeros_(self.rgb_projection.weight)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Never mutate checkpoint normalization statistics.
        return self

    def forward(self,x,pad,features,available,estimated):
        require(x.ndim==3 and x.shape[2]==self.base.proj.in_channels and min(x.shape[:2])>0, 'Invalid pose dimensions')
        b,t,_=x.shape
        for value in (x,pad,features):
            require(value.device.type=='cpu' and value.dtype==torch.float32, 'CPU float32 input required')
        require(pad.shape==(b,t) and bool(((pad==0)|(pad==1)).all()) and bool((pad.sum(1)>0).all()), 'Invalid time mask')
        require(bool(torch.isfinite(x).all()), 'Nonfinite pose')
        require(features.shape==(b,t,2,self.feature_dim), 'Invalid feature dimensions')
        for value in (available,estimated):
            require(value.device.type=='cpu' and value.dtype==torch.bool and value.shape==(b,t,2), 'Invalid hand mask')
        require(not bool((estimated & ~available).any()), 'Estimated image must be available')
        require(not bool((available & ~pad.bool().unsqueeze(-1)).any()), 'RGB in padded time')
        require(bool(torch.isfinite(features[available]).all()), 'Nonfinite available feature')
        clean=torch.where(available.unsqueeze(-1),features,torch.zeros_like(features))
        extra=torch.cat([clean.reshape(b,t,-1),available.float(),estimated.float()],dim=-1)
        m=pad.unsqueeze(1)
        h=self.base.proj(x.transpose(1,2)*m)*m
        h=h+self.rgb_projection(extra).transpose(1,2)*m
        for block in self.base.blocks: h=block(h,m)
        pooled=(h*m).sum(2)/pad.sum(1).clamp(min=1.).unsqueeze(1)
        return self.base.head(pooled)
