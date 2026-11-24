"""Unique module name avoids older experiment imports named core."""
from pathlib import Path
import sys,json,hashlib,time,importlib.util,copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/'word100_bilateral_completion_20261111'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sys.path.insert(0,str(P));prev=load('_boundary_previous',P/'completion_model.py')
rc,prior,old,mc,jc,bc=prev.rc,prev.prior,prev.old,prev.mc,prev.jc,prev.bc
R,A,J,B,M,D=prev.R,prev.A,prev.J,prev.B,prev.M,prev.D
read,write,sha=prev.read,prev.write,prev.sha
def event(kind,**kw):
 r=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps(r,ensure_ascii=False),flush=True)
