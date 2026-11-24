from pathlib import Path
import sys,json,hashlib,time,importlib.util,copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
R=O/'word100_candidate_role_learning_20261111';D=O/'word100_action_feedback_demo_20261108'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rc=load('_bilateral_role',R/'role_core.py')
ac,jc,mc,prior,old,bc,lc=rc.ac,rc.jc,rc.mc,rc.prior,rc.old,rc.bc,rc.lc
A,J,B,M=rc.A,rc.J,rc.B,rc.M
read,write,sha=rc.read,rc.write,rc.sha
def event(kind,**kw):
 row=dict(time=time.time(),kind=kind,**kw)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 print(json.dumps(row,ensure_ascii=False),flush=True)
def opposite(side):
 assert side in ['R','L'];return 'L' if side=='R' else 'R'

