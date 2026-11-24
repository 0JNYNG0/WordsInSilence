"""Operational bindings to frozen inputs/modules. No model implementation here."""
from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
import torch
H=Path(__file__).resolve().parent;L=H.parents[1];R=L/'runs/motion_position_pair_v1'
OLD=L/'ops/partial_encoder_pair_20260922';MOTION=L/'ops/motion_input_contract_20260922'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
old=module('frozen_partial_io',OLD/'local_io.py')
motion=module('motion_base',MOTION/'generated_v1.py')
read=old.read;sha=old.sha;write=old.write;plain=old.plain
def implementation():return module('paired_input_implementation',H/'generated_v1.py')
def policy():
    p=read(H/'comparison_policy.json')
    for path,d in p['pins'].items():assert sha(path)==d,path
    return p
def inputs(p):
    with np.load(p['cache']['path'],allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ['base_y','pair_context_y','teacher_base_logits','base_indices','context_indices','velocity_scale','classes']}
        base=old.unpack(z['base_x'],z['base_offsets']);context=old.unpack(z['pair_context_x'],z['pair_context_offsets'])
    return base,context,arrays
def parent(p):
    ck=torch.load(p['parent']['path'],map_location='cpu',weights_only=True);model,classes=old.u.rebuild_from_payload(ck);model.eval();torch.set_num_threads(2)
    return ck,model,classes
def checkpoint_model(spec,impl,p):
    assert sha(spec['path'])==spec['sha256'];ck=torch.load(spec['path'],map_location='cpu',weights_only=True)
    _,base,classes=parent(p);model=impl.PairedInputTCN(base,ck['state_dict']['velocity_scale'],ck['feature_mode']);model.load_state_dict(ck['state_dict']);model.eval()
    assert ck['classes']==classes
    return ck,model,classes
