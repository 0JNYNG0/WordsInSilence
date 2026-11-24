"""Execute one verified generic learner on pinned public TRAINING matrices only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy
import hashlib
import json
import sys
import time
import numpy as np
import torch

H=Path(__file__).resolve().parent;L=H.parents[1];OLD=H.parent/'context_training_pair_20260921'
sys.path[:0]=[str(H),str(OLD)]
from generated_candidate import train_head
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v
def write(path,data):
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(plain(data),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    ep=read(H/'execution_plan.json');policy=read(ep['policy']);audit=read(H/'IMPLEMENTATION_AUDIT.json')
    assert audit['status']=='PASS' and sha(H/'generated_candidate.py')==audit['candidate_module_sha256']
    for path,h in ep['source_pins'].items():assert sha(path)==h,path
    model_spec=policy['teacher_and_initialization'];assert sha(model_spec['path'])==model_spec['sha256']
    parent=torch.load(model_spec['path'],map_location='cpu',weights_only=True)
    with np.load(policy['data']['feature_path'],allow_pickle=False) as z:
        a={k:z[k].copy() for k in ep['required_training_keys']}
    with np.load(policy['data']['sample_schedule_path'],allow_pickle=False) as z:
        schedule={k:z[k].copy() for k in ep['required_training_schedule_keys']}
    with np.load(policy['data']['gram_path'],allow_pickle=False) as z:
        gram=z['gram'].copy();assert z['classes'].tolist()==parent['classes']
    assert a['classes'].tolist()==parent['classes'] and a['base_x'].shape==(2122,64) and a['pair_context_x'].shape==(224,64)
    assert gram.shape==(65,65) and gram.dtype==np.float64 and set(a['base_y'])==set(range(93))
    assert all(x.shape==(660,16) for x in schedule.values())
    meta=read(OLD/'feature_metadata.json')
    assert all(r['role']=='train' for r in meta['base']) and all(r['role']=='train' and 1<=r['signer_id']<=9 for r in meta['pair_context'])
    assert {r['signer_id'] for r in meta['base'] if 'signer_id' in r}==set(range(1,10))
    run=L/'runs/output_retention_candidate_v1';run.mkdir(exist_ok=False)
    input_pins={**ep['source_pins'],**{str(p):sha(p) for p in [H/'execution_plan.json',H/'generated_candidate.py',H/'IMPLEMENTATION_AUDIT.json',Path(__file__)]}}
    write(run/'preflight.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),input_pins=input_pins,parent=model_spec,baseline=policy['reused_baseline'],allowed_keys=ep['required_training_keys'],new_training_count=1,baseline_retraining=False))
    cfg=policy['settings'];kw={k:cfg[k] for k in ['steps','batch_size','learning_rate','weight_decay','anchor_coefficient']};kw['output_coefficient']=cfg['output_preservation_coefficient']
    torch.set_num_threads(cfg['cpu_threads']);assert torch.get_default_device()==torch.device('cpu')
    w0=parent['state_dict']['head.weight'].numpy().copy();b0=parent['state_dict']['head.bias'].numpy().copy()
    print('Starting the single fixed660-update candidate.',flush=True);start=time.time()
    output=train_head(a['base_x'],a['base_y'],a['pair_context_x'],a['pair_context_y'],w0,b0,schedule['base_indices'],schedule['pair_indices'],gram,**kw)
    seconds=time.time()-start
    assert all(np.array_equal(output['schedule'][k],schedule[k]) for k in schedule)
    ck=copy.deepcopy(parent)
    ck['state_dict']['head.weight']=torch.from_numpy(output['weight'].copy());ck['state_dict']['head.bias']=torch.from_numpy(output['bias'].copy())
    ck['output_retention_provenance']=dict(parent_sha256=model_spec['sha256'],policy_sha256=sha(ep['policy']),settings=plain(output['settings']),encoder_frozen=True,old_payload_metadata_is_ancestry=True)
    checkpoint=run/'checkpoint_final.pt';torch.save(ck,checkpoint)
    with (run/'training_arrays.npz').open('xb') as f:np.savez_compressed(f,weight=output['weight'],bias=output['bias'],pre_update_losses=output['pre_update_losses'],**output['schedule'])
    for p,h in input_pins.items():assert sha(p)==h,p
    assert sha(model_spec['path'])==model_spec['sha256']
    training=dict(status='TRAINING_COMPLETE_CHECKPOINT_FROZEN_BEFORE_DEVELOPMENT_EVALUATION',at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        candidate=dict(path=str(checkpoint),sha256=sha(checkpoint)),parent=model_spec,reused_baseline=policy['reused_baseline'],
        settings=output['settings'],seconds=seconds,optimizer_updates=660,new_training_runs=1,baseline_retraining=False,retries=0,
        final_anchor_distance_squared=output['final_anchor_distance_squared'],final_output_loss_raw=output['final_output_loss_raw'],
        training_arrays_sha256=sha(run/'training_arrays.npz'),new_encoder_forwards=0,user_arrays_read=0,development_arrays_read=0,reserved_arrays_read=0)
    write(run/'training.json',training)
    print(json.dumps(plain(training),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
