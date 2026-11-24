"""Read-only public-data diagnosis. No optimizer, fitting, DEV inference or selection."""
from pathlib import Path
import sys,json,hashlib,traceback,time
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;O=D.parent;ROOT=O.parents[2]
U=O/'word100_auxiliary_supervision_train_20261019'
P=O/'word100_frozen_order_aux_preflight_20261019'
B=O/'word100_body_boundary_20261017'
sys.path.insert(0,str(P));sys.path.insert(0,str(B))
from frozen_order_aux import FrozenOrderAux
from training_core import padded_arrays
SEEDS=(20261013,20261014,20261015);VIEWS=('masked','sensor')
CONDITIONS=('original','swap_first_two_thirds','reverse_thirds','prefix_two_thirds','suffix_third','last_feature')
SAMPLES=tuple(range(0,899,100))+(898,)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,x):
 p=Path(p);assert not p.exists(),p
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def statehash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def order(n,c):
 a,b,z=np.array_split(np.arange(n),3)
 assert len(z)>0
 return {'original':np.arange(n),'swap_first_two_thirds':np.r_[b,a,z],
  'reverse_thirds':np.r_[z,b,a],'prefix_two_thirds':np.r_[a,b],
  'suffix_third':z,'last_feature':np.array([n-1])}[c]
def freeze():
 ctx=read(U/'TRAIN_CONTEXT.json');rows=sorted((r for r in ctx['rows'] if r['role']=='train_candidate'),key=lambda x:x['clip_id'])
 assert len(rows)==899 and len({r['label_index'] for r in rows})==100
 assert {r['signer_id'] for r in rows}==set(range(1,10))
 assert not read(U/'CURRENT_EXECUTION.json')['running'] and not (U/'TRAIN_RUNNING.json').exists()
 # Reuse the prior closed unit's recorded protection hashes, without changing its evidence.
 pins=dict(ctx['pins'])
 for p in [Path(__file__),U/'FINAL_AUDIT.json',U/'ANALYSIS.json',U/'COMPARISON_INDEPENDENT.json',
           O/'word100_sequence_input_audit_20261019/SUMMARY.json']:
  pins[str(p)]=sha(p)
 for seed in SEEDS:
  for suffix in ('best.pt','RESULT.json','history.json'):
   p=U/f'FG16AUX01_{seed}_{suffix}';pins[str(p)]=sha(p)
 write(D/'SCOPE.json',{'at':now(),'status':'FROZEN_DIAGNOSIS_NO_TRAINING','authorization':'Public evidence diagnosis and common stage/order plan only; user requests review before further training.',
 'train_rows':rows,'labels':ctx['labels'],'seeds':SEEDS,'views':VIEWS,'conditions':CONDITIONS,'sample_indices':SAMPLES,
 'procedure':'Once per selected fixed checkpoint, compute public TRAIN899 original frozen TCN features. Change ONLY the sequence read by the fixed GRU. No DEV inference; DEV saved outputs/history only. All conditions specified before output. No optimization/threshold/seed/epoch selection.',
 'limits':['Feature perturbations are not raw videos or labeled invalid signs.','TCN features already contain bidirectional context (maximum61 input frames); suffix/last features can retain earlier raw information.','Thirds are equal frame-count blocks, not annotated semantic stages.','Report auxiliary sensitivity, not sequence correctness or generalization.','Existing selected checkpoints only; no best score or weight search.'],
 'metrics':['Original aux/base/final TRAIN label hit count (not validation).','Aux class agreement and centered-logit RMS change relative to original.','All100-class coverage and per-class count.','Saved DEV base/aux complementarity and correct-vs-rival residual direction.','Selected epoch versus recorded later learning trends.'],
 'forbidden':['training','DEV/heldout new inference','user/mentor recordings','production/model/web edits','downloads','Git','external transfer'],
 'pins':pins})
def main():
 if not (D/'SCOPE.json').exists():freeze()
 s=read(D/'SCOPE.json');assert not (D/'COMPLETE.json').exists()
 for p,h in s['pins'].items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 rows=s['train_rows'];xs={v:[] for v in VIEWS}
 for row in rows:
  for v in VIEWS:
   p=row['views'][v]['channel_file'];assert sha(p)==row['views'][v]['sha256']
   with np.load(p,allow_pickle=False) as z:xs[v].append(z['x_whole'].copy())
 lengths=np.array([[len(x) for x in xs[v]] for v in VIEWS]);assert lengths.min()>=3
 aux=np.zeros((3,2,899,6,100),np.float32);base=np.zeros((3,2,899,100),np.float32)
 samples={};start=time.monotonic();statechecks=[];max_original_error=0.
 with torch.inference_mode():
  for si,seed in enumerate(SEEDS):
   ck=torch.load(U/f'FG16AUX01_{seed}_best.pt',map_location='cpu',weights_only=True)
   bs={k.removeprefix('base.'):v for k,v in ck['state_dict'].items() if k.startswith('base.')}
   m=FrozenOrderAux(ck['model_config'],bs);m.load_state_dict(ck['state_dict']);m.requires_grad_(False);m.eval();before=statehash(m)
   for k,v in m.memory.state_dict().items():samples[f'w{si}_{k}']=v.numpy()
   samples[f'head{si}']=m.residual.weight.numpy()
   for vi,v in enumerate(VIEWS):
    for j in range(0,899,32):
     data=xs[v][j:j+32];xx,pad=padded_arrays(data)
     b,a,_,h,lens=m.components(torch.from_numpy(xx),torch.from_numpy(pad));base[si,vi,j:j+len(data)]=b.numpy()
     for ii in range(len(data)):
      idx=j+ii
      if idx in SAMPLES:samples[f'h{si}_{vi}_{idx}']=h[ii,:lens[ii]].numpy().copy()
     for ci,c in enumerate(CONDITIONS):
      seq=[h[ii,order(int(n),c)] for ii,n in enumerate(lens)];ll=torch.tensor([len(z) for z in seq])
      pp=torch.nn.utils.rnn.pad_sequence(seq,batch_first=True)
      logits=m.residual(m.remember(pp,ll));aux[si,vi,j:j+len(data),ci]=logits.numpy()
      if c=='original':
       err=float((a-logits).abs().max());max_original_error=max(max_original_error,err);assert err==0
    print(json.dumps({'seed':seed,'view':v,'completed_train_clips':899,'seconds':round(time.monotonic()-start,1)}),flush=True)
   assert statehash(m)==before;statechecks.append(before)
 np.savez_compressed(D/'PROBES.npz',aux=aux,base=base,lengths=lengths,labels=np.array([r['label_index'] for r in rows]),ids=np.array([r['clip_id'] for r in rows]))
 np.savez_compressed(D/'NUMERIC_SAMPLES.npz',**samples)
 for p,h in s['pins'].items():assert sha(p)==h,p
 write(D/'COMPLETE.json',{'status':'COMPLETE_DIAGNOSIS_PENDING_VERIFICATION','at':now(),'scope_sha256':sha(D/'SCOPE.json'),
 'probes_sha256':sha(D/'PROBES.npz'),'numeric_samples_sha256':sha(D/'NUMERIC_SAMPLES.npz'),'source_pins_unchanged':len(s['pins']),
 'model_state_unchanged':statechecks,'original_feature_pass_exact_error':max_original_error,
 'public_train_clips':899,'classes':100,'views':2,'seeds':3,'conditions':6,'seconds':time.monotonic()-start,
 'training_runs':0,'DEV_inferences':0,'user_recordings_read':0,'heldout_read':0,'production_changed':False})
if __name__=='__main__':
 try:main()
 except BaseException:
  p=D/'FIRST_ERROR.json'
  if not p.exists():write(p,{'at':now(),'traceback':traceback.format_exc()})
  raise
