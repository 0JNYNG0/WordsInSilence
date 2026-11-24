"""Public-only frozen-model sensitivity, feature distribution and saved DEV audit.
No optimizer, training, detector, devices, user/mentor or heldout inputs.
"""
import os,sys,json,hashlib,math,time,traceback,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;SRC=O/'word100_hand_evidence_training_20261018';ROOT=O.parents[2]
sys.path.insert(0,str(ROOT/'research/sign_model_pilot_v1/src'))
GROUPS={'position':(0,6,5),'extent':(6,3,0),'intersection':(9,3,0),'proximity':(12,3,0)}
SEEDS=[20261013,20261014,20261015];VIEWS=['masked','sensor']
CONDITIONS=['original']+[f'{mode}_{g}' for mode in ['zero','unknown'] for g in GROUPS]+['zero_all','unknown_all']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
def write(p,x,new=True):
 p=Path(p)
 if new and p.exists():raise FileExistsError(p)
 q=p.with_name(p.name+'.writing');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(q,p)
def pincheck(pins):
 for p,h in pins.items():assert sha(p)==h,p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def freeze():
 f=U/'CONTEXT.json'
 if f.exists():return read(f)
 c=read(SRC/'PREP_CONTEXT.json');assert read(SRC/'FINAL_AUDIT.json')['status']=='PASS';assert not (SRC/'TRAIN_RUNNING.json').exists()
 pins={str(p):sha(p) for p in [SRC/'PREP_CONTEXT.json',SRC/'COMPARISON.json',SRC/'FINAL_AUDIT.json',SRC/'adapter.py',SRC/'train.py',U/'analyze.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py']}
 rows=[]
 for r in c['records']:
  assert r['signer_id'] in (list(range(1,10)) if r['role']=='train_candidate' else [11,12,13])
  p=SRC/'inputs_v2'/f"{r['clip_id']}.json";q=read(p);pins[str(p)]=sha(p);pins.update(q['outputs']);rows.append({**{k:r[k] for k in ['clip_id','role','signer_id','label_index','label']},'views':q['views']})
 for seed in SEEDS:
  for arm in ['C0','C1']:
   for suffix in ['best.pt','RESULT.json','history.json']:
    p=SRC/f'{arm}_{seed}_{suffix}';pins[str(p)]=sha(p)
 ctx={'at':now(),'authorization':'공개 자료로 추가 정보별 성능 저하 원인과 다음 보완 계획 분석. 새 학습은 사용자 결과 검토 후. 기존C/사용자자료보존.','groups':GROUPS,'conditions':CONDITIONS,'seeds':SEEDS,'inference_scope':'TRAIN899 only, whole temporal view, both existing spatial views. C1 frozen eval. Not validation performance.','stored_DEV_analysis':'Re-use existing scores only; no new DEV inference; no selection or thresholds.','methods':['zero removes group off simplex (distribution shift)','unknown assigns category unknown/unavailable (in-simplex but unphysical joint combinations possible)','paired same-input frozen logits only; not retrained ablation or unique causal proof'],'rows':rows,'pins':pins,'training':0,'deployment':0,'user_mentor_heldout_reads':0}
 write(f,ctx);before=U/'before';before.mkdir(exist_ok=True)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,before/p.name)
 return ctx
def feature_stats(ctx,arr):
 out={};perclip=[];viewdiff={g:[0,0] for g in GROUPS};alignment=[]
 for role in ['train_candidate','development_only']:
  out[role]={}
  for view in VIEWS:
   total={g:np.zeros(sz,np.int64) for g,(_,sz,_) in GROUPS.items()};clipmean={g:[] for g in GROUPS};trans={g:[0,0] for g in GROUPS};lowmargin=[]
   for r in ctx['rows']:
    if r['role']!=role:continue
    d=arr[r['clip_id']][view];st=d['states_whole'];ent={'clip_id':r['clip_id'],'role':role,'view':view,'label':r['label'],'signer_id':r['signer_id'],'groups':{}}
    for j,(g,(_,sz,_)) in enumerate(GROUPS.items()):
     n=np.bincount(st[:,:,j].ravel(),minlength=sz);total[g]+=n;clipmean[g].append(n/n.sum());changes=int((st[1:,:,j]!=st[:-1,:,j]).sum());trans[g][0]+=changes;trans[g][1]+=2*(len(st)-1)
     ent['groups'][g]={'fraction':(n/n.sum()).tolist(),'changes_per_hand_frame':changes/max(1,2*(len(st)-1))}
    perclip.append(ent)
   out[role][view]={g:{'hand_frame_counts':total[g].tolist(),'frame_fraction':(total[g]/sum(total[g])).tolist(),'equal_clip_fraction':np.mean(clipmean[g],0).tolist(),'transition_fraction':trans[g][0]/trans[g][1]} for g in GROUPS}
 for r in ctx['rows']:
  a,b=[arr[r['clip_id']][v] for v in VIEWS];frames,ia,ib=np.intersect1d(a['whole_frame_id'],b['whole_frame_id'],return_indices=True);assert len(frames)
  assert np.array_equal(a['whole_ts'][ia],b['whole_ts'][ib])
  alignment.append({'clip_id':r['clip_id'],'masked_frames':len(a['whole_frame_id']),'sensor_frames':len(b['whole_frame_id']),'common_frames':len(frames)})
  for j,g in enumerate(GROUPS):viewdiff[g][0]+=int((a['states_whole'][ia,:,j]!=b['states_whole'][ib,:,j]).sum());viewdiff[g][1]+=len(frames)*2
 res={'distribution':out,'same_clip_spatial_view_disagreement':{g:{'different':x,'total':n,'fraction':x/n} for g,(x,n) in viewdiff.items()},'alignment':alignment,'perclip':perclip}
 write(U/'FEATURE_STATS.json',res);return res
def perturb(x,condition):
 if condition=='original':return x
 a=x.copy();mode,which=condition.split('_',1)
 for g,(off,sz,unk) in GROUPS.items():
  if which!='all' and g!=which:continue
  for side in (0,1):
   start=280+side*15+off;a[:,start:start+sz]=0
   if mode=='unknown':a[:,start+unk]=1
 assert np.array_equal(a[:,:280],x[:,:280]);return a
def score(a,y):
 order=np.argsort(-a,axis=1,kind='stable');mx=a.max(1);ce=mx+np.log(np.exp(a-mx[:,None]).sum(1))-a[np.arange(len(y)),y]
 return {'correct':int((order[:,0]==y).sum()),'ce':float(ce.mean()),'pred':order[:,0],'margins':a[np.arange(len(y)),y]-np.max(np.where(np.eye(100,dtype=bool)[y],-np.inf,a),axis=1)}
def frozen_inference(ctx,arr):
 import torch
 from pilot.model_masked_v1 import MaskedSignTCN
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 train=[r for r in ctx['rows'] if r['role']=='train_candidate'];assert len(train)==899
 y=np.array([r['label_index'] for r in train]);outputs=[]
 for seed in SEEDS:
  dest=U/f'FROZEN_TRAIN_{seed}.npz'
  assert not dest.exists(),'No repeating saved diagnostic inference'
  ck=torch.load(SRC/f'C1_{seed}_best.pt',map_location='cpu',weights_only=True);m=MaskedSignTCN(**ck['model_config']);m.load_state_dict(ck['state_dict']);m.eval()
  original_state={k:v.clone() for k,v in m.state_dict().items()}
  logits=np.zeros((2,len(CONDITIONS),899,100),np.float32)
  for vi,view in enumerate(VIEWS):
   for ci,cond in enumerate(CONDITIONS):
    with torch.no_grad():
     for at in range(0,899,32):
      rows=train[at:at+32];xs=[perturb(arr[r['clip_id']][view]['x_whole'],cond) for r in rows];n=max(map(len,xs));xx=np.zeros((len(xs),n,310),np.float32);pad=np.zeros((len(xs),n),np.float32)
      for j,x in enumerate(xs):xx[j,:len(x)]=x;pad[j,:len(x)]=1
      z=m(torch.from_numpy(xx),torch.from_numpy(pad));assert torch.isfinite(z).all();logits[vi,ci,at:at+len(xs)]=z.numpy()
    write(U/'STATUS.json',{'stage':'FROZEN_TRAIN_SENSITIVITY','seed':seed,'view':view,'condition':cond,'at':now(),'training':0},new=False)
   print('Sensitivity',seed,view,flush=True)
  assert all(torch.equal(v,original_state[k]) for k,v in m.state_dict().items())
  weight=m.proj.weight.detach().numpy()[:,:,0];pn={g:float(np.linalg.norm(weight[:,[280+side*15+k for side in (0,1) for k in range(off,off+sz)]])) for g,(off,sz,_) in GROUPS.items()}
  np.savez_compressed(dest,logits=logits,labels=y,clip_ids=np.array([r['clip_id'] for r in train]),conditions=np.array(CONDITIONS),views=np.array(VIEWS))
  detail={'seed':seed,'projection_weight_norm_only_not_importance':pn,'views':{}}
  for vi,view in enumerate(VIEWS):
   base=score(logits[vi,0].astype(float),y);detail['views'][view]={}
   for ci,cond in enumerate(CONDITIONS):
    z=score(logits[vi,ci].astype(float),y);gain=(base['pred']!=y)&(z['pred']==y);loss=(base['pred']==y)&(z['pred']!=y)
    detail['views'][view][cond]={'correct':z['correct'],'ce':z['ce'],'delta_ce':z['ce']-base['ce'],'mean_abs_logit_change':float(abs(logits[vi,ci]-logits[vi,0]).mean()),'mean_abs_true_margin_change':float(abs(z['margins']-base['margins']).mean()),'prediction_changed':int((z['pred']!=base['pred']).sum()),'gained':int(gain.sum()),'lost':int(loss.sum()),'net_correct':int(gain.sum()-loss.sum())}
  outputs.append(detail)
 write(U/'SENSITIVITY.json',{'scope':'TRAIN899, frozen eval only, no causal-training attribution','conditions':CONDITIONS,'runs':outputs,'logit_file_pins':{str(U/f'FROZEN_TRAIN_{s}.npz'):sha(U/f'FROZEN_TRAIN_{s}.npz') for s in SEEDS},'training':0,'new_DEV_inference':0})
def saved_analysis(ctx,stats):
 comp=read(SRC/'COMPARISON.json');out={'errors':{},'histories':{},'source_comparison_sha256':sha(SRC/'COMPARISON.json')};look={(r['clip_id'],r['view']):r for r in stats['perclip']}
 for seed in SEEDS:
  a=read(SRC/f'C0_{seed}_RESULT.json');b=read(SRC/f'C1_{seed}_RESULT.json')
  out['errors'][str(seed)]={}
  for view in VIEWS:
   aa={r['clip_id']:r for r in a['predictions'][view]};bb={r['clip_id']:r for r in b['predictions'][view]};q=[]
   for cid in aa:
    if aa[cid]['correct']==bb[cid]['correct']:continue
    q.append({'clip_id':cid,'change':'gain' if bb[cid]['correct'] else 'regression','label':look[cid,view]['label'],'groups':look[cid,view]['groups'],'C0_true_rank':1+sum(x>aa[cid]['logits'][aa[cid]['label_index']] for x in aa[cid]['logits']),'C1_true_rank':1+sum(x>bb[cid]['logits'][bb[cid]['label_index']] for x in bb[cid]['logits'])})
   out['errors'][str(seed)][view]=q
  out['histories'][str(seed)]={arm:{'selected_epoch':read(SRC/f'{arm}_{seed}_RESULT.json')['selected_epoch'],'selected_train_ce':read(SRC/f'{arm}_{seed}_history.json')[read(SRC/f'{arm}_{seed}_RESULT.json')['selected_epoch']-1]['train_ce'],'selected_dev_ce':read(SRC/f'{arm}_{seed}_RESULT.json')['selected_ce']} for arm in ['C0','C1']}
 write(U/'SAVED_DEV_ANALYSIS.json',out)
def main():
 ctx=read(U/'CONTEXT_v2.json');pincheck(ctx['pins']);assert not (U/'RUNNING.json').exists()
 write(U/'RUNNING.json',{'at':now(),'pid':os.getpid(),'scope':'diagnostics only'})
 try:
  arr={}
  for r in ctx['rows']:arr[r['clip_id']]={v:load(r['views'][v]) for v in VIEWS}
  stats=feature_stats(ctx,arr);saved_analysis(ctx,stats);frozen_inference(ctx,arr);pincheck(ctx['pins']);(U/'RUNNING.json').unlink();write(U/'STATUS.json',{'stage':'ANALYSIS_COMPLETE_PENDING_VERIFICATION','at':now(),'training':0},new=False)
 except BaseException:
  if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'at':now(),'traceback':traceback.format_exc(),'training':0})
  raise
if __name__=='__main__':main()
