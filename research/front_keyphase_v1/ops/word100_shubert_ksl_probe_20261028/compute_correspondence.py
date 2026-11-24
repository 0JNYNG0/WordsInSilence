"""New frozen native-input comparison. Reuses model features, no model call/fit."""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_shubert_feasibility_20261028';G=O/'word100_guide_video_feedback_20261028'
sys.path.insert(0,str(G))
from motion_observer import extract_features,read_observation
from compare_pipeline import local_cost,subset
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scalar(x):
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save(name,x):(U/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')
def cosine(a,b):
 na=np.linalg.norm(a,axis=-1);nb=np.linalg.norm(b,axis=-1)
 assert np.all(na>0) and np.all(nb>0)
 return np.clip(1-(a@b.T)/(na[:,None]*nb[None,:]),0,2)
def path_for(cost):
 n,m=cost.shape;d=np.full((n+1,m+1),np.inf);d[0,0]=0;prev=np.zeros((n,m),np.int8)
 for i in range(n):
  for j in range(m):
   vals=[d[i,j],d[i,j+1],d[i+1,j]];k=int(np.argmin(vals));d[i+1,j+1]=cost[i,j]+vals[k];prev[i,j]=k
 i,j=n-1,m-1;p=[]
 while i>=0 and j>=0:
  p.append([i,j]);k=prev[i,j];i-=k!=2;j-=k!=1
 p=p[::-1];assert p[0]==[0,0] and p[-1]==[n-1,m-1]
 assert {i for i,j in p}==set(range(n)) and {j for i,j in p}==set(range(m))
 mapping=[int(np.median([j for ii,j in p if ii==i])) for i in range(n)]
 return p,mapping,float(d[n,m]/len(p))
def main():
 if (U/'CORRESPONDENCE.json').exists():raise RuntimeError('Completed comparison exists')
 frozen=read(U/'FROZEN.json');samples=read(U/'SAMPLES.json')['samples']
 for p,h in frozen['source_pins'].items():assert sha(p)==h
 metrics={'frozen_before_distances':True,'no_labels_used_for_matching':True,
  'shape':['Per-hand per-feature agreement only where target consensus and other-word reference class support exist','Common assessable intersection across methods is primary; method-specific known and unknown denominators also retained','Both 0/1 support needed for balanced accuracy; other=(index0,open0) is not fist','Copied and missing observations always out of current-evidence support'],
  'stage':['Compare physical static and motion feature consensus at mapped nine anchors; not semantic stage truth','Report all anchors plus preexisting source-annotation active query interval subset','Same support intersection across methods primary; uncertainty remains in full denominator','Each reference separate; query clip and feature axis counts retained'],
  'no_accuracy_claim':'These are AI visual-draft agreement measurements, not human-validated KSL correctness or C/T3F recognition scores'}
 save('EVALUATION_METRICS.json',metrics)
 cache={};pins={};export=[]
 for r in frozen['rows']:
  alias=r['alias'];idx=np.array(r['anchors']);f=P/'features'/alias;i=P/'inputs'/alias
  done=read(f/'DONE.json')
  for name,h in done['output_pins'].items():assert sha(f/name)==h
  a=read_observation(i/'native_observation.npz');full=extract_features(a);g=subset(full,idx)
  prov=np.load(i/'provenance.npz');streams=np.load(f/'streams.npz');sh=np.load(f/'last_ffn.npy')
  assert len(sh)==len(full['x'])==r['frames']
  source_kind=prov['source_kind'][idx,:2];sid=[f"{r['blind_clip']}_{k+1:02d}" for k in range(9)]
  active=np.zeros(9,bool)
  for start,end in r['annotation_intervals']:active|=(g['timestamp_us']>=round(start*1e6))&(g['timestamp_us']<=round(end*1e6))
  cache[alias]={'row':r,'g':g,'sh':sh[idx],'DINO':{side:streams[name][idx] for side,name in [('L','left_hand'),('R','right_hand')]},'current':source_kind==1,'sid':sid,'active':active}
  pins[alias]={'native':sha(i/'native_observation.npz'),'provenance':sha(i/'provenance.npz'),'streams':sha(f/'streams.npz'),'last_ffn':sha(f/'last_ffn.npy')}
  export.extend([{'sample_id':sid[k],'alias':alias,'person':r['recorded_person_id'],'word_id':r['word_id'],'frame_id':int(g['frame_id'][k]),'timestamp_us':int(g['timestamp_us'][k]),'annotation_active':bool(active[k]),'L_current':bool(source_kind[k,0]==1),'R_current':bool(source_kind[k,1]==1)} for k in range(9)])
 refs=[z for z in cache.values() if z['row']['recorded_person_id'] in [1,2]];queries=[z for z in cache.values() if z['row']['recorded_person_id']==3]
 pairs=[];shape=[];start=time.perf_counter()
 for q in queries:
  for r in refs:
   if q['row']['word_id']!=r['row']['word_id']:continue
   for method,cost in [('geometry',local_cost(q['g'],r['g'])),('shubert',cosine(q['sh'],r['sh']))]:
    path,mapping,avg=path_for(cost)
    pairs.append({'query':q['row']['alias'],'reference':r['row']['alias'],'method':method,'path':path,'mapping':mapping,'query_sample_ids':q['sid'],'reference_sample_ids':r['sid'],'mean_path_cost':avg,'reference_scope_retained':True})
   pairs.append({'query':q['row']['alias'],'reference':r['row']['alias'],'method':'time_fraction','path':[[k,k] for k in range(9)],'mapping':list(range(9)),'query_sample_ids':q['sid'],'reference_sample_ids':r['sid'],'mean_path_cost':None,'reference_scope_retained':True})
  for side,slot in [('L',0),('R',1)]:
   cols=q['g']['groups'][side+'_finger_angle']
   candidates=[]
   for r in refs:
    if r['row']['word_id']==q['row']['word_id']:continue
    for k in range(9):
     if r['current'][k,slot] and r['g']['mask'][k,cols].all():candidates.append((r,k))
   for k in range(9):
    entry={'query_sample_id':q['sid'][k],'query':q['row']['alias'],'side':side,'candidate_ids':[r['sid'][j] for r,j in candidates],
     'query_current':bool(q['current'][k,slot]),'query_angles_valid':bool(q['g']['mask'][k,cols].all()),'matches':{}}
    if entry['query_current'] and entry['query_angles_valid'] and candidates:
     dist={'geometry':[float(np.abs(q['g']['x'][k,cols]-r['g']['x'][j,cols]).mean()) for r,j in candidates],
       'dino':[float(cosine(q['DINO'][side][k:k+1],r['DINO'][side][j:j+1])[0,0]) for r,j in candidates],
       'shubert':[float(cosine(q['sh'][k:k+1],r['sh'][j:j+1])[0,0]) for r,j in candidates]}
     for method,values in dist.items():
      best=int(np.argmin(values));r,j=candidates[best];entry['matches'][method]={'reference_sample_id':r['sid'][j],'distance':values[best]}
    shape.append(entry)
 output={'status':'COMPLETE_FIXED_FEATURE_MATCHING_NO_TRAINING','frozen_sha256':sha(U/'FROZEN.json'),'metrics_sha256':sha(U/'EVALUATION_METRICS.json'),'script_sha256':sha(__file__),
   'sample_metadata':export,'input_pins':pins,'stage_pairs':pairs,'shape_queries':shape,'compute_seconds':time.perf_counter()-start,'training_runs':0,'model_forward_calls':0,'private_access':False}
 save('CORRESPONDENCE.json',output);print('Computed',len(pairs),'stage method-pairs;',len(shape),'hand queries; no labels/fit')
if __name__=='__main__':main()
