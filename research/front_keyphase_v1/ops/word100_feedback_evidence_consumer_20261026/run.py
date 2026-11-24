from pathlib import Path
import sys,json,hashlib,datetime,traceback,time
import numpy as np
from collections import Counter
import consumer as C
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_broad_phase_alignment_20261026';Q=O/'word100_quality_preserving_phase_20261026';BP=O/'word100_broad_phase_preflight_20261026'
SCORES={'C':O/'word100_body_boundary_20261017/training_C_20261013_RESULT.json','T3F':O/'word100_frozen_c_temporal_20261019/T3F_20261013_RESULT.json'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=None,allow_nan=False)
def check(pins):
 for p,h in pins.items():assert sha(p)==h,p
def freeze():
 f=read(Q/'FROZEN.json');r=read(Q/'RESULT.json');v=read(Q/'VERIFIED.json');b=read(BP/'PREPARED.json');assert v['gate']['passed'];check(f['pins']);check(r['pins']);check(f['protected'])
 pins={**r['pins'],**f['pins'],**b['output_pins'],**{str(p):sha(p) for p in [Path(__file__),U/'consumer.py',U/'preflight.py',U/'PREFLIGHT.json',Q/'VERIFIED.json',BP/'PREPARED.json',*SCORES.values()]}}
 write(U/'FROZEN.json',dict(at=now(),pins=pins,protected=f['protected'],rows=r['rows'],input_rows=b['rows'],scores={k:str(p) for k,p in SCORES.items()},
  scope='Research-only consumer of existing4500public100word DEV comparisons. Original300 identity cases join saved C13/T3F13 masked ranks without reranking/reinference. Synthetic transformed cases recognition=null. No E3/known200/fresh199 files accessed, no model/runtime edit.',
  schema='Separate original instantaneous difference,3level sequence difference,2change difference,coordinate comparability and corroborated quality;perhand/perangle intervals on reference clock,source1refined/2retained. Differences only over observed cells; empty=null,notzero. Fixed audit grid not mandatory hand semantics. No correct/wrong gesture verdict.',
  gate='Every saved numeric array,denominator,interval,source andrank must match. All600wholehand missing controls keep missinghandshape/palm/handwrist difference null but arm observation unaffected. Uncertainangle cannot change neighbor. Missing/invalid input fails closed. ExistingC/T3F/protected pins unchanged.',
  verification='Separate consumer receipt arithmetic/interval reconstruction all4500; meaningful malformed/empty/partial/positiononly/neighbor fixtures; sameCodex arithmetic,not independent human truth.',
  training=0,detector=0,new_alignment=0,model_inference=0,operating_changed=False))
 print('FROZEN_FEEDBACK4500',flush=True)
def prediction_rows(f):
 out={}
 for key,p in f['scores'].items():
  rr=read(p)['predictions']['masked'];assert len(rr)==300
  out[key]={r['clip_id']:{k:r[k] for k in ('status','prediction','top5','correct','top5_correct')} for r in rr}
 return out
def run():
 f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);(U/'receipts').mkdir();write(U/'RUNNING.json',dict(at=now()));sources={x['clip_id']:x for x in f['input_rows']};scores=prediction_rows(f);index=[];pins={};count=Counter();start=time.perf_counter()
 for i,row in enumerate(f['rows']):
  cid=row['clip_id'];z=load(B/'cases'/(row['name']+'.npz'));q=load(Q/'cases'/(row['name']+'.npz'));orig=load(sources[cid]['output']);rec={k:v[cid] for k,v in scores.items()} if row['kind']=='identity' else None
  result=C.build(domain=z['domain'],point_ok=z['point_ok'],point_delta=z['point_delta'],sequence_ok=z['sequence_new_ok'],sequence_delta=q['sequence_delta'],sequence_quality=q['sequence_quality'],source=q['source'],reference_missing=z['missing_original'],reference_times_us=orig['timestamp_us'],reference_id=cid,query_id=row['name'],kind=row['kind'],recognition=rec)
  result['research_source']=dict(public_word_id=row['word_code'],display_label=sources[cid]['label'],signer_id=sources[cid]['signer_id'],split='development_only',reference_type='same_source_control',raw_case_sha256=sha(B/'cases'/(row['name']+'.npz')),quality_candidate_sha256=sha(Q/'cases'/(row['name']+'.npz')))
  p=U/'receipts'/(row['name']+'.json');write(p,result);pins[str(p)]=sha(p)
  summary=[dict(side=h['side'],groups=[{k:g[k] for k in ('group','audit_cells','point_compared_cells','sequence_compared_cells','corroborated_sequence_cells','evidence_state','point_mean_difference','sequence_level_mean_difference','sequence_change_mean_difference')} for g in h['groups']]) for h in result['hands']]
  index.append(dict(name=row['name'],clip_id=cid,label=sources[cid]['label'],word_code=row['word_code'],kind=row['kind'],receipt=str(p),hands=summary,recognition=rec));count['cases']+=1;count['with_saved_recognition']+=rec is not None
  if (i+1)%500==0:print('RECEIPTS',i+1,4500,flush=True)
 assert len(index)==4500 and count['with_saved_recognition']==300;check(f['pins']);check(f['protected']);write(U/'RESULT.json',dict(at=now(),status='CONSUMER_READY_PENDING_SEPARATE_VERIFICATION',rows=index,pins=pins,counts=dict(count),seconds=time.perf_counter()-start));(U/'RUNNING.json').unlink();print('COMPLETE_FEEDBACK',dict(count),flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=U/'FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
