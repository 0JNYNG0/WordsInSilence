from setup import *
import sys,time,copy,importlib.util,traceback
import numpy as np,torch
sys.path.insert(0,str(D))
from consumer import FeedbackConsumer,quarantine,A,normalize,assess
from positive_evidence import refine,along_forearm,CONTRACTS
G=O/'word100_observation_guard_20261108'; J=O/'word100_rtmw_joint_learning_20261108'; AUD=O/'word100_shape_direction_audit_20261109'
spec=importlib.util.spec_from_file_location('previous_audit',AUD/'audit.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def infer(c,x,word):
 wi=c.base.f['feedback']['words'].index(word);norm=normalize(x,c.base.stats)
 with torch.inference_mode():
  b=c.base.feedback(torch.from_numpy(norm[None]),torch.tensor([len(x)]),torch.tensor([wi]));h=b['latent'];p=b['attributes'].sigmoid();valid=torch.ones((1,len(x)),dtype=torch.bool)
  ph,d0=c.head(h,p,torch.tensor([wi]),valid,torch.zeros_like(p));_,d1=c.head(h,p,torch.tensor([wi]),valid,torch.ones_like(p))
 return dict(phase=ph[0].softmax(-1).numpy(),different_if0=d0[0].sigmoid().numpy(),different_if1=d1[0].sigmoid().numpy(),attributes=p[0].numpy())

def assess_pair(x,ts,fid,word,p):
 before=assess(word,p['phase'],p['different_if0'],p['different_if1'],x,ts,fid)
 t=time.perf_counter();after=refine(before,p['attributes'],x,ts,fid,p['phase']);ms=(time.perf_counter()-t)*1000
 return before,after,ms

def fit():
 assert not (U/'FIT_REVIEW.json').exists();c=FeedbackConsumer();out=[]
 for r in read(D/'PREPARED.json')['rows']:
  if r['role']!='fit':continue
  z=np.load(r['path']);word=read(D/'PREPARED.json')['words'][r['word']]
  # Existing frozen base attributes reused; only frozen phase/difference head is run.
  h=torch.from_numpy(z['h_full'][None]);p=torch.from_numpy(z['p_full'][None]);wi=torch.tensor([r['word']]);valid=torch.ones((1,len(z['raw_x'])),dtype=torch.bool)
  with torch.inference_mode():ph,d0=c.head(h,p,wi,valid,torch.zeros_like(p));_,d1=c.head(h,p,wi,valid,torch.ones_like(p))
  pred=dict(phase=ph[0].softmax(-1).numpy(),different_if0=d0[0].sigmoid().numpy(),different_if1=d1[0].sigmoid().numpy(),attributes=p[0].numpy())
  b,a,ms=assess_pair(z['raw_x'],z['timestamp_us'],z['frame_id'],word,pred)
  out.append(dict(clip_id=r['clip_id'],cards=a['cards'],before=[x['status'] for x in b['cards']],milliseconds=ms))
 write(U/'FIT_REVIEW.json',dict(rows=out,training=False,threshold_search=False))
 print(json.dumps([{'clip':r['clip_id'],'before':r['before'],'after':[c['status'] for c in r['cards']],'motion':[(c['id'],c['motion_evidence']['state'],round(c['motion_evidence'].get('signed_change',0),3)) for c in r['cards'] if c['motion_evidence']]} for r in out],ensure_ascii=False))

def evaluate():
 assert not (U/'COMPARISON.json').exists();torch.set_num_threads(4);c=FeedbackConsumer();out=[]
 pinfile=U/'EVALUATION_PINS.json'
 if pinfile.exists():
  assert read(pinfile)[str(U/'positive_evidence.py')]==sha(U/'positive_evidence.py')
  write(U/'RESUME_PINS.json',{str(U/'compare.py'):sha(U/'compare.py')})
 else:write(pinfile,{str(p):sha(p) for p in [U/'positive_evidence.py',U/'compare.py',U/'FROZEN.json',U/'GUIDE_CONTRACTS.json']})
 donor=np.load(read(J/'records/NIA_SL_WORD0739_REAL01_F.json')['raw_path'])['xy_px'][68,112:133]
 def save(cid,kind,word,x,ts,fid,p,guard=None,reused=False):
  name=cid+'_'+kind;path=U/'predictions'/(name+'.npz');result=U/'decisions'/(name+'.json')
  completed=path.exists() and result.exists()
  if completed:
   saved=read(result);before,after=saved['before'],saved['after'];ms=saved.get('milliseconds')
  else:before,after,ms=assess_pair(x,ts,fid,word,p)
  if guard is not None and not completed:
   for e in [before,after]:
    for card in e['cards']:
     sp=card.get('query_indices');sides=[0 if s=='L' else 1 for s in card['hands']]
     if sp and guard[sp[0]:sp[1]+1,sides].any():card.update(status='unknown',display_status='uncertain',observation_quarantine_overlap=True)
  if not completed:
   path.parent.mkdir(exist_ok=True)
   np.savez_compressed(path,x=x,timestamp_us=ts,frame_id=fid,**{k:p[k] for k in ['phase','different_if0','different_if1','attributes']})
   result.parent.mkdir(exist_ok=True);write(result,dict(before=before,after=after,milliseconds=ms))
  out.append(dict(clip_id=cid,kind=kind,word=word,before=[q['status'] for q in before['cards']],after=[q['status'] for q in after['cards']],required=[q.get('required',True) for q in after['cards']],
   components=[q['difference_components'] for q in after['cards']],new_contradictions=sum(q['display_status']=='difference' for q in after['cards']),milliseconds=ms,prediction=str(path),sha256=sha(path),decision=str(result),reused_prediction=reused))
 def pred_for(cid,kind,x,word):
  saved=U/'predictions'/(cid+'_'+kind+'.npz')
  return dict(np.load(saved)) if saved.exists() else infer(c,x,word)
 # Full-view normal regression uses saved predictions, without any re-inference.
 for r in read(D/'PREPARED.json')['rows']:
  if r['role']!='development':continue
  z=np.load(r['path']);s=np.load(D/'runs/42/predictions'/(r['clip_id']+'.npz'));p={k:s[k] for k in ['phase','different_if0','different_if1']};p['attributes']=z['p_full']
  save(r['clip_id'],'full_normal',read(D/'PREPARED.json')['words'][r['word']],z['raw_x'],z['timestamp_us'],z['frame_id'],p,reused=True)
 rows=[r for r in read(G/'INPUT_PLAN.json')['rows'] if r['person'] in [11,12,13] and r['word_id'] in read(D/'PREPARED.json')['words']]
 for r in rows:
  z=np.load(r['cache'] or G/'crops'/(r['clip_id']+'.npz'));ts=z['timestamp_us'];fid=z['frame_id'];xy=z['xy_px'];sc=z['score'];word=r['word_id'];lo,hi=r['intervals'][0];ids=np.flatnonzero((ts>=lo*1e6)&(ts<=hi*1e6));dt=int(np.median(np.diff(ts)))
  guard=quarantine(xy,sc,ts,640,480);x=A.convert(guard['xy_px'],sc,ts,fid,640,480)['tcn280'];cached=AUD/(r['clip_id']+'_normal_current_guard.npz')
  p=dict(np.load(cached)) if cached.exists() else pred_for(r['clip_id'],'crop_normal',x,word)
  save(r['clip_id'],'crop_normal',word,x,ts,fid,p,guard['hand_quarantined'],cached.exists())
  for speed in [.75,1.25]:
   idx=np.rint(np.linspace(0,len(x)-1,round(len(x)/speed))).astype(int);xx=x[idx].copy();tt=np.arange(len(idx),dtype=np.int64)*dt;ff=np.arange(len(idx));xx[:,-1]=np.r_[0,np.diff(tt)/1e6]
   save(r['clip_id'],'speed'+str(speed),word,xx,tt,ff,pred_for(r['clip_id'],'speed'+str(speed),xx,word),guard['hand_quarantined'][idx])
  # Camera translation/scale is tested in coordinate space with consistent bounds.
  # The adapter's normalization should make the resulting channels equivalent.
  moved=guard['xy_px']*.7+np.array([30,20]);visible=np.isfinite(guard['xy_px']).all(2)&(guard['xy_px'][:,:,0]>=0)&(guard['xy_px'][:,:,0]<640)&(guard['xy_px'][:,:,1]>=0)&(guard['xy_px'][:,:,1]<480);moved[~visible]=np.nan
  ag=A.convert(moved,sc,ts,fid,640,480)
  err=float(np.max(np.abs(ag['tcn280']-x)))
  save(r['clip_id'],'translation_scale',word,ag['tcn280'],ts,fid,pred_for(r['clip_id'],'translation_scale',ag['tcn280'],word),guard['hand_quarantined']);out[-1]['max_channel_delta']=err
  for size in [2,12]:
   xx=x.copy();start=int(ids[len(ids)//2]);end=min(len(xx),start+size);xx[start:end,18:186]=0;xx[start:end,195:279]=0
   save(r['clip_id'],'gap'+str(size),word,xx,ts,fid,pred_for(r['clip_id'],'gap'+str(size),xx,word))
  if word=='WORD1133':
   for kind in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']:
    xx=xy.copy();ss=sc.copy()
    if kind!='main_reversed':
     for i in ids:xx[i,112:133]=old.aligned_fist(donor,xx[i,112:133])
    if kind!='right_closed_hand_main':xx[ids]=xx[ids[::-1]];ss[ids]=ss[ids[::-1]]
    g=quarantine(xx,ss,ts,640,480);ch=A.convert(g['xy_px'],ss,ts,fid,640,480)['tcn280'];p=dict(np.load(AUD/(r['clip_id']+'_'+kind+'.npz')))
    save(r['clip_id'],kind,word,ch,ts,fid,p,g['hand_quarantined'],True)
  print(r['clip_id'],flush=True)
 summary={}
 for kind in sorted({r['kind'] for r in out}):
  rr=[r for r in out if r['kind']==kind]
  summary[kind]=dict(clips=len(rr),before_supported=sum(s=='supported' for r in rr for s in r['before']),after_supported=sum(s=='supported' for r in rr for s in r['after']),unknown=sum(s=='unknown' for r in rr for s in r['after']),new_contradictions=sum(r['new_contradictions'] for r in rr))
 write(U/'COMPARISON.json',dict(rows=out,summary=summary,training=False,private_used=False,synthetic_not_real_error_accuracy=True));print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
 try:fit() if '--fit' in sys.argv else evaluate()
 except BaseException:
  (U/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
