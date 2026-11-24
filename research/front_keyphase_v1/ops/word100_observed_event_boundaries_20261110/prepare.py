from core import *
from collections import defaultdict,Counter
import ast
assert not (U/'FROZEN.json').exists()
ann=read(U/'OBSERVED_BOUNDARIES.json');ri=read(U/'REVIEW_INPUTS.json')['rows'];checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
fit_ann=defaultdict(list)
for a in ann['rows']:
 if a['role']=='fit':fit_ann[a['clip_id']].append(a)
 for e in a['events']:ck('ordered_boundary_ranges',e['onset'][1]<=e['offset'][0])
for r in ri:ck('review_RGB_integrity:'+r['clip_id'],sha(r['source'])==r['source_sha256'] and sha(r['sheet'])==r['sheet_sha256'])
rows=old.rows();fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development'];ck('100words899vs300',len(fit)==899 and len(dev)==300 and len({r['class_index'] for r in fit})==100);ck('personsplit',not set(r['person'] for r in fit)&set(r['person'] for r in dev))
pos=np.zeros(3);neg=np.zeros(3);events=0;pose_disagreements=[]
for r in fit:
 z=np.load(PREV/'cache'/f"{r['clip_id']}.npz");y,w=targets(r,fit_ann.get(r['clip_id'],[]));payload={k:z[k] for k in z.files if k not in ['phase_y','phase_w','direct']};payload.update(boundary_y=y,boundary_w=w);ck('finitelabels:'+r['clip_id'],np.isfinite(y).all() and y.shape==w.shape and y.shape[0]==len(r['frame_id']))
 ck('sourceframeidentity:'+r['clip_id'],np.array_equal(r['frame_id'],np.arange(len(r['frame_id']))));dest=U/'cache'/f"{r['clip_id']}.npz";dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**payload)
 valid=payload['q_full'][:,:,-2:].max(-1)>.5;ww=w*valid[:,:,None];pos+=(y*ww).sum((0,1));neg+=((1-y)*ww).sum((0,1))
 for a in fit_ann.get(r['clip_id'],[]):
  for e in a['events']:
   events+=1;lo=e['onset'][1];hi=e['offset'][0];_,ok,info=prior.geometry(prior.physical(r['x_full']),r['timestamp_us'],lo,hi,a['axis'],a['moving'],a['anchor']);sign=1 if info.get('signed_change',0)>0 else -1
   if not ok or sign!=e['direction']-1:pose_disagreements.append(dict(clip_id=r['clip_id'],axis=a['axis'],range=[lo,hi],observed_direction=e['direction'],pose=info,eligible=ok))
ck('manual_train_coverage',len(fit_ann)==18 and events==29)
ck('zero_unreviewed_labels',all(r['role']=='fit' for r in ann['rows'] if r['clip_id'] in fit_ann))
# Behavioural decoder checks: missing end cannot imply completion; short evidence survives.
for probs,complete in [(np.array([[.9,.9,.1],[.9,.2,.1],[.9,.1,.1],[.9,.1,.9]]),1),(np.array([[.9,.9,.1],[.9,.1,.1],[.9,.1,.1],[.9,.1,.1]]),0),(np.array([[.9,.9,.1],[.9,.1,.9]]),0)]:
 ee,pp=decode(probs,np.ones(len(probs),bool));ck('no_forced_end',len(ee)==complete);ck('retain_partial',set(j for e in ee+pp for j in range(e['a'],e['b']+1))==set(range(len(probs))))
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'))
write(U/'LABEL_AUDIT.json',dict(fit_videos=18,fit_expressions=10,fit_events=29,fit_holds=17,dev_videos=4,dev_events=7,dev_holds=4,common_fit_videos=899,common_words=100,pos_cells=pos.tolist(),neg_cells=neg.tolist(),pose_disagreements=pose_disagreements,policy='RGB observation is source; coordinate disagreements recorded, not relabelled to model answers. Visibility mask excludes missing input from loss; reviewed event remains in evaluation denominator.',reviewer='Same Codex visual review, not independent expert.2frame contact sheets; uncertain boundary ranges preserved.'))
basecfg=read(PREV/'FROZEN.json');pins={str(p):sha(p) for p in [U/'OBSERVED_BOUNDARIES.json',U/'REVIEW_INPUTS.json',U/'core.py',PREV/'runs/42/model.pt',PREV/'FULLPATH_COMPLETE.json',PREV/'PARTIAL_COMPLETE.json']}
write(U/'FROZEN.json',dict(seeds=[42,43,44],primary_seed=42,epochs=12,steps=38,batch=24,extra_supervised=6,lr=.0003,head_lr=.001,word_loss=.2,pos_weight=np.clip(neg/np.maximum(pos,1),1,15).tolist(),decoder=dict(prob=.5,min_frames=4,active_fraction=.6,valid_fraction=.8),conditions=['normal','upper','slow','fast','reversed_main','missing_2','missing_long'],automatic_conditions=78,protections=basecfg['protections'],pins=pins,adoption='Normal/speed/crossing preserved, no missing/unfinished promotion; improved boundaries plus downstream gain needed for deployment.',no_private=True,no_default_change=True,model='Existing event geometry encoder continued;3sigmoid observed-activity/onset/offset outputs; original shape/direction frozen. No target word to boundary head.',evaluation='HistoricalDEV. New7RGBboundary events evaluated from4videos. Existing21movement11hold anchors retained as separate coarse metric.78saved complete-analysis conditions, no oldTCN/DTWrerun. All3seeds, no selection.'))
write(U/'PRETRAIN_CHECK.json',dict(passed=True,count=len(checks),checks=checks));write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',pid=None));event('prepared',fit=899,events=events,pose_conflicts=len(pose_disagreements),checks=len(checks))
