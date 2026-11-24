"""Separate saved-evidence checks; does not train or select a candidate."""
from common import *
import zlib
checks=[]
def check(name,condition,detail=None):
 checks.append(dict(name=name,pass_=bool(condition),detail=detail))
def pins(name,values):
 for p,h in values.items():check(name+':'+p,Path(p).is_file() and sha(p)==h)
pins('protected',read(U/'PROTECTED.json'))
for name in ['TRAIN_CONFIG.json','WHOLE_CENTER_CONFIG.json','CANDIDATE_FROZEN.json']:
 pins(name,read(U/name)['pins'])
sources=read(U/'READY_SOURCES.json')['rows'];labels=read(U/'LABELS.json')['rows'];train=read(U/'TRAIN_INPUTS.json');data=np.load(U/'TRAIN_INPUTS.npz')
for r in sources:
 for v,s in r['views'].items():
  p=Path(s['path']);check('public_video:'+r['clip_id']+v,sha(p)==s['sha256'] and (zlib.crc32(p.read_bytes())&0xffffffff)==s['crc'])
 x,f=load_channels(r['channel_path']);check('channels:'+r['clip_id'],np.isfinite(x).all() and len(x)==r['views']['F']['frames'] and np.array_equal(f,np.arange(len(x))))
fit={r['clip_id']for r in train['state_rows']+train['phase_rows']}
diag={r['clip_id']for r in labels if r['role']=='diagnostic'};confirm={r['clip_id']for r in labels if r['role']=='confirmation'}
check('split_no_clip_overlap',not(fit&diag or fit&confirm or diag&confirm))
check('fit_people_only_01_09',all(1<=r['person']<=9 and r['role']=='fit' for r in train['state_rows']+train['phase_rows']))
for kind,rows in [('state',train['state_rows']),('phase',train['phase_rows'])]:
 check(kind+'_finite',np.isfinite(data[kind+'_x']).all())
 check(kind+'_mask_excludes_unknown',not(data[kind+'_mask']&(data[kind+'_y']<0)).any())
 for i,r in enumerate(rows):
  if 'quality' in r:check(f'{kind}_source_quality_{i}',not(data[kind+'_mask'][i]&(np.array(r['quality'])!=1)).any())

# Recompute every saved metric independently from per-frame records.
for p in sorted(U.glob('*RESULTS.json')):
 if p.name=='REPLAY_RESULTS.json':continue
 d=read(p);rows=d['rows'];y=np.array([r['shape']for r in rows]);a=np.array([r['predictions']['available']for r in rows]);q=np.array([r['quality']for r in rows]);tp=np.array([r['phase']for r in rows]);mask=(y>=0)&a;b=np.array([r['predictions']['base']for r in rows]);oldok=mask&(b==y);oldwrong=mask&(b>=0)&(b!=y);change=(tp==1)&(y==1)&a
 for mode,m in d['metrics'].items():
  pred=np.array([r['predictions'][mode]for r in rows]);ok=mask&(pred==y);wrong=mask&(pred>=0)&(pred!=y);unknown=mask&(pred<0)
  calc=dict(known=int(mask.sum()),correct=int(ok.sum()),wrong=int(wrong.sum()),unknown=int(unknown.sum()),lost_previous_correct=int((oldok&~ok).sum()),new_correct=int((ok&~oldok).sum()),new_confident_errors=int((wrong&~oldwrong).sum()),extended_during_change_support=int(change.sum()),extended_during_change_correct=int((change&(pred==1)).sum()),wrong_on_bad_chain=int((wrong&(q==0)).sum()))
  check(p.name+':'+mode,calc==m,calc)
 pp=np.array([r['predictions']['phase']for r in rows]);pm=(tp>=0)&a
 calc=dict(known=int(pm.sum()),correct=int((pm&(pp==tp)).sum()),wrong=int((pm&(pp>=0)&(pp!=tp)).sum()),unknown=int((pm&(pp<0)).sum()),changes=int(((tp==1)&a).sum()),changes_found=int(((tp==1)&(pp==1)&a).sum()),stable_false_change=int(((tp==0)&(pp==1)&a).sum()))
 check(p.name+':phase',calc==d['phase_metrics'])
replay=read(U/'REPLAY_RESULTS.json');rows=replay['rows']
for mode,s in replay['summary'].items():
 calc=dict(cases=len(rows),normal_cases=sum(r['normal']for r in rows),normal_new_warnings=sum(r['modes'][mode]['normal_warning']for r in rows),old_details=sum(r['modes'][mode]['old_details']for r in rows),new_details=sum(r['modes'][mode]['details']for r in rows),retained_old_details=sum(r['modes'][mode]['retained_old_details']for r in rows))
 check('replay_count:'+mode,calc==s)
 check('replay_preserved:'+mode,s['normal_new_warnings']==0 and s['old_details']==s['new_details']==s['retained_old_details']==20 and all(r['coarse_direction_order_unchanged']for r in rows))

from runtime_v2 import *
torch.set_num_threads(2)
for kind,oldfolder in [('state','shared_finger_chain'),('whole_center','expanded_whole_hand')]:
 for seed in [42,43,44]:
  old=torch.load(P/oldfolder/f'{seed}.pt',map_location='cpu',weights_only=False)['state_dict'];new=torch.load(U/'models'/kind/f'{seed}.pt',map_location='cpu',weights_only=False)['state_dict']
  check(f'weights_changed_{kind}_{seed}',any(not torch.equal(old[k],new[k])for k in old))
# Rigid 2D transforms/scale preserve relative geometry; not a 3D claim.
x,_=load_channels(sources[0]['channel_path']);x=x[35:60].copy();g,a=features(x,'R');z=x.copy();pts=z[:,:102].reshape(-1,51,2);rot=np.array([[.8,-.6],[.6,.8]],np.float32);pts[:]=pts@rot*1.7+np.array([.3,-.2]);gg,aa=features(z,'R')
check('relative_2d_geometry_invariance',np.array_equal(a,aa) and np.allclose(g,gg,atol=2e-4),float(np.abs(g-gg).max()))
model=DualCenterObserver();zz=x.copy();zz[:,186:237]=0;none=model.all(zz,'R')
check('missing_points_not_repaired_as_truth',all((none[k]<0).all()for k in ['center_only','whole_center','dual_center_only','phase','dual_change_fill']))
original=model.all(x,'R');z=x.copy();z[:10]=0;z[11:]=0;changed=model.all(z,'R')
check('current_frame_branch_excludes_neighbor_context',all(np.allclose(original['probabilities'][k][:,10],changed['probabilities'][k][:,10],atol=1e-7)for k in ['state','whole_center']))
out=dict(status='PASS'if all(r['pass_']for r in checks)else'FAIL',checks=len(checks),passed=sum(r['pass_']for r in checks),protected_count=len(read(U/'PROTECTED.json')),failures=[r for r in checks if not r['pass_']],rows=checks,scope='Same Codex separate recomputation and contract verification; not independent expert review; PASS is evidence integrity, not candidate adoption.')
write(U/'VERIFIED.json',out);print({k:v for k,v in out.items()if k!='rows'})
