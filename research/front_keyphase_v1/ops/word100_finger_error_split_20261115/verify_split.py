from split_model import *
from split_runtime import SplitObserver
import collections,time
checks=[]
def ck(name,condition):
 assert condition,name;checks.append(name)
for p,h in read(U/'PROTECTED.json').items():ck('protected '+p,sha(p)==h)
for p,h in read(U/'CANDIDATE_FROZEN.json')['pins'].items():ck('frozen '+p,sha(p)==h)
labels=read(U/'LABELS.json')['rows'];paths={}
for r in labels:
 ck('public '+r['clip_id'],r['clip_id'].startswith('NIA_SL_WORD'))
 for k,p in [('source_sha256','path'),('raw_sha256','raw_path'),('channel_sha256','channel_path')]:paths[r[p]]=r[k]
for p,h in paths.items():ck('source '+p,sha(p)==h)
meta=read(U/'TRAIN_INPUTS.json');data=np.load(U/'TRAIN_INPUTS.npz')
for name in ['shape_rows','quality_rows']:ck(name+' FIT identities',all(1<=r['person']<=9 and r['role']=='fit'for r in meta[name]))
ck('three disjoint person roles',{r['person']for r in labels if r['role']=='fit'}.isdisjoint({10,16}))
for k in data.files:ck(k+' finite',np.isfinite(data[k]).all())
ck('unknown shape no gradient',not np.any(data['shape_mask']&(data['shape_y']<0)))
ck('unknown quality no gradient',not np.any(data['quality_mask']&(data['quality_y']<0)))
newmap={(r['clip_id'],r['hand'],r['frame']):r for r in labels if r['role']=='fit'}
for i,r in enumerate(meta['shape_rows']):
 nr=newmap.get((r['clip_id'],r['side'],r['frame']))
 if nr:ck(f'bad new chain mask {i}',not np.any(data['shape_mask'][i]&(np.array(nr['quality'])!=1)))
for role in ['diagnostic','historic','confirmation_reserved']:
 d=read(U/f'{role}_RESULTS.json');rr=d['rows'];y=np.array([r['shape']for r in rr]);a=np.array([r['predictions']['available']for r in rr]);m=(y>=0)&a
 for mode in ['base','shape_only','compatible_shape','quality_only','combined']:
  p=np.array([r['predictions'][mode]for r in rr]);v=d['metrics'][mode]
  ck(role+mode+' counts',v['correct']==int(((p==y)&m).sum())and v['wrong']==int(((p>=0)&(p!=y)&m).sum())and v['unknown']==int(((p<0)&m).sum()))
  ck(role+mode+' total',sum(v[k]for k in ['correct','wrong','unknown'])==v['known'])
 if role!='historic':ck(role+' identities',all(r['person']==(10 if role=='diagnostic' else 16)for r in rr))
 else:
  old=np.load(P/'SUPPLEMENT_PREDICTIONS.npz')['pred'];ck('admitted historic reproduced',all(np.array_equal(r['predictions']['base'],old[r['original_index']])for r in rr))
# Physical normalization tests do not use labels or tune thresholds.
r=next(r for r in labels if r['role']=='diagnostic');x,_=load_channels(r['channel_path']);f,a=shape_features(x,r['hand']);q,_=quality_features(x,r['hand']);inv=[]
for mode in ['scale_translation','in_plane_rotation','reflection']:
 xx=x.copy();pts=xx[:,:102].reshape(-1,51,2)
 if mode=='scale_translation':pts[:]=pts*1.7+np.array([.3,-.2])
 elif mode=='reflection':pts[:,:,0]*=-1
 else:
  angle=.37;mat=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);pts[:]=pts@mat.T
 ff,aa=shape_features(xx,r['hand']);qq,_=quality_features(xx,r['hand']);delta=max(float(np.max(abs(ff-f))),float(np.max(abs(qq-q))));ck(mode+' feature invariance',delta<.002 and np.array_equal(a,aa));inv.append(dict(transform=mode,max_difference=delta))
replay=read(U/'REPLAY_RESULTS.json')['summary']
for mode,s in replay.items():ck(mode+' replay preservation',s['cases']==122 and s['normal_cases']==64 and s['normal_new_warnings']==0 and s['old_details']==s['retained_old_details']==20)
for run in read(U/'TRAINED.json')['runs']:
 ck('trained pin '+str(run['seed'])+run['kind'],sha(run['path'])==run['sha256'])
 if run['kind']=='shape':
  a=torch.load(run['path'],map_location='cpu',weights_only=False);b=torch.load(P/'shared_finger_chain'/f"{run['seed']}.pt",map_location='cpu',weights_only=False);ck('actual fine tuning '+str(run['seed']),any(not torch.equal(v,b['state_dict'][k])for k,v in a['state_dict'].items()))
stats=dict(anchors=len(meta['shape_rows']),clips=len({r['clip_id']for r in meta['shape_rows']}),words=len({r['word']for r in meta['shape_rows']}),people=sorted({r['person']for r in meta['shape_rows']}))
write(U/'VERIFIED.json',dict(pass_count=len(checks),checks=checks,training_scope=stats,invariance=inv,implementation='Codex',verification='Codex separate deterministic recount, hash and invariance checks',independent_expert=False,user_material_used=False,web_changed=False,quality_reserved_positive_support=0))
print(json.dumps(dict(checks=len(checks),training_scope=stats,invariance=inv)))
