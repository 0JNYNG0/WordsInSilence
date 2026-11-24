from repair import *
from projection_guard import CombinedVisible,ProjectionHead
from train_shape import S

def summary(q):return dict(events=len(q['events']),criteria={c['id']:c['status']for c in q['criteria']},seconds=q['seconds'])
def main():
 torch.set_num_threads(2);out=HERE/'validation';out.mkdir(exist_ok=True);assert not(out/'RESULT.json').exists()
 old=SparseVisible();new=CombinedVisible();normal=[];controls=[]
 rows=read(HERE/'PUBLIC_INPUTS.json')['rows']+read(HERE/'public_camera/INPUTS.json')['rows']+read(HERE/'public_views/INPUTS.json')['rows']
 hold=read(V/'web/SOURCES.json')['normal'];print('HOLD_KEYS',list(hold),flush=True)
 # Known previous held-person sample is regression now, not a new holdout.
 p=V/'holdout/EXTRACTED.json'
 if p.exists():print('HOLD_EXTRACT',read(p),flush=True)
 for r in rows:
  raw=dict(np.load(r['raw']));a=old.analyze(raw);b=new.analyze(raw);normal.append(dict(id=r['id'],role=r['role'],before=summary(a),after=summary(b)))
  write(out/(r['id']+'.json'),dict(before=a,after=b))
  if r['id'].endswith('_F')and r['person']in[13,14,15,16,17,18]:
   end=b['events'][0]['end']+1 if b['events']else len(raw['frame_id'])//2
   for kind in['first_only','constant','missing','fast','slow','scaled','one_frame_gap','long_gap','second_before_arrival']:
    z={k:v.copy()for k,v in raw.items()};n=len(z['frame_id'])
    if kind in['first_only','second_before_arrival']:
     cut=end if kind=='first_only'else (b['events'][1]['arrival_index']-1 if len(b['events'])>1 else end)
     for k in['xy_px','score','timestamp_us','frame_id']:z[k]=z[k][:cut]
    elif kind=='constant':z['xy_px']=np.repeat(z['xy_px'][end:end+1],n,0)
    elif kind=='missing':z['xy_px'][:,91:]=np.nan
    elif kind in['fast','slow']:z['timestamp_us']=(z['timestamp_us']*(.65 if kind=='fast'else 1.5)).astype(np.int64)
    elif kind=='scaled':z['xy_px']=z['xy_px']*.75+np.array([90,40],np.float32)
    elif kind=='one_frame_gap':z['xy_px'][(b['events'][0]['start']+b['events'][0]['arrival_index'])//2,112:133]=np.nan
    elif kind=='long_gap':z['xy_px'][b['events'][0]['start']:b['events'][0]['arrival_index']+1,112:133]=np.nan
    aa=old.analyze(z);bb=new.analyze(z);count=len(bb['events']);expected='no_completion'if kind in['first_only','constant','missing','long_gap','second_before_arrival']else'no_regression'
    if expected=='no_completion':check=count<2
    else:check=all(c['status']!='difference'for c in bb['criteria'])and count>=len(aa['events'])
    controls.append(dict(id=r['id'],kind=kind,expected=expected,check=check,before=summary(aa),after=summary(bb)))
  print(r['id'],summary(a)['events'],summary(b)['events'],summary(b)['criteria'],flush=True)
 dev=np.load(B/'local_shape/CONSENSUS_DEVELOPMENT.npz');d=np.load(S/'INPUTS.npz');coarse,_=ProjectionHead().windows(d['x'],d['availability']);before=dev['states'];after=before.copy();after[(before==1)&(coarse==0)]=-1;mask=dev['dev'];target=dev['target']
 shape=dict(development_total=int(mask.sum()),downgraded=int(((before!=after)&mask).sum()),lost_correct=int(((before==target)&(after!=target)&mask).sum()),wrong_before=int(((before>=0)&(before!=target)&mask).sum()),wrong_after=int(((after>=0)&(after!=target)&mask).sum()))
 write(out/'RESULT.json',dict(normal=normal,controls=controls,shape=shape,controls_passed=sum(r['check']for r in controls),controls_total=len(controls),private_data=False))
 print('FINAL',shape,'controls',sum(r['check']for r in controls),len(controls),flush=True)
if __name__=='__main__':main()
