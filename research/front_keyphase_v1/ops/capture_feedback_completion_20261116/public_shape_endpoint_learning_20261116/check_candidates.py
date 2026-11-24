from form_candidate import *
from train_relation import RelationCandidate
from train_shape import S

class JointLesson(FormLesson):
 def __init__(self):super().__init__();self.relation=RelationCandidate()

def summary(q):return dict(events=len(q['events']),criteria={c['id']:c['status']for c in q['criteria']})
def main():
 torch.set_num_threads(2);old=ThanksCandidate();form=FormLesson();joint=JointLesson();dest=D/'comparison';dest.mkdir(exist_ok=True)
 rows=read(H/'PUBLIC_INPUTS.json')['rows']+read(H/'public_camera/INPUTS.json')['rows']+read(H/'public_views/INPUTS.json')['rows']+read(H/'confirmation_views/INPUTS.json')['rows'];results=[];controls=[]
 for r in rows:
  assert 'Temp'not in r['raw']and 'user_'not in r['raw'];z=dict(np.load(r['raw']))
  saved=dest/f'{r["id"]}.json'
  if saved.exists():
   q=read(saved);a,b,c=q['before'],q['form'],q['joint']
  else:
   a=old.analyze(z);b=form.analyze(z);c=joint.analyze(z);write(saved,dict(before=a,form=b,joint=c))
  row=dict(id=r['id'],role=r['role'],before=summary(a),form=summary(b),joint=summary(c));results.append(row)
  if row['before']!=row['form']or row['before']!=row['joint']:print(row,flush=True)
  if r['id'].endswith('_F')and r['person']in[13,14,15,16,17,18]and len(a['events'])>=2:
   n=len(z['frame_id']);first=a['events'][0]['arrival_index'];second=a['events'][1]['arrival_index']
   for kind in ['first_only','constant','missing','fast','slow','scaled','second_before_arrival','long_gap']:
    d={k:v.copy()for k,v in z.items()}
    if kind in ['first_only','second_before_arrival']:
     end=first+1 if kind=='first_only'else second-1
     for key in ['xy_px','score','timestamp_us','frame_id']:d[key]=d[key][:end]
    elif kind=='constant':d['xy_px']=np.repeat(d['xy_px'][first:first+1],n,axis=0)
    elif kind=='missing':d['xy_px'][:,91:]=np.nan
    elif kind in ['fast','slow']:d['timestamp_us']=(d['timestamp_us']*(.65 if kind=='fast'else 1.5)).astype(np.int64)
    elif kind=='scaled':d['xy_px']=d['xy_px']*.75+np.array([90,40],np.float32)
    elif kind=='long_gap':d['xy_px'][a['events'][0]['start']:first+1,112:]=np.nan
    aa=old.analyze(d);bb=form.analyze(d);cc=joint.analyze(d)
    expected='not_completed'if kind in ['first_only','constant','missing','second_before_arrival','long_gap']else'normal_preserved'
    cs=summary(cc);passed=cs['events']<2 if expected=='not_completed'else cs['events']>=len(aa['events'])and all(v!='difference'or summary(aa)['criteria'][k]=='difference'for k,v in cs['criteria'].items())
    controls.append(dict(id=r['id'],kind=kind,expected=expected,before=summary(aa),form=summary(bb),joint=cs,passed=passed))
 write(dest/'CONTROLS.json',controls)
 truth=read(B/'source_corrected/SOURCE_TRUTH.json')['rows'];by={r.get('clip_id',r['id']):r for r in rows};states=[]
 for cid in {r['clip_id']for r in truth if r['split']=='development'}:
  z=dict(np.load(by[cid]['raw']));ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));f,q,_=old.feature(ch);a,_=old.relation(f,q,ch['timestamp_us']);b,_=joint.relation(f,q,ch['timestamp_us'])
  for t in truth:
   if t['clip_id']==cid:states.append(dict(**t,before=int(a[t['frame']]),after=int(b[t['frame']])))
 meta=read(S/'INPUTS.json')['rows'];dev=np.load(B/'local_shape/CONSENSUS_DEVELOPMENT.npz');target=dev['target'].copy();target[next(i for i,r in enumerate(meta)if r['clip_id']=='NIA_SL_WORD0071_REAL13_F'and r['frame']==82 and r['side']=='L')]=2
 extras={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']};cache={};anchors=[]
 for i in np.flatnonzero(dev['dev']):
  r=meta[i];key=r['clip_id'],r['side'];cid=r['clip_id']
  if key not in cache:
   path=extras[cid]['raw']if cid in extras else read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')['raw_path'];raw=dict(np.load(path));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
   before=old.shape.sequence(ch,r['side'])['states'];form.shape.sequence(ch,r['side']);after,_,_=form.form_states(ch,r['side']);cache[key]=(ch['frame_id'],before,after)
  ids,a,b=cache[key];j=int(np.flatnonzero(ids==r['frame'])[0]);anchors.append(dict(clip=cid,side=r['side'],frame=r['frame'],truth=int(target[i]),before=int(a[j]),after=int(b[j])))
 result=dict(normal=results,controls=controls,relation_anchors=states,shape_anchors=anchors,
  relation_lost_correct=sum(x['before']==x['state']and x['after']!=x['state']for x in states),relation_new_wrong=sum(x['after']>=0 and x['after']!=x['state']and (x['before']<0 or x['before']==x['state'])for x in states),
  shape_lost_correct=sum(x['before']==x['truth']and x['after']!=x['truth']for x in anchors),shape_new_wrong=sum(x['after']>=0 and x['after']!=x['truth']and (x['before']<0 or x['before']==x['truth'])for x in anchors),private_data=False)
 write(dest/'RESULT.json',result);print('SUMMARY',{k:v for k,v in result.items()if k not in['normal','controls','relation_anchors','shape_anchors']},'control_failed',sum(not x['passed']for x in controls),flush=True)
if __name__=='__main__':main()
