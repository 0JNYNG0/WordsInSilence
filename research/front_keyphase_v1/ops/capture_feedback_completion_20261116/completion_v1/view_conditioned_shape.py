"""Restrict the new open-hand recovery to its reviewed upright-view domain.

The new examples support upright open hands. Foreshortened/folded flat views
remain on the legacy observer; no word identity or expected goal enters this.
"""
from evaluate_parts import RepairedShape,RepairedCoarse
from plane_model import *
import adapter

class ViewConditionedShape:
 def __init__(self):self.legacy=ShapeConsensus();self.repaired=RepairedCoarse();self.plane=PlaneObserver()
 def sequence(self,ch,hand,plane=None):
  x=ch['tcn280'];old=self.legacy.sequence(x,hand);b,_=self.legacy.expanded.sequence(x,hand);c,_=self.repaired.sequence(x,hand)
  proposed,_=combine(old['fingers'],c,b);p=self.plane.sequence(ch,hand)if plane is None else plane
  # Recovery only of the specifically source-expanded open upright view.
  eligible=(old['states']<0)&(proposed==0)&(p==0)&old['available'].all(1)
  old['legacy_states']=old['states'].copy();old['states']=np.where(eligible,0,old['states']);old['recovered']=eligible
  return old

def main():
 torch.set_num_threads(2);out=V/'view_conditioned';out.mkdir(exist_ok=True);assert not(out/'FROZEN.json').exists()
 write(out/'FROZEN.json',dict(scope='Unknown -> open recovery only in source-reviewed projected upright form; preserve all legacy known outputs',
  cause='Visible out-of-plane curl in a flat view was overcalled open. New handshape sources primarily establish upright open form; no flat-view recovery claim.',
  class_guard='plane==upright, recovered coarse==open, two coarse ensembles agree, individual known fingers cannot conflict, numeric observation valid',
  thresholds_changed=False,word_identity_input=False,
  acceptance='No extra wrong under corrected source DEV labels; preserve previously known outputs; recover public upright cases. Must next pass unseen public and motion/hand substitution controls.',
  model_sha256={str(p):sha(p)for folder in['shape_source_repair','plane']for p in sorted((V/folder).glob('*.pt'))}))
 d=np.load(S/'INPUTS.npz');prev=np.load(B/'local_shape/CONSENSUS_DEVELOPMENT.npz');net=ViewConditionedShape()
 b,_=net.legacy.expanded.windows(d['x'],d['availability']);c,_=net.repaired.windows(d['x'],d['availability']);proposal,_=combine(prev['fingers'],c,b)
 rows=read(S/'INPUTS.json')['rows'];outpred=prev['states'].copy();details=[];sourcecache={}
 extra={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']}
 for i in np.flatnonzero(prev['dev']&(outpred<0)&(proposal==0)):
  r=rows[i];cid=r['clip_id']
  if cid not in sourcecache:
   if cid in extra:path=extra[cid]['raw']
   else:path=read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')['raw_path']
   z=dict(np.load(path));sourcecache[cid]=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']))
  ch=sourcecache[cid];frame=int(np.flatnonzero(ch['frame_id']==r['frame'])[0]);p=net.plane.sequence(ch,r['side'])[frame]
  if p==0:outpred[i]=0
  details.append(dict(source=r,plane=int(p),proposal=int(proposal[i]),output=int(outpred[i])))
 y=prev['target'].copy();i=next(i for i,r in enumerate(rows)if r['clip_id']=='NIA_SL_WORD0071_REAL13_F'and r['frame']==82 and r['side']=='L');y[i]=2;dev=prev['dev']
 from shape_candidate import metrics
 summary={k:metrics(y,q,dev)for k,q in[('before',prev['states']),('after',outpred)]}
 rr=read(B/'local_shape/fresh/EXTRACTED.json')['rows'];truth=read(B/'local_shape/fresh/SOURCE_TRUTH.json')['rows'];cache={}
 for r in rr:
  z=dict(np.load(r['raw_path']));ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));cache[r['clip_id']]={h:net.sequence(ch,h)for h in['L','R']}
 known=[dict(**r,after=int(cache[r['clip_id']][r['hand']]['states'][r['frame']]),before=int(cache[r['clip_id']][r['hand']]['legacy_states'][r['frame']]))for r in truth]
 ty=np.array([r['state']for r in known]);mask=np.ones(len(ty),bool);diag={k:metrics(ty,np.array([r[k]for r in known]),mask)for k in['before','after']}
 gate=summary['after']['wrong']<=summary['before']['wrong']and diag['after']['wrong']<=diag['before']['wrong']
 write(out/'COMPARISON.json',dict(development=summary,diagnostic=diag,details=details,rows=known,component_gate=gate,web_admitted=False))
 print(json.dumps(dict(development=summary,diagnostic=diag,gate=gate,details=[dict(clip=r['source']['clip_id'],plane=r['plane'],out=r['output'])for r in details]),ensure_ascii=False))
if __name__=='__main__':main()
