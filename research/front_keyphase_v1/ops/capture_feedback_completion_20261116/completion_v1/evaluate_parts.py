from train_terminal_relation import *
from shape_candidate import metrics
sys.path.insert(0,str(V.parent))
from diagnose_terminal import sources as progress_sources,adapter
sys.path.insert(0,str(B/'acquisition_motion_followup'))
from motion_model import Observer as Motion
from check_motion_bridge import bridge

class RepairedCoarse(LocalShape):
 def __init__(self):
  self.kind='temporal';self.models=[]
  for seed in[42,43,44]:
   ck=torch.load(V/'shape_source_repair'/f'{seed}.pt',map_location='cpu',weights_only=False);m=HandHead('temporal');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))

class RepairedShape(ShapeConsensus):
 def __init__(self):super().__init__();self.repaired=RepairedCoarse()
 def sequence(self,x,hand):
  old=super().sequence(x,hand);b,_=self.expanded.sequence(x,hand);c,_=self.repaired.sequence(x,hand)
  new,_=combine(old['fingers'],c,b);new[~old['available'].all(-1)]=-1
  old['legacy_states']=old['states'].copy();old['states']=np.where(old['states']>=0,old['states'],new)
  old['recovered']=((old['legacy_states']<0)&(old['states']>=0));return old

def shape_check():
 dest=V/'shape_source_repair';d=np.load(S/'INPUTS.npz');prev=np.load(B/'local_shape/CONSENSUS_DEVELOPMENT.npz');net=RepairedShape()
 b,_=net.expanded.windows(d['x'],d['availability']);c,_=net.repaired.windows(d['x'],d['availability']);new,_=combine(prev['fingers'],c,b);new=np.where(prev['states']>=0,prev['states'],new)
 orig=prev['target'];corrected=orig.copy();rr=read(S/'INPUTS.json')['rows'];ix=next(i for i,r in enumerate(rr)if r['clip_id']=='NIA_SL_WORD0071_REAL13_F'and r['frame']==82 and r['side']=='L');corrected[ix]=2
 dev=prev['dev'];summary={label:{k:metrics(y,q,dev)for k,q in [('before',prev['states']),('after',new)]}for label,y in [('archived_truth',orig),('source_corrected_truth',corrected)]}
 truth=read(B/'local_shape/fresh/SOURCE_TRUTH.json')['rows'];rr={r['clip_id']:r for r in read(B/'local_shape/fresh/EXTRACTED.json')['rows']}
 obs={k:{h:net.sequence(np.load(r['channel_path'])['tcn280'],h)for h in['L','R']}for k,r in rr.items()};rows=[]
 for r in truth:
  z=obs[r['clip_id']][r['hand']];rows.append(dict(**r,before=int(z['legacy_states'][r['frame']]),after=int(z['states'][r['frame']])))
 y=np.array([r['state']for r in rows]);mask=np.ones(len(rows),bool);diagnostic={k:metrics(y,np.array([r[k]for r in rows]),mask)for k in['before','after']}
 sc=summary['source_corrected_truth'];gate=sc['after']['wrong']<=sc['before']['wrong']and sc['after']['correct']>=sc['before']['correct']and diagnostic['after']['wrong']<=diagnostic['before']['wrong']
 result=dict(development=summary,known_diagnostic=diagnostic,rows=rows,gate=gate,web_admitted=False)
 write(dest/'COMPARISON.json',result);print('SHAPE',json.dumps({k:v for k,v in result.items()if k!='rows'}),flush=True)

def relation_check():
 out=V/'terminal_relation';old=RelationObserver();new=TerminalRelation();ff=Features();motion=Motion('mlp32');truth=read(B/'source_corrected/SOURCE_TRUTH.json')['rows'];rows=[]
 for r in read(B/'INPUTS.json')['rows']:
  if r['role']!='development':continue
  ch=dict(np.load(r['channels']));f,q,_=ff(ch);a,_=old(f,q,ch['timestamp_us']);b,_=new(f,q,ch['timestamp_us'])
  for t in truth:
   if t['clip_id']==r['clip_id']:rows.append(dict(**t,before=int(a[t['frame']]),after=int(b[t['frame']])))
 y=np.array([r['state']for r in rows]);mask=np.ones(len(y),bool);summary={k:metrics(y,np.array([r[k]for r in rows]),mask)for k in['before','after']}
 controls=[]
 for r in progress_sources():
  raw=dict(np.load(r['raw_path']));n=len(raw['frame_id']);end={14:54,15:58,16:51,17:90,18:60}[r['person']]
  for kind in['normal','first_only','constant','missing','fast','slow','scaled']:
   d={k:v.copy()for k,v in raw.items()}
   if kind=='first_only':
    for k in['xy_px','score','timestamp_us','frame_id']:d[k]=d[k][:end+1]
   elif kind=='constant':d['xy_px']=np.repeat(d['xy_px'][end:end+1],n,0)
   elif kind=='missing':d['xy_px'][:,91:]=np.nan
   elif kind in['fast','slow']:d['timestamp_us']=(d['timestamp_us']*(.65 if kind=='fast'else 1.5)).astype(np.int64)
   elif kind=='scaled':d['xy_px']=d['xy_px']*.75+np.array([90,40],np.float32)
   ch=adapter.convert(d['xy_px'],d['score'],d['timestamp_us'],d['frame_id'],int(d['source_width']),int(d['source_height']));f,q,_=ff(ch);a,_=old(f,q,ch['timestamp_us']);b,_=new(f,q,ch['timestamp_us']);m=motion.sequence(ch)
   before=bridge(a,m);after=bridge(b,m);expected=0 if kind in['constant','missing']else 1 if kind=='first_only'else 2
   controls.append(dict(person=r['person'],kind=kind,before=before,after=after,expected=expected,passed=len(after)==expected))
 gate=summary['after']['wrong']<=summary['before']['wrong']and all(r['passed']for r in controls)
 write(out/'COMPARISON.json',dict(development=summary,controls=controls,gate=gate,web_admitted=False,states=rows))
 print('RELATION',json.dumps(dict(development=summary,passed=sum(r['passed']for r in controls),total=len(controls),gate=gate)),flush=True)
 print('FAILURES',json.dumps([r for r in controls if not r['passed']]),flush=True)

if __name__=='__main__':
 torch.set_num_threads(2);shape_check();relation_check()
