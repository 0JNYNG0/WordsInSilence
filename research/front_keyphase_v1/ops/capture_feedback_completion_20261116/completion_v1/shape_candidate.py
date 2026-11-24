from train_shape import *
class ContinuedShape(LocalShape):
 def __init__(self):
  self.kind='temporal';self.models=[]
  for seed in [42,43,44]:
   ck=torch.load(V/'shape_continuation'/f'{seed}.pt',map_location='cpu',weights_only=False)
   m=HandHead('temporal');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))
class CandidateShape(ShapeConsensus):
 def __init__(self):
  super().__init__();self.before=ContinuedShape()

def metrics(y,q,mask):return dict(total=int(mask.sum()),correct=int(((q==y)&mask).sum()),wrong=int(((q>=0)&(q!=y)&mask).sum()),unknown=int(((q<0)&mask).sum()))
def main():
 torch.set_num_threads(2);dest=V/'shape_continuation';assert not(dest/'COMPARISON.json').exists()
 d=np.load(S/'INPUTS.npz');baseline=np.load(B/'local_shape/CONSENSUS_DEVELOPMENT.npz');q=np.load(dest/'predictions.npz')['states'][:len(d['x'])]
 old=ShapeConsensus();new=CandidateShape();expanded,_=old.expanded.windows(d['x'],d['availability']);new_s,_=combine(baseline['fingers'],q,expanded)
 y=baseline['target'];dev=baseline['dev'];summary=dict(before=metrics(y,baseline['states'],dev),after=metrics(y,new_s,dev),
  correct_to_wrong=int((dev&(baseline['states']==y)&(new_s>=0)&(new_s!=y)).sum()),correct_to_unknown=int((dev&(baseline['states']==y)&(new_s<0)).sum()))
 truth=read(B/'local_shape/fresh/SOURCE_TRUTH.json')['rows'];sources={r['clip_id']:r for r in read(B/'local_shape/fresh/EXTRACTED.json')['rows']}
 obs={k:{h:(old.sequence(np.load(r['channel_path'])['tcn280'],h),new.sequence(np.load(r['channel_path'])['tcn280'],h))for h in['L','R']}for k,r in sources.items()}
 rows=[]
 for r in truth:
  a,b=obs[r['clip_id']][r['hand']];i=r['frame'];rows.append(dict(**r,before=int(a['states'][i]),after=int(b['states'][i])))
 arr_y=np.array([r['state']for r in rows]);mask=np.ones(len(rows),bool)
 fresh={k:metrics(arr_y,np.array([r[k]for r in rows]),mask)for k in['before','after']}
 # Every legacy DEV sample is retained, not only the new target word.
 changes=[dict(row=int(i),source=read(S/'INPUTS.json')['rows'][i],target=int(y[i]),before=int(baseline['states'][i]),after=int(new_s[i]))for i in np.flatnonzero(dev&(baseline['states']!=new_s))]
 gate=summary['correct_to_wrong']==0 and summary['correct_to_unknown']==0 and summary['after']['wrong']<=summary['before']['wrong'] and fresh['after']['wrong']<=fresh['before']['wrong']
 write(dest/'COMPARISON.json',dict(development=summary,known_separate_public=fresh,rows=rows,changes=changes,component_gate=gate,web_admitted=False,
  note='Both comparison groups are previously used diagnostics, not new final validation. Shared individual finger models unchanged.'))
 print(json.dumps(dict(development=summary,separate=fresh,gate=gate),ensure_ascii=False))
if __name__=='__main__':main()
