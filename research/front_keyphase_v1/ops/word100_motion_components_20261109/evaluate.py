from common import *
from motion import *
from train_motion import physical,makebase
from check_inputs import mirror
import statistics

VARIANTS=['normal','upper','time_slow','time_fast','reverse','missing_2','missing_middle','mirror','translate_scale']
def summary(rr):
 accepted=[r for r in rr if r['eligible'] and r['confident']]
 return dict(n=len(rr),eligible=sum(r['eligible'] for r in rr),raw_correct=sum(r['eligible'] and r['pred']==r['target'] for r in rr),accepted=len(accepted),accepted_correct=sum(r['pred']==r['target'] for r in accepted),withheld=len(rr)-len(accepted),opposite=sum(r['eligible'] and r['target']!=1 and r['pred']==2-r['target'] for r in rr),by_axis={str(j):dict(n=sum(r['axis']==j for r in rr),correct=sum(r['axis']==j and r['eligible'] and r['pred']==r['target'] for r in rr)) for j in range(5)},moving=dict(n=sum(r['target']!=1 for r in rr),correct=sum(r['target']!=1 and r['eligible'] and r['pred']==r['target'] for r in rr)),hold=dict(n=sum(r['target']==1 for r in rr),correct=sum(r['target']==1 and r['eligible'] and r['pred']==r['target'] for r in rr)))

def main():
 assert (U/'INPUT_CHECK.json').exists() and (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists()
 cfg=read(U/'FROZEN.json');rows=[r for r in read(U/'READY.json')['rows'] if r['role']=='development']
 torch.set_num_threads(4);torch.set_num_interop_threads(1);base=makebase();models={};timing={}
 for c in cfg['models']:
  for seed in cfg['seeds']:
   name=f'{c}_{seed}';s=torch.load(U/'runs'/name/'model.pt',weights_only=True,map_location='cpu');m=MotionNet(len(s['mean']));m.load_state_dict(s['model']);m.eval();models[name]=(m,s['mean'],s['std'])
 write(U/'EVAL_FROZEN.json',dict(variants=VARIANTS,geometry_deadbands=[.08,.10,.10,.08,.08],same_historical_DEV=True,primary='motion_only_42',denominator='32 sparse RGB-reviewed interval pairs per condition; missing and low confidence stay in denominator',mirror_scope='Motion-only invariance; frozen latent is kept in anatomical binding, not a new visual inference',time_scope='Timestamp dilation only; no resampling of the source imagery',point_shape_order='Previous deployed base unchanged; direction-only additional readout'))
 records=[]
 with torch.inference_mode():
  for r in rows:
   z=np.load(r['path'])
   for variant in VARIANTS:
    inp=z['x_upper' if variant=='upper' else 'x_full'];x0=physical(inp);h=base(torch.from_numpy(inp[None]),torch.tensor([len(inp)]),torch.tensor([r['word']]))['latent'][0].numpy()
    for pairid,p in enumerate(r['pairs']):
     x=x0.copy();ts=z['timestamp_us'].copy();a,b=p['a'],p['b'];moving,anchor='R','L';reverse=variant=='reverse';target=2-p['target'] if reverse else p['target']
     if variant=='time_slow':ts=ts*1.25
     if variant=='time_fast':ts=ts*.75
     if variant=='mirror':x=mirror(x);moving,anchor='L','R'
     if variant=='translate_scale':x[:,:102]=(x[:,:102].reshape(-1,51,2)*1.7+np.array([2.4,-.8])).reshape(-1,102)
     if variant.startswith('missing_'):
      n=2 if variant=='missing_2' else max(3,int(np.ceil((b-a+1)*.5)));start=(a+b-n+1)//2;x[start:start+n,186:237]=0
     f,eligible,info=features(x,ts,a,b,KINDS[p['axis']],moving,anchor,reverse);aa,bb=(b,a) if reverse else (a,b);latent=np.r_[h[aa],h[bb],h[bb]-h[aa]]
     meta=dict(clip_id=r['clip_id'],person=r['person'],axis=p['axis'],pair=pairid,a=a,b=b,variant=variant,target=target,eligible=bool(eligible),evidence=info)
     delta=info.get('signed_change',0);floor=[.08,.10,.10,.08,.08][p['axis']];gp=2 if delta>floor else 0 if delta<-floor else 1
     records.append(dict(meta,model='geometry',pred=gp if eligible else -1,confident=bool(eligible),confidence=None))
     for name,(m,mean,std) in models.items():
      ff=np.r_[f,latent] if name.startswith('motion_plus') else f
      prob=m((torch.tensor(ff).float()-mean)/std).softmax(-1).numpy();pred=int(prob.argmax());confidence=float(prob.max())
      records.append(dict(meta,model=name,pred=pred if eligible else -1,confident=bool(eligible and confidence>=cfg['confidence']),confidence=confidence))
  for name,(m,mean,std) in models.items():
   inp=torch.zeros((6,len(mean)));times=[]
   for k in range(105):
    start=time.perf_counter();m((inp-mean)/std);elapsed=time.perf_counter()-start
    if k>=5:times.append(elapsed)
   timing[name]=dict(median_ms=statistics.median(times)*1000,p95_ms=float(np.percentile(times,95))*1000,scope='Six precomputed relation intervals, classifier only, CPU; excludes extraction/phase/DTW/rendering')
 write(U/'SCORES.json',records);result={name:{v:summary([r for r in records if r['model']==name and r['variant']==v]) for v in VARIANTS} for name in ['geometry']+list(models)}
 write(U/'EVALUATION_COMPLETE.json',dict(scores_sha256=sha(U/'SCORES.json'),results=result,timing=timing,private_used=False,independent_test=False))
 print(json.dumps({n:{v:result[n][v] for v in ['normal','upper']} for n in result},ensure_ascii=False))
if __name__=='__main__':main()
