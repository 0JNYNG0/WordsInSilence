from core import *
from association import decide,project,prior
import statistics,traceback
def compact(d):return dict(display=[c['display_status'] for c in d['cards']],status=[c['status'] for c in d['cards']],required=[c.get('required',True) for c in d['cards']],phase_present=[all(e['span'] is not None for e in c['phase_evidence']) for c in d['cards']])
def summary(rows):
 out={}
 for inv,kind in sorted({(r['inventory'],r['kind']) for r in rows}):
  rr=[r for r in rows if r['inventory']==inv and r['kind']==kind];s={}
  for arm in ['deployed','prior','after']:
   cc=[r[arm] for r in rr];s[arm]=dict(clips=len(cc),difference=sum('difference' in c['display'] for c in cc),uncertain=sum('uncertain' in c['display'] for c in cc),missing=sum('missing_candidate' in c['status'] for c in cc),similar=sum(all(v=='similar' for v,q in zip(c['display'],c['required']) if q) for c in cc),required_present=sum(all(v for v,q in zip(c['phase_present'],c['required']) if q) for c in cc))
  out[inv+'/'+kind]=s
 return out
def main():
 assert (E/'TRAINING_COMPLETE.json').exists();assert not (E/'COMPARISON.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 write(E/'EVAL_CODE_PINS.json',{str(p):sha(p) for p in [E/'evaluate.py',E/'association.py',E/'core.py',E/'FROZEN.json']})
 mainrows={(r['clip_id'],r['kind']):r for r in read(U/'INPUTS.json')['rows'] if r['role']=='development'};legacy={(r['clip_id'],r['kind']):r for r in read(P/'EVAL_INPUTS.json')['rows']};results={}
 for seed in [42,43,44]:
  name=f'final_{seed}';m,ck=encoder(name);h=ProgressHead();h.load_state_dict(torch.load(E/'runs'/name/'model.pt',weights_only=True,map_location='cpu'));h.eval();rows=[];times=[]
  with torch.inference_mode():
   for oldcase in read(N/'scores'/f'{name}_cases.json'):
    cid=oldcase['clip_id'];kind=oldcase['kind'];inv=oldcase['inventory'];key=oldcase['key'];src=(mainrows if inv=='main144' else legacy)[(cid,kind)];assert sha(src['path'])==src['sha256'];z=np.load(src['path']);raw=z['raw'] if inv=='main144' else z['raw_x'];ts=z['timestamp_us'];fid=z['frame_id']
    web=np.load(N/'predictions/web'/f'{key}.npz');gp=np.load(N/'predictions'/name/f'{key}.npz')['prob'];start=time.perf_counter();f=features(raw,ts,m,ck);p=h(torch.tensor(f)).softmax(-1).numpy();sec=time.perf_counter()-start
    out=decide(src['word_id'],p,web['phase'],prior.expand(gp),web['d0'],web['d1'],web['p'],raw,ts,fid);dest=E/'decisions'/name/f'{key}.json';write(dest,out);pred=E/'predictions'/name/f'{key}.npz';pred.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pred,prob=p)
    rows.append(dict(key=key,clip_id=cid,kind=kind,inventory=inv,deployed=oldcase['before'],prior=oldcase['after'],after=compact(out),decision=str(dest),sha256=sha(dest)))
    if inv=='main144' and kind=='full_normal':times.append(sec)
  assert len(rows)==237;s=summary(rows);write(E/'scores'/f'{name}_cases.json',rows);results[name]=dict(by_kind=s,additional_forward_seconds=dict(median=statistics.median(times),p95=float(np.percentile(times,95)),scope='new frozen geometry+progress head only; not whole web'),conditions=len(rows));write(E/'scores'/f'{name}.json',results[name]);event('evaluated',run=name,normal=s['main144/full_normal'],omit=s['main144/omit_final_event'])
 write(E/'COMPARISON.json',dict(primary='final_42',models=results,web_deployed=False,baseline_neural_reinference=False));write(E/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_HELDOUT_AUDIT'))
if __name__=='__main__':
 try:main()
 except BaseException:
  (E/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
