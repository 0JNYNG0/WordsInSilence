from core import *
guard=load('_completion_identity_guard',E/'completion_guard.py')
metrics=load('_completion_guard_metrics',E/'evaluate.py')
def main():
 assert not (E/'GUARD_COMPARISON.json').exists()
 write(E/'GUARD_FROZEN.json',dict(kind='post-hoc integration correction, same frozen heads; not new independent evaluation',reason='Progression uncertainty is not proof of incomplete motion. Replacing all observation windows regressed normal shape judgments.',rule='Keep deployed windows and all existing differences; only positive conflicting event identity may withhold similarity.',thresholds='Existing0.65 and60ms reused, no search/tuning',all_seeds=[42,43,44],primary=42,code_sha256=sha(E/'completion_guard.py'),first_sha256=sha(E/'COMPARISON.json'),followup_sha256=sha(E/'FOLLOWUP.json'),scope='No new neural inference, training or DEV-based seed choice'))
 mainrows={(r['clip_id'],r['kind']):r for r in read(U/'INPUTS.json')['rows'] if r['role']=='development'};legacy={(r['clip_id'],r['kind']):r for r in read(P/'EVAL_INPUTS.json')['rows']};result={}
 for seed in [42,43,44]:
  name=f'final_{seed}';rows=[]
  for r in read(E/'scores'/f'{name}_cases.json'):
   key=r['key'];src=(mainrows if r['inventory']=='main144' else legacy)[(r['clip_id'],r['kind'])];z=np.load(src['path']);raw=z['raw'] if r['inventory']=='main144' else z['raw_x'];p=np.load(E/'predictions'/name/f'{key}.npz')['prob'];base=read(N/'decisions/web'/f'{key}.json');new=guard.apply_guard(base,p,raw,z['timestamp_us']);dest=E/'guard_decisions'/name/f'{key}.json';write(dest,new);rows.append(dict(r,after=metrics.compact(new),decision=str(dest),sha256=sha(dest)))
  write(E/'guard_scores'/f'{name}_cases.json',rows);s=metrics.summary(rows);result[name]=dict(by_kind=s);event('guard_compared',run=name,normal=s['main144/full_normal'],omit=s['main144/omit_final_event'])
 write(E/'GUARD_COMPARISON.json',dict(models=result,posthoc=True,web_default_changed=False,independent_evaluation=False))
if __name__=='__main__':main()
