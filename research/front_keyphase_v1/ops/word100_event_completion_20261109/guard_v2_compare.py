from core import *
guard=load('_completion_guard_v2',E/'completion_guard_v2.py');metrics=load('_completion_guard_v2_metrics',E/'evaluate.py')
def main():
 assert not (E/'GUARD_V2_COMPARISON.json').exists();contractpath=O/'word100_positive_direction_20261109/GUIDE_CONTRACTS.json';contracts=read(contractpath)['contracts']
 write(E/'GUARD_V2_FROZEN.json',dict(kind='post-hoc semantic contract consistency repair; no new model',reason='Existing guide marks stroke and optional arrival as one directional motion, but new event classes were incorrectly treated as competing actions.',rule='Use existing contract.phase_groups as compatible identities; preserve original windows and all shape/direction checks.',existing_contract_path=str(contractpath),existing_contract_sha256=sha(contractpath),first_guard_sha256=sha(E/'GUARD_COMPARISON.json'),code_sha256=sha(E/'completion_guard_v2.py'),v1_code_sha256=sha(E/'completion_guard.py'),no_new_threshold=True,no_word_branch=True,no_model_selection=True,independent_evaluation=False))
 mainrows={(r['clip_id'],r['kind']):r for r in read(U/'INPUTS.json')['rows'] if r['role']=='development'};legacy={(r['clip_id'],r['kind']):r for r in read(P/'EVAL_INPUTS.json')['rows']};result={}
 for seed in [42,43,44]:
  name=f'final_{seed}';rows=[]
  for r in read(E/'scores'/f'{name}_cases.json'):
   key=r['key'];src=(mainrows if r['inventory']=='main144' else legacy)[(r['clip_id'],r['kind'])];z=np.load(src['path']);raw=z['raw'] if r['inventory']=='main144' else z['raw_x'];p=np.load(E/'predictions'/name/f'{key}.npz')['prob'];base=read(N/'decisions/web'/f'{key}.json');new=guard.apply_guard(base,p,raw,z['timestamp_us'],contracts);dest=E/'guard_v2_decisions'/name/f'{key}.json';write(dest,new);rows.append(dict(r,after=metrics.compact(new),decision=str(dest),sha256=sha(dest)))
  write(E/'guard_v2_scores'/f'{name}_cases.json',rows);s=metrics.summary(rows);result[name]=dict(by_kind=s);event('contract_repaired',run=name,normal=s['main144/full_normal'],crop=s['legacy93/crop_normal'],omit=s['main144/omit_final_event'])
 write(E/'GUARD_V2_COMPARISON.json',dict(models=result,posthoc=True,web_default_changed=False,independent_evaluation=False))
if __name__=='__main__':main()
