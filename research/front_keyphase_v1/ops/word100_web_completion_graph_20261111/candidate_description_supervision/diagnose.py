from core import *
B=U/'episode_stage_contrast';base={(r['seed'],r['clip_id'],r['kind'],r['card']):r for r in read(B/'BINDING_V2_SCORES.json')['rows']};comp=read(W/'COMPARISON.json');out=[]
for e in comp['predictions']:
 if e['kind']!='dynamic_complete_blocks':continue
 d=np.load(e['path']);rr=[r for k,r in base.items() if k[:3]==(e['seed'],e['clip_id'],e['kind'])]
 for i,r in enumerate(rr):
  if 'truth' not in r:continue
  ids=[k for k,c in enumerate(r['candidates']) if c['has_motion']];scores=[]
  for (ri,ci,j),p in zip(d['spec'],d['p']):
   if ri!=i:continue
   c=r['candidates'][ids[ci]];obs=set(c['body_observed']);w=set(r['truth']['witness']);other=set(r['truth']['wrong_witness']);scores.append(dict(span=c['span'],p=p[j].tolist(),witness_covered=w<=obs,wrong_points=len(other&obs),has_motion=c['has_motion']))
  out.append(dict(seed=e['seed'],clip_id=e['clip_id'],family=r['family'],witness=r['truth']['witness'],candidates=scores,available_clean_complete_witness=any(c['witness_covered'] and not c['wrong_points'] for c in scores)))
write(W/'CANDIDATE_AVAILABILITY_DIAGNOSIS.json',dict(at=stamp(),rows=out,score_only_not_reinference=True))
for r in out:
 if r['seed']==42:print(r)
