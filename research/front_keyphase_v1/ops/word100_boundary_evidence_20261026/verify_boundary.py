"""Independent saved-score recount and input contract audit. No inference."""
from pathlib import Path
import json,hashlib,sys,collections,math
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
R=Path('C:/MyUnityProjects/WordsInSilence');O=R/'research/front_keyphase_v1/ops'
U=O/'word100_boundary_evidence_20261026';B=O/'word100_body_boundary_20261017'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
f=read(U/'FROZEN.json')
for p,h in f['pins'].items():assert sha(p)==h,p
rows=read(U/'INPUT_AUDIT.json')['rows'];ctx=read(B/'training_CONTEXT.json')
assert len(rows)==2398 and len({(r['clip_id'],r['view']) for r in rows})==2398
assert collections.Counter(r['role'] for r in rows)=={'train_candidate':1798,'development_only':600}
sources={r['clip_id']:r for r in ctx['ordered_rows']};checked=0;summary={}
for r in rows:
 recpath=B/'data/records'/f"{r['clip_id']}.json";assert sha(recpath)==r['record_sha256']
 rec=read(recpath);v=r['view'];z=load(rec['views'][v]['observation']);ch=load(rec['views'][v]['channel_file'])
 assert sha(rec['views'][v]['observation'])==r['observation_sha256']
 assert sha(rec['views'][v]['channel_file'])==r['channel_sha256']==sources[r['clip_id']]['views'][v]['sha256']
 usable=z['usable'];frames=[i for i in range(len(usable)) if any(bool(x) for x in usable[i,33:75])]
 lo,hi=frames[0],frames[-1];assert [lo,hi]==r['span'] and hi-lo+1==r['frames']
 assert int(sum(usable[lo,33:75]))==r['first_hand_points']
 assert int(sum(usable[hi,33:75]))==r['last_hand_points']
 assert sum(not any(usable[i,33:75]) for i in range(lo,hi+1))==r['internal_no_hand_frames']
 assert np.array_equal(z['frame_id'][lo:hi+1],ch['whole_frame_id'])
 assert np.array_equal(z['timestamp_us'][lo:hi+1],ch['whole_ts'])
 # Independently derive shoulder scalar and a real body-relative channel.
 xy=z['xy_px'][lo:hi+1];ok=usable[lo:hi+1,11]&usable[lo:hi+1,12]
 lengths=[math.hypot(float(xy[i,11,0]-xy[i,12,0]),float(xy[i,11,1]-xy[i,12,1])) for i in range(len(xy)) if ok[i]]
 scale=float(np.median(lengths));assert scale>0
 for i in [0,len(xy)//2,len(xy)-1]:
  if ok[i]:
   for out,j in enumerate((0,11,12,13,14,15,16,23,24)):
    if usable[lo+i,j]:
     expected=(xy[i,j].astype(float)-(xy[i,11].astype(float)+xy[i,12].astype(float))/2)/scale
     assert np.allclose(ch['x_whole'][i,2*out:2*out+2],expected,atol=2e-6,rtol=2e-6)
     checked+=2
 key=f"{r['role']}/{v}";s=summary.setdefault(key,dict(clips=0,sparse_start=0,sparse_end=0,wrist_only_edges=0))
 s['clips']+=1;s['sparse_start']+=r['first_hand_points']<=2;s['sparse_end']+=r['last_hand_points']<=2;s['wrist_only_edges']+=sum(r['wrist_only_edges'])
reports={}
for name in ('C13','T3F13'):
 obj=read(U/f'{name}_STRESS_SCORES.json');old=read(Path(obj['saved_original_scores']))['predictions'];reports[name]={}
 for v,conditions in obj['scores'].items():
  baseline={r['clip_id']:r for r in old[v]};reports[name][v]={}
  for kind,pp in conditions.items():
   assert len(pp)==300 and {p['clip_id'] for p in pp}==set(baseline)
   right=[];wrong=[];changed=0;hits=0;top5=0
   for p in pp:
    zz=p['logits'];assert len(zz)==100 and all(math.isfinite(x) for x in zz)
    rank=sorted(range(100),key=lambda j:(-zz[j],j));correct=rank[0]==p['label_index'];base=baseline[p['clip_id']]
    assert rank[0]==p['prediction'] and rank[:5]==p['top5'] and correct==p['correct']
    hits+=correct;top5+=p['label_index'] in rank[:5];changed+=rank[0]!=base['prediction']
    if correct and not base['correct']:right.append(p['clip_id'])
    if not correct and base['correct']:wrong.append(p['clip_id'])
   m=obj['metrics'][v][kind];basehits=sum(p['correct'] for p in baseline.values())
   assert (hits,top5,changed)==(m['correct'],m['top5'],m['label_changed'])
   assert set(right)==set(m['corrected']) and set(wrong)==set(m['new_errors'])
   reports[name][v][kind]=dict(correct=hits,baseline=basehits,delta_pp=(hits-basehits)/3,changed=changed,gains=len(right),losses=len(wrong))
stress=read(U/'STRESS_INPUTS.json')['rows'];assert len(stress)==1800
assert all(r['hand_scales_before']==r['hand_scales_after'] for r in stress)
assert all(r['central_masks_unchanged'] and r['central_intervals_unchanged'] for r in stress)
out=dict(status='PASS',implementer='Codex',verifier='Codex separate scalar/input implementation; not independent human ground truth',
 input_records_verified=len(rows),scalar_body_coordinates=checked,raw_score_rows_verified=3600,
 natural_edges=summary,scores=reports,hand_scale_changes=0,
 central_coordinate_changes=sum(r['central_coordinate_max_change']>0 for r in stress),
 limits=['Synthetic boundary contamination is not measured natural-user frequency or cause.',
         'Sparse natural boundaries include valid entering/leaving hands and are not error labels.',
         'Public development audit, not new heldout evaluation. No accuracy improvement claim.'])
with (U/'VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
