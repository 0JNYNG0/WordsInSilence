"""Record root's direct six-page native RGB review; no predictions used."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec={
 11:dict(early=[40,44],setup=[49,51,52],transit=[56,59,61],terminal=[65,69,73],motion=[52,65],stable=[[40,44],[66,73]]),
 12:dict(early=[53,55,90],setup=[63,64,65],transit=[68,70,72],terminal=[77,79,82],motion=[65,77],stable=[[53,57],[79,84]]),
 13:dict(early=[38,42],setup=[50,52,54],transit=[56,60,64],terminal=[73,77,81],motion=[55,73],stable=[[38,42],[77,83]])}
dest=U/'REVIEW_DEV_1133.json';assert not dest.exists()
out=[]
for q in read(U/'DEV_PROCESS_REVIEW_MANIFEST.json')['rows']:
 if q['word_id']!='WORD1133':continue
 s=spec[q['person']];frames={f['frame']:f for f in q['frames']}
 for p in q['pages']:assert sha(p['path'])==p['sha256']
 relations=[]
 for group,setup,terminal in [('early',0,0),('setup',1,0),('transit',0,None),('terminal',0,1)]:
  for fid in s[group]:
   f=frames[fid];assert sha(f['path'])==f['sha256']
   reason={'early':'The hands are in approach/rest or separate finishing positions, not the specified proximal-to-wrist relation.',
    'setup':'Right hand is visibly beside the proximal left arm. Projected adjacency, not verified contact.',
    'transit':'Right hand is progressing along the left arm. The distal-arrival boundary is not assigned at this intermediate image.',
    'terminal':'Right hand is visibly beside the distal left forearm/wrist; small body/hand shifts do not negate adjacency.'}[group]
   relations.append(dict(frame=fid,timestamp_us=f['timestamp_us'],setup_relation=setup,terminal_relation=terminal,reason=reason))
 motion=[]
 for interval,label in [(s['motion'],1)]+[(iv,0) for iv in s['stable']]:
  lo,hi=interval
  motion.append(dict(start=lo,end=hi,start_us=frames[lo]['timestamp_us'],end_us=frames[hi]['timestamp_us'],label=label,
   reason='Native chronological RGB shows right hand progressing along the left arm toward the wrist.' if label else
   'The specified along-arm travel is absent during this approach/rest or distal maintenance interval; natural small motion is allowed.'))
 out.append(dict(id=q['id'],clip_id=q['clip_id'],word_id=q['word_id'],person=q['person'],role='development',
  source_sha256=q['source_sha256'],physical_query=q['physical_query'],reviewed_pages=q['pages'],
  reviewed_all_native_frames=[f['frame'] for f in q['frames']],
  predicate_definitions=dict(setup_relation='Right hand beside proximal left arm/upper arm.',
   terminal_relation='Right hand beside distal left forearm/wrist; projected relation, not verified contact.'),
  relations=sorted(relations,key=lambda r:r['frame']),motion_intervals=motion,
  sequence_links=[dict(setup_frames=s['setup'],motion_interval=s['motion'],terminal_frames=s['terminal'],visible_chain='yes',
   reason='Visible proximal-arm relation is followed through the intervening native images toward the distal wrist relation; no required exact sweep length inferred.')],
  review_bounds_sufficient=True,missing_context=None,whole_word_correctness=None,required_extent=None,
  reviewer='Codex root direct all six chronological sheets via view_image; not independent human/KSL expert, not blind to source word.',
  rationale='Same original FIT physical rubric. This is prereview for new candidate; repeated DEV is not independent final evidence.'))
assert len(out)==3
dest.write_text(json.dumps(dict(status='DIRECT_DEVELOPMENT_PHYSICAL_REVIEW',rows=out,
 manifest_sha256=sha(U/'DEV_PROCESS_REVIEW_MANIFEST.json'),model_predictions_used=False,new_training=0),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('DEV1133 native3 reviewed, no model or language-error claims')
