from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f={'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'candidate_sha256':sha(U/'identity_v3.py'),'v2_candidate_sha256':sha(U/'identity_v2.py'),'v1_candidate_sha256':sha(U/'identity.py'),'v2_result_sha256':sha(U/'RESULT_V2.json'),'reason':'V2 quarantined all matching predictions, including SAME-SIDE HaMeR confirming an existing observed hand. That is not left/right identity conflict and unnecessarily removes normal anchors. Correct scope to cross-side matching only. Preserve V2 evidence.','change':'Quarantine both slots only when saved fingertip duplicate check matches a different requested anatomical side. Keep same-side confirmation eligible for ordinary current-arm checks. Same parameters/frames/predictions, no threshold tuning.'}
with (U/'FROZEN_V3.json').open('x',encoding='utf-8') as q:json.dump(f,q,ensure_ascii=False,indent=2)
s=(U/'run_v2.py').read_text('utf-8').replace('from identity_v2 import PARAMS,build_anchors,get_tracks,assess,choose','from identity_v3 import PARAMS,build_anchors,get_tracks,assess,choose,conflict_mask')
s=s.replace("assert not (U/'RESULT_V2.json').exists()","assert sha(U/'identity_v3.py')==read(U/'FROZEN_V3.json')['candidate_sha256'];assert not (U/'RESULT_V3.json').exists()")
start=s.index("gray,fps=video_gray(m['rgb_path'],z['frame_id']); suppressed=")
end=s.index("        anchors=build_anchors",start)
s=s[:start]+"gray,fps=video_gray(m['rgb_path'],z['frame_id']); suppressed=conflict_mask(read(G/'REQUESTED_DIAGNOSTIC_V2.json'),k,z['full_valid'].shape[:2])\n"+s[end:]
for a,b in [('outputs_v2','outputs_v3'),("U/'tracks_v2'","U/'tracks_v3'"),('TRACK_INDEX_V2.json','TRACK_INDEX_V3.json'),('DIAGNOSTICS_V2.json','DIAGNOSTICS_V3.json'),("U/'RESULT_V2.json',report","U/'RESULT_V3.json',report")]:s=s.replace(a,b)
# Only this unit's output path changed. G/outputs_v2 remains the frozen preceding baseline.
s=s.replace("G/'outputs_v3'","G/'outputs_v2'")
with (U/'run_v3.py').open('x',encoding='utf-8') as q:q.write(s)
print('V3 mask correction frozen')
