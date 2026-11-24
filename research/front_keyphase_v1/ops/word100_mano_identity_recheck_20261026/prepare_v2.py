from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f={'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'candidate_sha256':sha(U/'identity_v2.py'),'v1_candidate_sha256':sha(U/'identity.py'),'v1_result_sha256':sha(U/'RESULT.json'),'base_protocol_sha256':sha(U/'FROZEN.json'),
'reason':'V1 anchors can come from a known HaMeR-vs-original duplicate conflict, reaffirming the wrong original slot. Also close fingertip-only rivals can veto distinct wrists even though that rival failed wrist correspondence. Preserve V1 as negative finding.',
'change':'Exclude every already known duplicate-conflict hand slot from both before/after anchor sets, including saved non-trigger context comparisons. Unique selection compares only hypotheses satisfying BOTH wrist and fingertip correspondence. No thresholds changed, no word exceptions.',
'scope':'Same7public video243saved HaMeR requests/57missing slots and same2broad RGB scenes. All original observations retained; unavailable two-sided image evidence stays unresolved. Broad is diagnostic only.',
'finger_truth':'Identity/position support does not validate reconstructed finger shape. No new neural inference/training/production change/user data.'}
with (U/'FROZEN_V2.json').open('x',encoding='utf-8') as q:json.dump(f,q,ensure_ascii=False,indent=2)
s=(U/'run.py').read_text('utf-8').replace('from identity import PARAMS,build_anchors,get_tracks,assess,choose','from identity_v2 import PARAMS,build_anchors,get_tracks,assess,choose')
s=s.replace("assert not (U/'RESULT.json').exists()","assert sha(U/'identity_v2.py')==read(U/'FROZEN_V2.json')['candidate_sha256'];assert not (U/'RESULT_V2.json').exists()")
s=s.replace("U/'outputs'","U/'outputs_v2'").replace("U/'tracks'","U/'tracks_v2'")
old="gray,fps=video_gray(m['rgb_path'],z['frame_id']); anchors=build_anchors(z['full_xy'],z['full_valid'],z['pose'],z['pose_valid'],inspect_pair);cache={}"
new="""gray,fps=video_gray(m['rgb_path'],z['frame_id']); suppressed=np.zeros(z['full_valid'].shape[:2],bool)
        for cr in read(G/'REQUESTED_DIAGNOSTIC_V2.json'):
            if cr['clip_index']==k and cr['conflict']:
                suppressed[cr['frame'],cr['side']]=True
                for pair in cr['pairs']:
                    if pair['hold_new']:suppressed[cr['frame'],pair['existing_side']]=True
        anchors=build_anchors(z['full_xy'],z['full_valid'],z['pose'],z['pose_valid'],inspect_pair,suppressed);cache={}"""
assert old in s;s=s.replace(old,new)
s=s.replace("roi_box=z['roi_box'],anchors=anchors)","roi_box=z['roi_box'],anchors=anchors,anchor_suppressed=suppressed)")
s=s.replace("U/'TRACK_INDEX.json'","U/'TRACK_INDEX_V2.json'").replace("U/'DIAGNOSTICS.json'","U/'DIAGNOSTICS_V2.json'").replace("U/'RESULT.json',report","U/'RESULT_V2.json',report")
with (U/'run_v2.py').open('x',encoding='utf-8') as q:q.write(s)
print('V2 frozen; no thresholds changed')
