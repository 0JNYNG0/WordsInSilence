from pathlib import Path
import json
D=Path(__file__).resolve().parent;P=D.parent/'selective_rtmw_followup_20261117'
s=(P/'selective_retry.py').read_text(encoding='utf-8').replace('selective_front_rtmw_retry_v2','selective_front_rtmw_retry_v3')
s=s.replace('if len(other) >= 3:', 'if a <= i <= b and len(other) >= 3:')
needle='            one,two=hand_outputs\n'
addition='''            # Only missing frames will be repaired. Known bracket frames are
            # still checked against both crops and original points, but no new
            # points are inserted there. Reject overlap of either NEW hand pose
            # at every frame that would receive repaired points.
            if a <= i <= b:
                opposite=112 if start==91 else 91
                other=raw['xy_px'][i,opposite:opposite+21]
                other=other[visible(other,w,h)]
                if len(other)>=3:
                    for observed in hand_outputs:
                        area,_=cv2.intersectConvexConvex(cv2.convexHull(observed.astype('float32')),cv2.convexHull(other.astype('float32')))
                        if area>0:return done('new_projected_overlap')
'''
assert needle in s;s=s.replace(needle,needle+addition)
(D/'selective_retry.py').write_text(s,encoding='utf-8')
for name in ['probe.py','retry_runtime.py','serve.py','check_capture.py','recheck_private.py']:(D/name).write_bytes((P/name).read_bytes())
v=(P/'verify_public.py').read_text(encoding='utf-8')
v=v.replace("G/'selective_rtmw_retry_20261117/web/ADMISSION.json'","G/'selective_rtmw_followup_20261117/web/ADMISSION.json'")
v=v.replace('[3,4,11,12,16,17]','[3,4,6,7,11,12,16,17]').replace("r['person']in[3,4]","r['person']in[6,7]")
v=v.replace("assert read(D/'DECODE_PROFILE.json')['pixels_equal']","assert read(G/'selective_rtmw_followup_20261117/DECODE_PROFILE.json')['pixels_equal']")
v=v.replace('added persons3/4 conditions','added persons6/7 conditions')
(D/'verify_public.py').write_text(v,encoding='utf-8')
(D/'check_guards.py').write_bytes((P/'check_guards.py').read_bytes())
s=(P/'admit.py').read_text(encoding='utf-8').replace("P=G/'selective_rtmw_retry_20261117'","P=G/'selective_rtmw_followup_20261117'")
s=s.replace('len(pairs)==16','len(pairs)==24').replace("read(D/'DECODE_PROFILE.json')","read(P/'DECODE_PROFILE.json')")
(D/'admit.py').write_text(s,encoding='utf-8')
s=(P/'check_http.py').read_text(encoding='utf-8').replace('selective_front_rtmw_retry_v2','selective_front_rtmw_retry_v3');(D/'check_http.py').write_text(s,encoding='utf-8')
contract=dict(goal='Do not veto visible missing frames solely because an already observed bracket overlaps another hand. Still reject any missing-frame envelope or recovered hand overlap.',evidence=str(P/'OVERLAP_DETAIL.json'),public_only_candidate_selection=True,private_training=False,private_tuning=False,private_candidate_selection=False,frozen_limits=dict(frames=6,duration_us=300000,calls=16,seconds=1.2),unchanged=['Weights, crops, score thresholds, all-joint crop consistency, bracket agreement','Frozen C·T3-F and TCN ranking','Known verdict/error time preservation','Unknown criteria without missing points do not trigger retry'],validation='Prior32 normals, previous24 public dropout controls plus8 added person6/7 dropout cases, blank/long/outside/error controls, capture path, source hashes. Private diagnosis only after public admission.')
(D/'CONTRACT.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
note=dict(changes=['Exempt observed bracket frame from projected overlap veto only','Add both-crop recovered hand overlap rejection at actual missing frames'],why='REAL16 missing frames45..49 have no envelope overlap; only frame50 (not missing) overlaps. REAL17 missing frame87 also overlaps, so remains blocked.',not_changed='No hidden-frame hallucination or interpolation. Every numeric gate unchanged.',no_private_selection=True,implementation='Codex local')
(D/'ENGINEERING_NOTE.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf-8')
