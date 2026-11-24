"""Finite FIT-only search aid for continuously moving, visible hands.

Scores rank already stored coordinate evidence only; no label or model score is
used. They never establish absence of a hold. Original failed tests stay fixed.
"""
from pathlib import Path
import sys,json,hashlib,datetime
sys.dont_write_bytecode=True
import numpy as np
N=Path(__file__).resolve().parent
O=N.parent
U=O/'word100_core_feature_split_20261101'
W=O/'word100_interval_motion_20261101'
S=N/'support_inventory'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    S.mkdir(exist_ok=False)
    assert read(N/'NATIVE_DIAGNOSIS.json')['status']=='SOURCE_DIAGNOSIS_COMPLETE_TRAINING_NOT_READY'
    contract=dict(status='FROZEN_BEFORE_SELECTION_AND_NEW_RGB',scope='Existing60FITclips180windows only; no dev/evaluation, private recordings or model scores.',
        pool='All180fixed13-native-framewindows, bothhands360opportunities. Exclude existing failed24windowIDs only to avoid rechecking same images. Keep every exclusion/reason.',
        eligibility='All13 observed masked native hand-wrist XY and palm XY finite/usable in stored fixedROI inputs. Coordinates are torso-centred contemporaneous shoulder-width units.',
        ranking='For each hand compute12 consecutive2D step lengths separately for wrist and palm. Score=min(all24step lengths). Rank descending; exact ties sample_id. Sensor jitter/identity errors may still rank high: direct nativeRGB determines support, score is not a label.',
        selection='Round-robin personID1,2,3 then handR,L;2rounds, choose highest remaining candidate with unique clip. Maximum12windows, no result-driven replacements or extra search. If a group lacks an eligible candidate recordshortfall.',
        visual='Same fixed native full/crop/pairs as N, same source13frame IDs/PTS. Target hand hold only. Entire13 observed identity/position/ROI and definite absence required for0. Unknown remainsnull. A clear maintained>=3frame interval gives1. All original rubric cautions remain.',
        interpretation='Class-support inventory, not independent reliability/accuracy or a way to dilute failed24 denominator. One source diagnosis unresolved case remains explicit. No full288 relabeling/training from search ranking or reviewer agreement alone.',
        stop='Stop after at most12 selectedwindows. Never swap uncertain/holdpositive/ROIineligible cases for next-ranked scenes.',
        pins={str(p):sha(p) for p in [Path(__file__),N/'NATIVE_DIAGNOSIS.json',N/'PLAN_FROZEN.json',W/'RUBRIC.json',W/'PILOT_MANIFEST.json',U/'INPUTS.json',U/'VISUAL_INDEX.json',O/'word100_guide_video_feedback_20261028/motion_observer.py']})
    write(S/'PLAN_FROZEN.json',contract)
    prior={r['sample_id'] for r in read(W/'PILOT_MANIFEST.json')['rows']}
    inputs={r['alias']:r for r in read(U/'INPUTS.json')['rows'] if r['role']=='fit'}
    visuals={r['alias']:r for r in read(U/'VISUAL_INDEX.json')['rows']}
    assert len(inputs)==60
    candidates=[]; window_count=0
    for alias,inp in sorted(inputs.items()):
        assert sha(inp['input'])==inp['input_sha256']
        with np.load(inp['input']) as z:
            x=z['x']; mask=z['mask']; ids=z['frame_id'];ts=z['timestamp_us']
            names=inp['feature_names'];lookup={int(v):i for i,v in enumerate(ids)}
            for v in visuals[alias]['windows']:
                window_count+=1
                ix=np.array([lookup[f] for f in v['frame_ids']]);assert len(ix)==13
                for h in 'RL':
                    cols=[names.index(h+'_'+k) for k in ['hand_wrist_x','hand_wrist_y','palm_center_x','palm_center_y']]
                    a=x[ix][:,cols];m=mask[ix][:,cols]
                    reasons=[]
                    if v['sample_id'] in prior:reasons.append('previous24window')
                    if not m.all() or not np.isfinite(a).all():reasons.append('incomplete_observed_fixedROI_position')
                    steps=np.concatenate([np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1),np.linalg.norm(np.diff(a[:,2:],axis=0),axis=1)])
                    score=float(steps.min()) if not reasons else None
                    candidates.append(dict(sample_id=v['sample_id'],alias=alias,clip_id=inp['clip_id'],person=inp['person'],word_id=inp['word_id'],hand=h,
                        eligible=not reasons,exclude_reasons=reasons,score=score,
                        steps=steps.tolist() if not reasons else None,
                        frame_ids=v['frame_ids'],timestamps_us=v['timestamps_us'],source=inp['source'],source_sha256=inp['source_sha256'],
                        input=inp['input'],input_sha256=inp['input_sha256'],roi_bottom=inp['box'][3]))
    assert window_count==180 and len(candidates)==360
    selected=[];used=set();short=[]
    for roundno in range(2):
        for person in [1,2,3]:
            for h in 'RL':
                eligible=sorted([r for r in candidates if r['eligible'] and r['person']==person and r['hand']==h and r['clip_id'] not in used],key=lambda r:(-r['score'],r['sample_id']))
                if not eligible:short.append(dict(round=roundno,person=person,hand=h));continue
                r=dict(eligible[0],selection_round=roundno,support_alias=f'C{len(selected)+1:02}')
                selected.append(r);used.add(r['clip_id'])
    write(S/'SELECTION.json',dict(status='FIXED_BEFORE_RGB_REVIEW',opportunities=360,windows=180,selected=selected,candidates=candidates,shortfalls=short,
        model_scores_read=0,coordinate_rank_not_truth=True,original_test_denominator_unchanged=True,
        pins=contract['pins']))
    print(json.dumps(dict(selected=len(selected),eligible=sum(r['eligible'] for r in candidates),shortfalls=short,samples=[(r['sample_id'],r['hand']) for r in selected])))
if __name__=='__main__':main()
