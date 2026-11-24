"""Saved-output arithmetic inspection; does not run measurements or alignment."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
r=read(U/'RESULT.json'); totals=defaultdict(lambda:defaultdict(int)); n=0
for pair in r['pairs']:
    p=U/'data'/f"{pair['query']}__{pair['reference']}_comparison.npz"
    assert pair['query_source_sha256']!=pair['reference_source_sha256']
    with np.load(p,allow_pickle=False) as z:
        for k in ('angles','body_position','face_distance','face_direct'):
            both=z[k+'_both'];delta=z[k+'_difference'];st=pair['stats'][k]
            assert both.shape==delta.shape
            assert int(both.sum())==st['jointly_observed']
            assert np.array_equal(np.isfinite(delta),both)
            den=int(z['planned'].sum()*np.prod(both.shape[1:]));assert den==st['planned']
            assert st['insufficient']==den-int(both.sum())
            if both.any():assert abs(float(np.median(delta[both]))-st['median_abs_difference'])<1e-5
            else:assert st['median_abs_difference'] is None
            for key in ('planned','query_observed','reference_observed','jointly_observed','insufficient'):totals[k][key]+=st[key]
    n+=1
for k,x in totals.items():
    for key,val in x.items():assert val==r['totals'][k][key]
for p in U.glob('example_*.svg'):ET.parse(p)
pins=read(U/'BEFORE.json');after=read(U/'AFTER.json');assert pins==after
for p,h in pins.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
schema={'hand_order':['anatomical_left','anatomical_right'],'position_entity_order':['hand_wrist','pose_wrist','elbow'],
 'face_entity_order':['thumb_tip','index_tip','middle_tip','ring_tip','little_tip','hand_wrist','palm_center','pose_wrist','elbow','forearm_nearest'],
 'face_region_order':['forehead','cheek_R','cheek_L','mouth','chin','nose'],
 'source_geometry_module':'research/front_keyphase_v1/src/wis_lab/core/hand_face_evidence.py',
 'source_face_region_module':'research/front_keyphase_v1/src/wis_lab/core/face_body_relations.py',
 'raw_evidence_preservation':'Original pinned evidence.npz distance/available/direct_comparable remain unchanged. New pair NPZ saves per-component differences, joint evidence mask, query time and reference mapping.',
 'no_physical_contact':'2D proximity and projection are not contact ground truth.',
 'zero_or_low_difference':'Only observable projected measurements were similar; not evidence of correct signing.'}
(U/'SCHEMA.json').write_text(json.dumps(schema,indent=2)+'\n',encoding='utf-8')
out={'status':'PASS','saved_pairs':n,'pin_count':len(pins),'svg_xml_count':len(list(U.glob('example_*.svg'))),
 'checks':['per-pair finite delta exactly equals comparable evidence','fixed planned denominator matches query annotation frames','sum independently matches all feature totals','distinct query/reference content hashes','source pins still unchanged','SVG well-formed XML'],
 'implementation_actor':'Codex local_feature_inventory','verification_kind':'same agent separate saved-array arithmetic; no human truth'}
(U/'VERIFY_SAVED.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))
