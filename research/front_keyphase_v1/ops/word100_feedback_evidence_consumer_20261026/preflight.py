from pathlib import Path
import json,datetime,hashlib
import numpy as np
import consumer as C
U=Path(__file__).resolve().parent
def base():
 sh=(7,2,23);yes=np.ones(sh,bool)
 return dict(domain=yes.copy(),point_ok=yes.copy(),point_delta=np.zeros(sh),sequence_ok=yes.copy(),sequence_delta=np.zeros(sh+(5,)),sequence_quality=yes.copy(),source=np.ones(sh,np.uint8),reference_missing=np.zeros(sh,bool),reference_times_us=np.arange(7)*40000,reference_id='public_reference',query_id='synthetic',kind='fixture')
def hide(x,hand,chs,frames):
 for k in ['point_ok','sequence_ok','sequence_quality']:x[k][frames,hand,chs]=False
 x['point_delta'][frames,hand,chs]=np.nan;x['sequence_delta'][frames,hand,chs]=np.nan;x['source'][frames,hand,chs]=0
def main():
 tests=[];x=base();r=C.build(**x);assert r['whole_gesture_correct'] is None and r['hands'][0]['groups'][0]['point_mean_difference']==0;tests.append('observed_zero_is_not_correct_gesture')
 x=base();hide(x,0,slice(None),slice(None));r=C.build(**x);assert all(g['point_mean_difference'] is None and g['sequence_level_mean_difference'] is None and g['evidence_state']=='no_comparable_evidence' for g in r['hands'][0]['groups']);tests.append('unobserved_zero_never_fabricated')
 x=base();hide(x,0,slice(0,17),slice(None));hide(x,0,slice(21,23),slice(None));r=C.build(**x);assert r['hands'][0]['groups'][0]['point_compared_cells']==0 and r['hands'][0]['groups'][2]['point_compared_cells']==14;tests.append('position_never_fills_finger_and_arm_retained')
 x=base();x['sequence_quality'][2:5,0,3]=False;r=C.build(**x);assert r['hands'][0]['angles'][3]['corroborated_cells']==4 and r['hands'][0]['angles'][2]['corroborated_cells']==7;tests.append('uncertain_angle_keeps_neighbor')
 x=base();hide(x,1,1,slice(2,5));r=C.build(**x);a=r['hands'][1]['angles'][1];assert a['audit_cells']==7 and a['point_compared_cells']==4 and a['missing_point_intervals']==[dict(first=2,last=4,samples=3,reference_start_s=.08,reference_end_s=.16)];tests.append('gap_denominator_and_clock_preserved')
 x=base();x['sequence_delta'][...,3:]=4.;r=C.build(**x);g=r['hands'][0]['groups'][0];assert g['sequence_level_mean_difference']==0 and g['sequence_change_mean_difference']==4;tests.append('levels_and_changes_separate')
 x=base();before={k:v.copy() for k,v in x.items() if isinstance(v,np.ndarray)};r=C.build(**x);assert all(np.array_equal(x[k],v,equal_nan=True) for k,v in before.items());tests.append('inputs_readonly')
 a=C.build(**base(),recognition={'prediction':0});b=C.build(**base(),recognition={'prediction':99});assert a['hands']==b['hands'];tests.append('classifier_never_controls_feedback')
 for name,alter in [('missing_point_zero',lambda x:x['point_ok'].__setitem__((0,0,0),False)),('partial_sequence_values',lambda x:x['sequence_delta'].__setitem__((0,0,0,0),np.nan)),('quality_without_observation',lambda x:x['sequence_ok'].__setitem__((0,0,0),False)),('unknown_provenance',lambda x:x['source'].__setitem__((0,0,0),0)),('negative_difference',lambda x:x['point_delta'].__setitem__((0,0,0),-1)),('nonmonotone_clock',lambda x:x['reference_times_us'].__setitem__(1,0))]:
  x=base();alter(x)
  try:C.build(**x)
  except ValueError:tests.append('reject_'+name)
  else:raise AssertionError(name)
 p=U/'consumer.py';out=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS_CONTRACT_FIXTURES',tests=tests,consumer_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),real_video_validation_pending=True)
 with (U/'PREFLIGHT.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
 print('PREFLIGHT',len(tests),'PASS')
if __name__=='__main__':main()
