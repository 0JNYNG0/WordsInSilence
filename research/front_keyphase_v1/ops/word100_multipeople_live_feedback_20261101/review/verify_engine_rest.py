"""Synthetic final reference-rest/denominator/output audit. No public/model calls."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
import traceback
import numpy as np

sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent; U=R.parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def run():
 e=module('_rest_final',U/'engine.py');old=module('_rest_before',R/'ENGINE_PRE_REST_REPAIR.py')
 names=json.loads((U/'PUBLIC_INPUTS.json').read_text(encoding='utf-8-sig'))['feature_names']
 tests=[]
 def check(name,condition,details=None):tests.append(dict(name=name,passed=bool(condition),details=details))
 def sample(n=20):
  x=np.zeros((n,69));x[:,11:25]=.2;x[:,39:53]=.2;x[:,[25,53]]=1
  return dict(x=x,mask=np.ones((n,69),bool),probabilities=np.tile([.9,.1,.1,.1,.9,.9,.1],(n,1)),frame_id=np.arange(n),timestamp_us=np.arange(n)*33333,feature_names=names)
 def row(rest=(),cid='r1',person=1):
  return dict(clip_id=cid,person=person,label='synthetic',word_id='WORD0001',intervals=[[0,10]],stages=[dict(id='main',canonical_stage_id='main',title='synthetic',frames=[0,19],confirmed_rest_sides=list(rest),reference_rest_evidence={'synthetic':True,'linguistic_requiredness':None},boundary_provenance={'kind':'synthetic_inferred'})])
 def family(c,name):return c['stages'][0]['families'][name]
 q,r=sample(),sample();q['x'][:,25]=-1
 savedq=copy.deepcopy(q);savedr=copy.deepcopy(r)
 c=e.compare_one(q,r,row(['L']));axis=family(c,'hand_axis')
 check('one-side-rest alignment and scoped axis readout agree',c['distance']==0 and axis['peak_distance']==0)
 check('axis original expected retained',axis['original_expected']==80 and axis['expected']==40 and axis['scope_excluded']==40)
 check('axis original possible and scope excluded possible retained',axis['original_possible']==80 and axis['scoped_possible']==40 and axis['scope_excluded_possible']==40)
 check('excluded actual axis difference retained as original evidence',axis['original_evidence']['peak_distance']==2 and axis['original_evidence']['peak_at']['coordinate']==0)
 check('scoped dimensions exclude left and retain right',axis['dimension_reference_expected']==[0,0,20,20] and axis['dimension_scope_excluded']==[20,20,0,0])
 check('no input coordinate or mask modification',all(np.array_equal(a[k],b[k],equal_nan=True) for a,b in [(q,savedq),(r,savedr)] for k in ['x','mask','probabilities','frame_id','timestamp_us']))
 for key in e.FAMILIES:
  v=family(c,key)
  check('original count closure '+key,v['original_expected']==v['expected']+v['scope_excluded'] and v['original_possible']==v['scoped_possible']+v['scope_excluded_possible'])
  check('coordinate excluded count closure '+key,sum(v['dimension_scope_excluded'])==v['scope_excluded'] and sum(v['dimension_original_reference_expected'])==v['original_expected'])
 left=family(c,'L_shape')
 check('fully excluded shape has null scoped distance and retained original evidence',not left['comparison_required'] and left['expected']==0 and left['distance'] is None and left['original_expected']==40 and left['original_evidence']['distance']==0)
 check('rest rule keeps source identity and unknown linguistic requiredness',c['reference_scope_rules'][0]['stage_id']=='main' and c['reference_scope_rules'][0]['linguistic_requiredness'] is None and c['stages'][0]['scope_source_stage_ids']==['main'])
 missing=sample();missing['mask'][:]=False;missing['probabilities'][:]=np.nan
 cm=e.compare_one(missing,r,row(['L']))
 check('query missing cannot reduce original or scoped reference denominators',all(all(family(cm,k)[f]==family(c,k)[f] for f in ['original_expected','expected','scope_excluded','original_possible','scoped_possible']) for k in e.FAMILIES))
 check('active query missing remains unknown',family(cm,'arm_path')['observed']==0 and family(cm,'arm_path')['distance'] is None)
 q2=sample();q2['x'][:,53]=-1
 c2=e.compare_one(q2,r,row(['L']))
 check('rest left does not discard active right axis',family(c2,'hand_axis')['peak_distance']==2 and c2['distance']>0)
 c3=e.compare_one(q2,r,row(['R']))
 check('right rest exclusion symmetric',c3['distance']==0 and family(c3,'hand_axis')['dimension_reference_expected']==[20,20,0,0])
 both=e.compare_one(q,r,row(['L','R']))
 check('both rest excludes all axis scoped opportunity but retains raw',family(both,'hand_axis')['expected']==0 and family(both,'hand_axis')['original_expected']==80 and family(both,'hand_axis')['scope_excluded']==80 and family(both,'hand_axis')['original_evidence']['peak_distance']==2)
 unknownr=sample();unknownr['mask'][:,25:27]=False
 cu=e.compare_one(sample(),unknownr,row(['L']))
 va=family(cu,'hand_axis')
 check('unknown reference support not confused with excluded opportunity',va['original_expected']==40 and va['scope_excluded']==0 and va['scope_excluded_possible']==40 and va['original_possible']==80)
 no_rest=row();a=sample();a['x'][8,41]=.7
 before=old.compare_one(a,r,no_rest);after=e.compare_one(a,r,no_rest)
 check('no-rest original alignment and distance exactly preserved',before['path']==after['path'] and before['distance']==after['distance'])
 check('no-rest family metric values preserved',all(all(before['stages'][0]['families'][k][f]==after['stages'][0]['families'][k][f] for f in ['distance','peak_distance','coverage','expected','observed','query_expected','query_observed']) for k in e.FAMILIES))
 overlapping=row(['L']);overlapping['stages'][0]['frames']=[0,11]
 overlapping['stages'].append(dict(id='second',canonical_stage_id='second',title='second',frames=[8,19],confirmed_rest_sides=[]))
 co=e.compare_one(q,r,overlapping);v=co['stages'][1]['families']['hand_axis']
 check('overlap uses identical global reference coordinate scope',v['original_expected']==48 and v['expected']==40 and v['scope_excluded']==8 and co['stages'][1]['scope_source_stage_ids']==['main'])
 bad=row();bad['stages'][0]['frames']=[-1,19]
 try:e.compare_one(sample(),sample(),bad);rejected=False
 except ValueError:rejected=True
 check('negative reference frame cannot wrap around',rejected)
 # Evaluate only synthetic arrays with an injected synthetic bank and fixed limits.
 rows=[row(['L'],f'r{i}',i) for i in [1,2,3]];bank={v['clip_id']:sample() for v in rows}
 original_bank,original_read=e.bank,e.read
 try:
  e.bank=lambda:({'rows':rows},bank)
  e.read=lambda _: {'ranges':{k:{'p90':.05,'peak':.05} for k in e.FAMILIES}}
  output=e.evaluate_arrays(sample(),'WORD0001')
 finally:e.bank,e.read=original_bank,original_read
 stage=output['stages'][0]
 check('all individual references retained',len(output['references'])==3 and {v['id'] for v in output['references']}=={'r1','r2','r3'})
 check('integrated partial scope explicitly named',stage['partial_scope'] and '제외한 항목은 유사 판정에 포함되지 않습니다' in stage['message'] and e.LABELS['L_shape'] in stage['scope_excluded_feature_names'])
 check('no linguistic correctness manufactured',stage['linguistic_correctness'] is None and all(f['linguistic_requiredness'] is None for f in stage['features']))
 check('one fixed whole-reference pair across all families',all(f['reference_ids']==output['selected_whole_references'] for f in stage['features']))
 for reference in output['references']:
  s=reference['stages'][0]
  check('per-reference scope and IDs preserved '+reference['id'],s['partial_scope'] and all(f['reference_ids']==[reference['id']] for f in s['features']))
  check('per-reference denominator projection exact '+reference['id'],all(f['original_expected'][0]==f['scoped_expected'][0]+f['scope_excluded'][0] for f in s['features']))
  check('per-reference original PTS arrays exact '+reference['id'],reference['query_frame_times']==(sample()['timestamp_us']/1e6).tolist() and reference['reference_frame_times']==(bank[reference['id']]['timestamp_us']/1e6).tolist())
 check('rest-position finger motion limitation explicit',any('쉼 위치' in v and '손가락' in v for v in output['limitations']))
 app=(U/'app.js').read_text(encoding='utf-8');html=(U/'index.html').read_text(encoding='utf-8')
 check('UI explains rest context rather than semantic irrelevance','참고의 쉼 맥락 · 별도 범위' in app and '교육적으로 중요하지 않다는 뜻은 아닙니다' in app)
 check('UI explains two selected whole references and all individual results','동일한 두 참고' in html and 'selected_whole_references' in app)
 return tests


if __name__=='__main__':
 destination=R/'VERIFY_ENGINE_REST_FINAL.json'
 if destination.exists():raise SystemExit('Preserve existing final audit')
 pin=sha(U/'engine.py');tests=[];error=None
 try:tests=run()
 except BaseException:error=traceback.format_exc()
 d=dict(status='PASS' if error is None and all(t['passed'] for t in tests) else 'FAIL',checks=len(tests),results=tests,exception=error,
  engine_sha256=pin,engine_unchanged=sha(U/'engine.py')==pin,verifier_sha256=sha(__file__),
  prior_counterexample_sha256=sha(R/'REST_SCOPE_FIRST_FAILURE.json'),
  scope='Synthetic reference-only scope/count/arithmetic tests and UI text review. No public paths/model/adapter execution.',
  limitations=['Reference-rest is a supported geometric proximity context, not human linguistic non-requiredness.','Important finger movement may occur near the reference rest position; its educational importance remains unvalidated.','Reference-scope exclusions change the compared subspace; before and after raw evidence remain separately available.','Scope corrections were made before the root-owned new bank calibration; no public score was used to select thresholds.'],
  actions=dict(public_paths=0,model_inference=0,training=0,camera=0,user_recordings=0))
 with destination.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':d['status'],'checks':len(tests),'failed':[t['name'] for t in tests if not t['passed']],'exception':error}))
 if d['status']!='PASS':raise SystemExit(1)
