from pathlib import Path
import json,copy,hashlib
from sequence_bundle import build_feature,FEATURES
from evidence_adapter import export_window
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=U/'CONTRACT_VERIFY.json';assert not out.exists()
b=read(U/'BUNDLE.json');fixed=read(U/'FROZEN.json');checks={}
assert all(sha(p)==h for p,h in b['source_pins'].items())
exports=[];roundtrip=0
for s in b['sequences']:
    e=export_window(s);assert set(e)=={'side','expected_frame_count','expected_from_pts_us','expected_to_pts_us','observations','training_ready','note'}
    for f in FEATURES:
        x=s['features'][f]
        recovered=[v['value'] for r in x['runs'] for v in [{'value':r['value']}]*r['count']]
        assert recovered==x['values']==[fr['direct'][f] for fr in s['frames']]
        roundtrip+=len(recovered)
    stripped=copy.deepcopy(s)
    for field in ['label','word_id','clip_id','source_alias','role','recorded_person_id']:stripped[field]='CHANGED_METADATA'
    assert export_window(stripped)==e
    for f in FEATURES:
        for o,fr in zip(e['observations'],s['frames']):assert o['features'][f]['source_value']==fr['direct'][f]
    exports.append({'audit_case_id':s['case_id'],'observation_input':e})
checks['roundtrip_feature_values']=roundtrip;checks['metadata_target_independence_windows']=len(exports)
tested=0
# This tests retention contracts, not recognition at different real signing speeds.
for s in b['sequences']:
    fr=s['frames']
    for f in FEATURES:
        baseline=build_feature(fr,f)
        for factor in [0.5,0.75,1.5,2.0]:
            alt=copy.deepcopy(fr);t0=alt[0]['pts_us']
            for a in alt:a['pts_us']=round(t0+(a['pts_us']-t0)*factor)
            z=build_feature(alt,f)
            assert z['values']==baseline['values']
            assert [(x['kind'],x['from_index'],x['to_index']) for x in z['links']]==[(x['kind'],x['from_index'],x['to_index']) for x in baseline['links']]
            assert [(x['kind'],x['from_index'],x['to_index']) for x in z['gaps']]==[(x['kind'],x['from_index'],x['to_index']) for x in baseline['gaps']]
            tested+=1
        alt=copy.deepcopy(fr)
        for a in alt:a['direct'][f]=None
        z=build_feature(alt,f)
        assert z['coverage']['expected_samples']==13 and z['coverage']['known_samples']==0
        assert z['gaps'][0]['kind']=='entire_window_unobserved' and not z['no_change_certified']
        assert (z['coverage']['expected_from_pts_us'],z['coverage']['expected_to_pts_us'])==(fr[0]['pts_us'],fr[-1]['pts_us'])
        for i in range(13):
            alt=copy.deepcopy(fr);alt[i]['direct'][f]=None;z=build_feature(alt,f)
            assert z['values'][i] is None and z['coverage']['expected_samples']==13
            assert any(i in g['indices'] for g in z['gaps'])
            for j in range(13):assert z['values'][j]==alt[j]['direct'][f]
checks['timestamp_scale_contracts']=tested;checks['single_missing_feature_cases']=36*2*13;checks['whole_feature_missing_windows']=72
def frames(v):return [{'frame_id':100+i,'pts_us':i*33333,'direct':{'index_only':None,'open_four':x}} for i,x in enumerate(v)]
short=build_feature(frames([0,0,1,0,0]),'open_four')
assert [(x['from_index'],x['to_index'],x['from_value'],x['to_value']) for x in short['links']]==[(1,2,0,1),(2,3,1,0)]
samegap=build_feature(frames([1,None,None,1]),'open_four');assert samegap['links'][0]['kind']=='same_endpoints_unresolved_gap'
different=build_feature(frames([0,None,None,1]),'open_four');assert different['links'][0]['kind']=='bounded_feature_change' and different['links'][0]['exact_transition_time'] is None
reverse=build_feature(frames([1,None,None,0]),'open_four');assert (reverse['links'][0]['from_value'],reverse['links'][0]['to_value'])==(1,0)
fake={'side':'R','frames':frames([1])};fake['frames'][0]['direct']['index_only']=1
e=export_window(fake);assert all(v['source_value']==1 and v['comparison_value'] is None for v in e['observations'][0]['features'].values())
checks.update(one_frame_pulse_preserved=True,equal_endpoints_do_not_fill_unknown=True,transition_order_preserved=True,contradiction_preserves_source_and_withholds_comparison=True)
assert all(sha(p)==h for p,h in fixed['protected'].items())
dump(U/'OBSERVATION_EXPORT.json',{'status':'READ_ONLY_DRAFT_INTERFACE_NOT_MODEL','windows':exports,'contains_target_inputs':False,'not_training_ready':True})
dump(out,{'status':'PASS','checks':checks,'scope':'Synthetic transformations test representation contracts, not speed recognition accuracy, hidden-hand recovery, or independent ground truth.','source_pins_unchanged':len(b['source_pins']),'operating_pins_unchanged':len(fixed['protected']),'implementation_and_verification':'Root self-check; separate saved-data verifier reported independently.'})
print(json.dumps(checks,ensure_ascii=False))
