"""Separate saved observation/channel and score arithmetic, no detector/training/inference."""
from common import *
from collections import Counter
import math,importlib.util

def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def inputs(early=False):
 ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins'])
 if not early:assert not (U/'PREP_RUNNING.json').exists() and read(U/'PREP_COMPLETE.json')['records']==199
 paths=sorted((DATA/'records').glob('*.json'));assert len(paths)==199 or early
 if early:paths=paths[:2]
 assert paths
 sys.path.insert(0,str(B));import verify_boundary as vb
 io=vb.input_oracle_module();gg=mod('heldout_observation_groups',B/'verify_full_inputs.py');gg.group_selfcheck()
 byid={r['clip_id']:r for r in ctx['rows']};assert len(byid)==199
 if not early:assert {p.stem for p in paths}==set(byid)
 groups={v:{} for v in VIEWS};details=[];tokens=set();source_hashes=set();values=hidden_points=0;record_pins={}
 for p in paths:
  rec=read(p);cid=rec['clip_id'];row=byid[cid];record_pins[str(p)]=sha(p)
  assert rec['context_sha256']==sha(U/'EVAL_CONTEXT.json') and rec['role']=='heldout_evaluation_only' and rec['training']==0
  assert all(rec[k]==row[k] for k in ['clip_id','signer_id','word_code','label_index','canonical_label'])
  check(rec['pins']);assert rec['source_sha256']==sha(rec['source_file'])
  assert rec['source_sha256'] not in ctx['prior_source_sha256'] and rec['source_sha256'] not in source_hashes;source_hashes.add(rec['source_sha256'])
  native=vb.Observation.load(rec['native_capture']['candidate']);cal=rec['calibration']
  method=vb.assert_calibration(native,int(native.sidecar['width']),int(native.sidecar['height']),read(B/'design_data_COEFFICIENT.json'),cal)
  outcomes={};original=io.load(rec['native_capture']['candidate'])
  for cap in ['native_capture','sensor_capture']:
   if cap not in rec:continue
   c=rec[cap];check(c['pins']);token=c['detector_instance'];assert token not in tokens;tokens.add(token)
   side=read(Path(c['baseline']).with_suffix('.sidecar.json'));assert side['detector_instance']==token and side['dropped_non_monotonic_pts']==0
   if cap=='sensor_capture':assert side['source_bbox_px']==c['box']==cal['box_source_pixels']
  if cal['status']!='CALIBRATED':
   for v in VIEWS:
    assert rec['views'][v]['channel_file'] is None;groups[v][cid]='input_unavailable';outcomes[v]={'status':rec['views'][v]['feature_info']['reason']}
  else:
   box=cal['box_source_pixels'];crop=io.load(rec['sensor_capture']['candidate']);io.same(original['timestamp_us'],crop['timestamp_us']);io.same(original['frame_id'],crop['frame_id'])
   masked=io.mask(original,box);sensor=io.mask(crop,box)
   for v,expected_obs in [('masked',masked),('sensor',sensor)]:
    info=rec['views'][v];actual_obs=io.load(info['observation']);io.validate(actual_obs)
    for k in expected_obs:io.same(expected_obs[k],actual_obs[k])
    expected,detail=io.feature_oracle(actual_obs,row);groups[v][cid]=gg.observation_group(actual_obs);outcomes[v]=detail
    if detail['status']=='OK':
     got=io.load(info['channel_file']);assert set(got)==set(expected)
     for k in expected:io.same(expected[k],got[k])
     values+=got['x_whole'].size
    else:assert info['channel_file'] is None and info['feature_info']['reason']
   hidden=original['observed']&~masked['observed'];altered={k:v.copy() for k,v in original.items()}
   altered['xy_px'][hidden]=[box[2]+500,box[3]+500];altered['conf'][hidden]=.95
   a,_=io.feature_oracle(masked,row);b,_=io.feature_oracle(io.mask(altered,box),row)
   for k in a:io.same(a[k],b[k])
   hidden_points+=int(hidden.sum())
  details.append({'clip_id':cid,'method':method,'outcomes':outcomes,'box':cal.get('box_source_pixels'),'frames':rec['frames']})
 out={'status':'PASS_EARLY_NOT_FULL' if early else 'PASS','at':now(),'records':len(paths),'requested':199,'matched_channel_values':values,'hidden_points_perturbed':hidden_points,'unique_detector_instances':len(tokens),'input_failures':{v:[x['clip_id'] for x in details if x['outcomes'][v]['status']!='OK'] for v in VIEWS},'record_pins':record_pins,'groups':groups,'group_counts':{v:dict(Counter(g.values())) for v,g in groups.items()},'details':details,'no_source_sha_duplicate_in_prior1199_or_test':True,'numeric_verification_only_not_semantic_correctness':True,'new_inference':0,'training':0,'verifier_pins':{str(Path(__file__)):sha(__file__),str(Path(io.__file__)):sha(io.__file__),str(Path(vb.__file__)):sha(vb.__file__),str(Path(gg.__file__)):sha(gg.__file__)}}
 write(U/('EARLY_INPUT_CHECK.json' if early else 'INPUTS_VERIFIED.json'),out,new=True)
 print(json.dumps({k:out[k] for k in ['status','records','matched_channel_values','hidden_points_perturbed','input_failures','group_counts']},ensure_ascii=False),flush=True)

if __name__=="__main__":inputs("--early" in sys.argv)
