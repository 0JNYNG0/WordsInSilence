"""Separate saved observation/channel and score arithmetic, no detector/training/inference."""
from common import *
from collections import Counter
import math,importlib.util

def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def inputs(early=False):
 ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins'])
 if not early:assert not (U/'PREP_RUNNING.json').exists() and read(U/'PREP_COMPLETE.json')['records']==200
 paths=sorted((DATA/'records').glob('*.json'));assert len(paths)==200 or early
 if early:paths=paths[:2]
 assert paths
 sys.path.insert(0,str(B));import verify_boundary as vb
 io=vb.input_oracle_module();gg=mod('heldout_observation_groups',B/'verify_full_inputs.py');gg.group_selfcheck()
 byid={r['clip_id']:r for r in ctx['rows']};assert len(byid)==200
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
 out={'status':'PASS_EARLY_NOT_FULL' if early else 'PASS','at':now(),'records':len(paths),'requested':200,'matched_channel_values':values,'hidden_points_perturbed':hidden_points,'unique_detector_instances':len(tokens),'input_failures':{v:[x['clip_id'] for x in details if x['outcomes'][v]['status']!='OK'] for v in VIEWS},'record_pins':record_pins,'groups':groups,'group_counts':{v:dict(Counter(g.values())) for v,g in groups.items()},'details':details,'no_source_sha_duplicate_in_prior1199_or_test':True,'numeric_verification_only_not_semantic_correctness':True,'new_inference':0,'training':0,'verifier_pins':{str(Path(__file__)):sha(__file__),str(Path(io.__file__)):sha(io.__file__),str(Path(vb.__file__)):sha(vb.__file__),str(Path(gg.__file__)):sha(gg.__file__)}}
 write(U/('EARLY_INPUT_CHECK.json' if early else 'INPUTS_VERIFIED.json'),out,new=True)
 print(json.dumps({k:out[k] for k in ['status','records','matched_channel_values','hidden_points_perturbed','input_failures','group_counts']},ensure_ascii=False),flush=True)

def recount(scores):
 assert len(scores)==200 and len({x['clip_id'] for x in scores})==200
 assert Counter(x['signer_id'] for x in scores)=={17:100,18:100} and Counter(x['label_index'] for x in scores)=={i:2 for i in range(100)}
 for x in scores:
  if x['status']=='OK':
   a=x['logits'];assert len(a)==100 and all(map(math.isfinite,a));order=sorted(range(100),key=lambda i:(-a[i],i));m=max(a);ce=m+math.log(sum(math.exp(y-m) for y in a))-a[x['label_index']]
   assert order[0]==x['prediction'] and order[:5]==x['top5']
   assert x['correct']==(order[0]==x['label_index']) and x['top5_correct']==(x['label_index'] in order[:5])
  else:assert x['logits'] is None and not x['correct'] and not x['top5_correct'];ce=math.log(100)
  assert abs(ce-x['cross_entropy'])<1e-10
 return {'correct':sum(x['correct'] for x in scores),'top5':sum(x['top5_correct'] for x in scores),'failures':sum(x['status']!='OK' for x in scores),'ce':sum(x['cross_entropy'] for x in scores)/200}

def scores():
 ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins']);assert not (U/'EVAL_RUNNING.json').exists();vi=read(U/'INPUTS_VERIFIED.json');check(vi['record_pins'])
 expected_rows={x['clip_id']:x for x in ctx['rows']}
 allscores={};metric={};pins={};pairs={};classes=[]
 for arm in ['C','T3F']:
  metric[arm]={v:[] for v in VIEWS}
  for seed in SEEDS:
   p=U/f'{arm}_{seed}_SCORES.json';r=read(p);pins[str(p)]=sha(p)
   assert r['context_sha256']==sha(U/'EVAL_CONTEXT.json') and r['checkpoint']==ctx['models'][arm][str(seed)]
   assert r['model_state_before']==r['model_state_after'] and r['training']==0 and not r['checkpoint_reselected']
   assert r['epoch']==ctx['models'][arm][str(seed)]['selected_epoch'] and r['input_groups']==vi['groups']
   for v in VIEWS:
    rr=r['predictions'][v];assert {x['clip_id'] for x in rr}==set(expected_rows)
    for x in rr:
     e=expected_rows[x['clip_id']]
     assert (x['signer_id'],x['label_index'])==(e['signer_id'],e['label_index'])
    m=recount(rr);m['signers']={str(s):sum(x['correct'] for x in rr if x['signer_id']==s) for s in [17,18]}
    m['groups']={g:{'total':sum(vi['groups'][v][x['clip_id']]==g for x in rr),'correct':sum(x['correct'] and vi['groups'][v][x['clip_id']]==g for x in rr)} for g in sorted(set(vi['groups'][v].values()))}
    assert {x['clip_id'] for x in rr if x['status']!='OK'}==set(vi['input_failures'][v]);metric[arm][v].append(m);allscores[arm,seed,v]=rr
 for v in VIEWS:
  pairs[v]=[]
  for seed in SEEDS:
   aa={x['clip_id']:x for x in allscores['C',seed,v]};bb={x['clip_id']:x for x in allscores['T3F',seed,v]};assert aa.keys()==bb.keys()
   changed=[]
   for cid,a in aa.items():
    b=bb[cid];assert (a['signer_id'],a['label_index'])==(b['signer_id'],b['label_index'])
    if a['correct']!=b['correct']:changed.append({'clip_id':cid,'label':ctx['labels'][a['label_index']],'signer_id':a['signer_id'],'kind':'gain' if b['correct'] else 'loss','C_prediction':ctx['labels'][a['prediction']] if a['prediction'] is not None else None,'T3F_prediction':ctx['labels'][b['prediction']] if b['prediction'] is not None else None})
   gain=sum(x['kind']=='gain' for x in changed);loss=len(changed)-gain;pairs[v].append({'seed':seed,'gains':gain,'losses':loss,'delta_pp':(gain-loss)/2,'changes':changed})
 for i,label in enumerate(ctx['labels']):
  x={'index':i,'label':label}
  for v in VIEWS:x[v]={a:sum(s['correct'] for seed in SEEDS for s in allscores[a,seed,v] if s['label_index']==i) for a in ['C','T3F']}
  classes.append(x)
 summary={a:{v:{'counts':[x['correct'] for x in ms],'top5_counts':[x['top5'] for x in ms],'mean':sum(x['correct'] for x in ms)/6,'range':[min(x['correct'] for x in ms)/2,max(x['correct'] for x in ms)/2],'top5_mean':sum(x['top5'] for x in ms)/6} for v,ms in vm.items()} for a,vm in metric.items()}
 sensor=[p['delta_pp'] for p in pairs['sensor']];checks={'sensor_mean_higher':sum(sensor)>0,'two_paired_seeds_improve':sum(d>0 for d in sensor)>=2,'masked_mean_preserved':summary['T3F']['masked']['mean']>=summary['C']['masked']['mean'],'sensor_top5_preserved':summary['T3F']['sensor']['top5_mean']>=summary['C']['sensor']['top5_mean']}
 for v in VIEWS:
  checks[f'{v}_top5_preserved']=summary['T3F'][v]['top5_mean']>=summary['C'][v]['top5_mean']
  for s in ['17','18']:checks[f'{v}_signer_{s}']=sum(x['signers'][s] for x in metric['T3F'][v])>=sum(x['signers'][s] for x in metric['C'][v])
  for g in vi['group_counts'][v]:checks[f'{v}_group_{g}']=sum(x['groups'][g]['correct'] for x in metric['T3F'][v])>=sum(x['groups'][g]['correct'] for x in metric['C'][v])
 remaining=[]
 for seed in SEEDS:
  for v in VIEWS:
   for x in allscores['T3F',seed,v]:
    if not x['correct']:remaining.append({'seed':seed,'view':v,'clip_id':x['clip_id'],'signer_id':x['signer_id'],'label':ctx['labels'][x['label_index']],'prediction':ctx['labels'][x['prediction']] if x['prediction'] is not None else None,'status':x['status'],'group':vi['groups'][v][x['clip_id']]})
 out={'status':'PASS_SAVED_SCORE_RECOUNT','at':now(),'summary':summary,'metrics':metric,'pairs':pairs,'all_classes':classes,'remaining_errors':remaining,'primary_improvement_maintained':checks['sensor_mean_higher'] and checks['two_paired_seeds_improve'],'stronger_consistency_checks':checks,'stronger_consistency_pass':all(checks.values()),'unique_videos':200,'signer_ids':[17,18],'seeds':SEEDS,'predictions_verified':2400,'pins':pins,'new_inference':0,'training':0,'test_now_consumed':True,'statistical_significance_claimed':False,'automatic_adoption':False,'verifier_sha256':sha(__file__)}
 write(U/'COMPARISON_INDEPENDENT.json',out,new=True)
 print(json.dumps({k:out[k] for k in ['status','summary','primary_improvement_maintained','stronger_consistency_checks']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 if '--scores' in sys.argv:scores()
 else:inputs('--early' in sys.argv)
