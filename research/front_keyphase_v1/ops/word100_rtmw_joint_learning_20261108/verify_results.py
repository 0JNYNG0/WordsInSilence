"""Separate saved-output arithmetic and protection audit. No model inference."""
from common import *
import numpy as np,torch,av
def main():
 assert not (U/'FINAL_VERIFICATION.json').exists();checks=[]
 def check(name,value):
  checks.append(dict(name=name,passed=bool(value)))
  if not value:raise AssertionError(name)
 a=read(U/'SOURCE_AUDIT.json');f=read(U/'FROZEN.json');check('frozen sources',sha(U/'SOURCES.json')==f['source_manifest_sha256'])
 for p,h in {**a['protected'],**a['pins'],**f['pins']}.items():check('unchanged '+str(p),sha(p)==h)
 inputs=read(U/'INPUT_VERIFICATION.json');check('all input arithmetic passed',inputs['status']=='PASS' and inputs['clips']==1199)
 runtime=read(U/'RUNTIME_CHECK.json');check('runtime missing input contract',runtime['status']=='PASS')
 for p,h in runtime['pins'].items():check('runtime check pin '+p,sha(p)==h)
 tc=read(U/'TCN_TRAINING_COMPLETE.json');check('six TCN conditions',len(tc['runs'])==6)
 for run in tc['runs']:
  d=U/'tcn_runs'/f"{run['condition']}_{run['seed']}";check('TCN epoch count '+d.name,len(run['history'])==run['epochs']==80);check('TCN checkpoint '+d.name,sha(d/'model.pt')==run['checkpoint_sha256']);check('TCN weights changed '+d.name,run['weight_delta_l2']>0)
  for view,path in run['prediction_paths'].items():
   pred=read(path);check('300 denominators '+d.name+view,len(pred)==300)
   one=three=0
   for r in pred:
    ranks=[] if r['logits'] is None else sorted(range(100),key=lambda i:(-r['logits'][i],i))[:3]
    c1=bool(ranks and ranks[0]==r['y']);c3=r['y'] in ranks
    check('TCN row '+d.name+view+r['clip_id'],r['top1_correct']==c1 and r['top3_correct']==c3);one+=c1;three+=c3
   check('TCN totals '+d.name+view,(one,three)==(run['metrics'][view]['top1'],run['metrics'][view]['top3']))
 fb=read(U/'FEEDBACK_COMPARISON.json');fixed=read(U/'feedback_viewfix/FEEDBACK_COMPARISON.json')
 check('initial and corrected comparisons',len(fb['cases'])==len(fixed['cases'])==9)
 check('uniform framing fix verified',read(U/'feedback_viewfix/VIEW_FIX_VERIFICATION.json')['status']=='PASS')
 for case in fb['cases']+fixed['cases']:
  values=[]
  for j,m in enumerate(case['attributes']['targets']):
   samples=[r for r in case['label_rows'] if r['mask'][j]];pos=[r for r in samples if r['y'][j]>.5];neg=[r for r in samples if r['y'][j]<=.5]
   tp=sum(r['probability'][j]>=.5 for r in pos);tn=sum(r['probability'][j]<.5 for r in neg)
   check('feature tally '+case['condition']+str(case['seed'])+str(j),(len(samples),len(pos),len(neg),tp,tn)==(m['n'],m['positive'],m['negative'],m['tp'],m['tn']))
   if pos and neg:values.append((tp/len(pos)+tn/len(neg))/2)
  check('feature macro '+case['condition']+str(case['seed']),abs(sum(values)/len(values)-case['attributes']['macro_balanced_accuracy'])<1e-12)
  for label,key in [('direct_rows','direct_stage'),('dtw_rows','dtw_stage')]:
   rr=case[label];s=case[key];check('stage '+case['condition']+str(case['seed'])+label,len(rr)==s['n']==92 and sum(r['truth']==r['prediction'] for r in rr)==s['correct'])
 for r in read(U/'FEEDBACK_TRAINING_COMPLETE.json')['runs']+read(U/'feedback_viewfix/FEEDBACK_TRAINING_COMPLETE.json')['runs']:
  check('feedback epochs '+r['condition']+str(r['seed']),len(r['history'])==r['epochs']==120)
  check('feedback weight change '+r['condition']+str(r['seed']),r['weight_delta_l2']>0)
  if r['condition']=='pose_transfer':check('time layers copied',bool(r['transferred_keys']) and not any(k.startswith(('rgb.','update_')) for k in r['transferred_keys']))
 control=read(U/'FEEDBACK_CONTROLS.json');fixed_control=read(U/'feedback_viewfix/FEEDBACK_CONTROLS.json')
 check('two preserved 432 engineering control sets',len(control['rows'])==len(fixed_control['rows'])==432)
 for r in control['rows']+fixed_control['rows']:
  check('all frames in alignment '+r['condition']+str(r['seed'])+r['clip_id']+r['kind'],r['all_query_frames_in_alignment'])
  if r['kind']=='gap_30_frames':check('long missing retained as unknown',r['insufficient_hand_frames']>=30)
 upper=read(U/'feedback_viewfix/FEEDBACK_UPPER_VIEW.json')
 check('upper view six fixed diagnostic candidates',len(upper['rows'])==6)
 for r in upper['rows']:
  rr=r['stage_rows'];check('upper stage counts '+r['condition']+str(r['seed']),len(rr)==r['stage_total']==92 and sum(z['direct']==z['truth'] for z in rr)==r['stage_direct_correct'] and sum(z['dtw']==z['truth'] for z in rr)==r['stage_dtw_correct'])
  for p in r['predictions']:check('upper prediction hash '+p['clip_id']+r['condition']+str(r['seed']),sha(p['path'])==p['sha256'])
 breakdown=read(U/'FEEDBACK_BREAKDOWN.json');names=dict(zip(f['classes'],f['labels']))
 check('breakdown source',breakdown['source_sha256']==sha(U/'feedback_viewfix/FEEDBACK_COMPARISON.json'))
 for r in breakdown['rows']:
  check('word name '+r['word'],names[r['word']]==r['label'])
  cc=[c for c in fixed['cases'] if c['condition']==r['condition']];known=correct=direct=aligned=anchors=0
  for c in cc:
   for row in c['label_rows']:
    if r['word'] not in row['clip_id']:continue
    for k,m in enumerate(row['mask']):
     if m:known+=1;correct+=bool((row['probability'][k]>=.5)==(row['y'][k]>.5))
   direct+=sum(x['truth']==x['prediction'] for x in c['direct_rows'] if r['word'] in x['clip_id'])
   aligned+=sum(x['truth']==x['prediction'] for x in c['dtw_rows'] if r['word'] in x['clip_id'])
   anchors+=sum(r['word'] in x['clip_id'] for x in c['direct_rows'])
  check('word breakdown '+r['word']+r['condition'],(known,correct,direct,aligned,anchors)==(r['known_feature_n'],r['known_feature_correct'],r['direct_correct'],r['dtw_correct'],r['stage_n']))
 timing=read(U/'MODEL_TIMING.json');check('timing 12 pairs',len(timing['rows'])==12)
 for r in timing['rows']:check('timing saved result',sha(r['output'])==r['sha256'])
 stream=read(U/'TRAINED_STREAM_COMPLETE.json');source={r['clip_id']:r for r in read(U/'SOURCES.json')}
 stream_check=[]
 for r in stream['runs']:
  check('stream hash '+r['clip_id'],sha(r['raw_path'])==r['raw_sha256']);raw=np.load(r['raw_path']);rec=read(U/'records'/f"{r['clip_id']}.json");base=np.load(rec['raw_path'])
  check('stream count '+r['clip_id'],r['frames']==r['recorded_frames']==r['rtmw_frames']==len(raw['frame_id']))
  check('original PTS '+r['clip_id'],np.array_equal(raw['timestamp_us'],base['timestamp_us']) and np.array_equal(raw['frame_id'],base['frame_id']))
  check('same input pixels '+r['clip_id'],np.array_equal(raw['frame_sha256'],base['frame_sha256']))
  # Fixed model and identical input pixels: require numerical agreement and retain differences.
  maxxy=float(np.nanmax(np.abs(raw['xy_px']-base['xy_px'])));maxscore=float(np.nanmax(np.abs(raw['score']-base['score'])))
  check('RTMW numeric reuse agreement '+r['clip_id'],np.allclose(raw['xy_px'],base['xy_px'],atol=1e-4,rtol=1e-6,equal_nan=True) and np.allclose(raw['score'],base['score'],atol=1e-4,rtol=1e-6,equal_nan=True))
  pts=[]
  with av.open(r['recording']) as con:
   st=con.streams.video[0]
   for frame in con.decode(st):pts.append(round(float(frame.pts*st.time_base)*1e6))
  pts=np.asarray(pts,np.int64);pts-=pts[0]
  check('record PTS '+r['clip_id'],np.array_equal(pts,raw['timestamp_us']))
  check('no loss '+r['clip_id'],all(q['accepted'] and q['recorded'] for q in r['receipts']))
  for condition in r['conditions']:
   q=read(condition['output']);check('research uncertainty '+r['clip_id']+condition['condition'],q['normal_or_wrong'] is None and not q['educational_correctness_validated'])
   check('same extracted input only',q['reference_video_reads']==0 and q['rgb_forward_calls']==0 and not q['word_classifier_uses_requested_target'])
   tcname=condition['condition'].split('+')[0];saved=read(U/'tcn_runs'/f'{tcname}_20261013'/'PREDICTIONS_full.json');row=next(x for x in saved if x['clip_id']==r['clip_id'])
   expected=[f['classes'][i] for i in row['top3']]
   check('stream TCN ranks match fixed full-view evaluation '+r['clip_id']+tcname,[x['word'] for x in q['tcn_top3']]==expected)
  stream_check.append(dict(clip_id=r['clip_id'],frames=r['frames'],max_xy_difference_px=maxxy,max_raw_score_difference=maxscore,recording_PTS_exact=True))
 write(U/'FINAL_VERIFICATION.json',dict(status='PASS',checks=checks,checks_total=len(checks),stream=stream_check,
  provenance='Same Codex independent saved-array arithmetic and source-pin checking; not independent human educational ground truth',
  original_models_unchanged=True,private_inputs=0,model_reinference=0))
 event('final_verification_pass',checks=len(checks))
if __name__=='__main__':main()
