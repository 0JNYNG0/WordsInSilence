"""Read-only arithmetic and media verification, separate from implementation."""
from task import *
import numpy as np,av
def main():
 checks=[]
 def check(name,value,detail=None):
  checks.append(dict(name=name,pass_=bool(value),detail=detail))
 f=read(U/'FROZEN.json')
 for p,h in {**f['protected'],**f['source_pins']}.items():check('protected '+str(p),sha(p)==h)
 for pinfile in ['TRAIN_CODE_PINS.json','EVALUATION_CODE_PINS.json']:
  for p,h in read(U/pinfile).items():check('frozen code '+str(p),sha(p)==h)
 prep=read(U/'PREPARED.json');fit=[r for r in prep['rows'] if r['role']=='fit'];dev=[r for r in prep['rows'] if r['role']=='development'];check('36FIT12DEV',len(fit)==36 and len(dev)==12);check('recorded ID split',set(r['person'] for r in fit)==set(range(1,10)) and set(r['person'] for r in dev)=={11,12,13})
 for r in prep['rows']:
  check('input hash '+r['clip_id'],sha(r['path'])==r['sha256']);a=np.load(r['path']);check('input monotonic '+r['clip_id'],np.all(np.diff(a['timestamp_us'])>0) and np.all(np.diff(a['frame_id'])>0))
  if r['role']=='development':check('no DEV training-stage labels '+r['clip_id'],np.all(a['stage_y']==-1))
 tr=read(U/'TRAINING_COMPLETE.json')
 for r in tr['runs']:
  check('head checkpoint '+str(r['seed']),sha(U/'runs'/str(r['seed'])/'model.pt')==r['model_sha256']);check('100epochs positive update '+str(r['seed']),r['epochs']==100 and r['weight_delta']>0)
 ev=read(U/'EVALUATION.json')
 for seed in [42,43,44]:
  rs=[r for r in ev['rows'] if r['seed']==seed];check('84 comparisons '+str(seed),len(rs)==84)
  for r in rs:check('output hash '+r['clip_id']+r['kind']+str(seed),sha(r['output'])==r['sha256'])
  fs=[r for r in ev['feature_pairs'] if r['seed']==seed];acc=sum((r['probability']>=.5)==bool(r['truth']) for r in fs)/len(fs);check('feature pair arithmetic '+str(seed),len(fs)==388 and abs(acc-ev['summary'][str(seed)]['feature_pair_accuracy'])<1e-12)
 scope=read(U/'CONTROL_SCOPE_AUDIT.json');check('do not call all half-cuts true semantic errors',scope['first_gate_not_valid_as_all_word_semantic_error_accuracy'])
 total=0;largest_pts_error=0;largest_joint_delta=0
 for r in read(U/'PARALLEL_COMPLETE.json')['runs']:
  d=U/'parallel'/r['word'];s=read(d/'rtmw/SESSION.json');z=np.load(d/'rtmw/observations.npz');seq=np.load(U/'examples'/(r['word']+'_normal')/'observations.npz');total+=len(z['frame_id']);check('no replay loss '+r['word'],r['input_complete'] and len(z['frame_id'])==r['frames']==r['recorded_frames']==r['rtmw_frames']);check('once only RTMW '+r['word'],s['detector_calls']==r['frames'] and r['second_pass_detector_calls']==0)
  for i,record in enumerate(r['records']):check('frame pixels and PTS '+r['word']+str(i),z['frame_sha256'][i]==record['hash'] and int(z['timestamp_us'][i])==record['timestamp_us'])
  with av.open(str(d/'video.mp4')) as video:pts=np.array([round(float(fr.pts*fr.time_base)*1e6) for fr in video.decode(video=0)])
  delta=int(np.max(np.abs(pts-z['timestamp_us']))) if len(pts)==len(z['timestamp_us']) else -1;largest_pts_error=max(largest_pts_error,delta);check('encoded PTS '+r['word'],0<=delta<=2,delta)
  diff=float(np.nanmax(np.abs(z['xy_px']-seq['xy_px'])));largest_joint_delta=max(largest_joint_delta,diff);check('same RTMW coordinates across concurrency '+r['word'],np.allclose(z['xy_px'],seq['xy_px'],atol=1e-5,rtol=0,equal_nan=True),diff)
  result=read(d/'result.json');check('single pattern, no RGB reference inference '+r['word'],result['rgb_model_calls']==0 and result['reference_video_reads']==0 and result['normal_or_wrong'] is None and not result['overall_pass'])
 check('505 frames total',total==505)
 check('capture contracts',read(U/'CAPTURE_CONTRACT_TEST.json')['pass_']);check('browser22',read(U/'BROWSER_VERIFIED.json')['status']=='PASS' and read(U/'BROWSER_VERIFIED.json')['count']==22)
 decision=read(U/'RELEASE_DECISION.json');check('automatic correctness remains disabled',not decision['automatic_educational_correction_enabled'] and not decision['automatic_pass_enabled']);check('old deployment not edited',all(sha(p)==h for p,h in f['protected'].items()))
 write(U/'FINAL_VERIFICATION.json',dict(status='PASS' if all(x['pass_'] for x in checks) else 'FAIL',checks=len(checks),rows=checks,total_replayed_frames=total,max_video_pts_error_us=largest_pts_error,max_joint_delta_px=largest_joint_delta,verifier='Same Codex separate saved-output/media verification; not independent human educational validation',real_camera_test=False));print('PASS' if all(x['pass_'] for x in checks) else 'FAIL',len(checks),[x['name'] for x in checks if not x['pass_']]);assert all(x['pass_'] for x in checks)
if __name__=='__main__':main()
