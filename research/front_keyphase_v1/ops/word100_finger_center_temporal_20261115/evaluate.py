from runtime import *
import time
role=sys.argv[1];version=sys.argv[2]if len(sys.argv)>2 else'v1';prefix='v2_'if version=='v2'else'';modes=['base','dual_center_only','dual_change_override','dual_change_fill']if version=='v2'else['base','center_only','change_override','change_fill'];assert role in ['historic','diagnostic','confirmation','previous_diagnostic','previous_confirmation'];assert not(U/f'{prefix}{role}_RESULTS.json').exists()
if role=='confirmation':
 for p,h in read(U/'CANDIDATE_FROZEN.json')['pins'].items():assert sha(p)==h
torch.set_num_threads(2)
if version=='v2':
 from runtime_v2 import DualCenterObserver
 model=DualCenterObserver()
else:model=CenterTemporalObserver()
cache={};rows=[];timings=[]
if role=='historic':
 extras={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']};data=np.load(P/'INPUTS.npz')
 for i,r in enumerate(read(P/'INPUTS.json')['rows']):
  if r['role']!='development':continue
  path=extras[r['clip_id']]['channels']if r['clip_id']in extras else read(J/'records'/f'{r["clip_id"]}.json')['channel_path'];rows.append(r|dict(hand=r['side'],shape=data['y'][i],phase=[-1]*4,quality=[-1]*4,channel_path=path,original_index=i))
elif role.startswith('previous_'):
 desired='diagnostic'if role=='previous_diagnostic'else'confirmation_reserved';rows=[r|dict(phase=[-1]*4)for r in read(N/'LABELS.json')['rows']if r['role']==desired]
 for r in rows:
  if r['clip_id']=='NIA_SL_WORD2748_REAL10_F'and r['frame']==61:r['shape'][0]=0 # prior source-audit, never training
else:rows=[r for r in read(U/'LABELS.json')['rows']if r['role']==role]
results=[]
for r in rows:
 key=r['channel_path'],r['hand']
 if key not in cache:
  x,fid=load_channels(r['channel_path']);t=time.perf_counter();cache[key]=(model.all(x,r['hand']),fid);timings.append(time.perf_counter()-t)
 z,fid=cache[key];j=int(np.flatnonzero(fid==r['frame'])[0]);results.append(r|dict(predictions={k:z[k][j]for k in modes+['phase','available']},probabilities={k:v[:,j]for k,v in z['probabilities'].items()}))
y=np.array([r['shape']for r in results]);a=np.array([r['predictions']['available']for r in results]);truthphase=np.array([r['phase']for r in results]);q=np.array([r['quality']for r in results]);mask=(y>=0)&a;base=np.array([r['predictions']['base']for r in results]);oldcorrect=(base==y)&mask;metrics={};changes=(truthphase==1)&(y==1)&a
for mode in modes:
 p=np.array([r['predictions'][mode]for r in results]);metrics[mode]=dict(known=int(mask.sum()),correct=int(((p==y)&mask).sum()),wrong=int(((p>=0)&(p!=y)&mask).sum()),unknown=int(((p<0)&mask).sum()),lost_previous_correct=int(((p!=y)&oldcorrect).sum()),new_correct=int(((p==y)&mask&~oldcorrect).sum()),new_confident_errors=int(((p>=0)&(p!=y)&mask&~((base>=0)&(base!=y))).sum()),extended_during_change_support=int(changes.sum()),extended_during_change_correct=int(((p==1)&changes).sum()),wrong_on_bad_chain=int(((p>=0)&(p!=y)&mask&(q==0)).sum()))
pp=np.array([r['predictions']['phase']for r in results]);pm=(truthphase>=0)&a;phase_metrics=dict(known=int(pm.sum()),correct=int(((pp==truthphase)&pm).sum()),wrong=int(((pp>=0)&(pp!=truthphase)&pm).sum()),unknown=int(((pp<0)&pm).sum()),changes=int(((truthphase==1)&a).sum()),changes_found=int(((truthphase==1)&(pp==1)&a).sum()),stable_false_change=int(((truthphase==0)&(pp==1)&a).sum()))
write(U/f'{prefix}{role}_RESULTS.json',dict(metrics=metrics,phase_metrics=phase_metrics,rows=results,timing=dict(research_all_variants_median_seconds=float(np.median(timings)),scope='Old9models plus new branches full sequence CPU; not full webcam latency')));print(json.dumps(dict(role=role,version=version,metrics=metrics,phase=phase_metrics)))
