from split_runtime import *
import time
role=sys.argv[1];assert role in ['diagnostic','historic','confirmation_reserved']
if role=='confirmation_reserved':
 frozen=read(U/'CANDIDATE_FROZEN.json')
 for p,h in frozen['pins'].items():assert sha(p)==h
torch.set_num_threads(2);model=SplitObserver();rows=[];cache={};elapsed=[];data=np.load(P/'INPUTS.npz')
if role=='historic':
 extras={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']}
 for i,r in enumerate(read(P/'INPUTS.json')['rows']):
  if r['role']!='development':continue
  path=extras[r['clip_id']]['channels']if r['clip_id']in extras else read(J/'records'/f"{r['clip_id']}.json")['channel_path'];rows.append(r|dict(hand=r['side'],shape=data['y'][i],quality=[-1]*4,channel_path=path,original_index=i))
else:rows=[r for r in read(U/'LABELS.json')['rows']if r['role']==role]
results=[]
for r in rows:
 key=r['channel_path'],r['hand']
 if key not in cache:
  x,fid=load_channels(r['channel_path']);t=time.perf_counter();cache[key]=(model.all(x,r['hand']),fid);elapsed.append(time.perf_counter()-t)
 z,fid=cache[key];j=int(np.flatnonzero(fid==r['frame'])[0]);results.append(r|dict(predictions={k:z[k][j]for k in ['base','shape_only','compatible_shape','quality_only','combined','quality','available']},probabilities={k:z['probabilities'][k][:,j]for k in ['shape','quality']}))
y=np.array([r['shape']for r in results]);q=np.array([r['quality']for r in results]);av=np.array([r['predictions']['available']for r in results]);mask=(y>=0)&av;base=np.array([r['predictions']['base']for r in results]);oldcorrect=(base==y)&mask;metrics={}
for kind in ['base','shape_only','compatible_shape','quality_only','combined']:
 p=np.array([r['predictions'][kind]for r in results]);metrics[kind]=dict(known=int(mask.sum()),correct=int(((p==y)&mask).sum()),wrong=int(((p>=0)&(p!=y)&mask).sum()),unknown=int(((p<0)&mask).sum()),lost_previous_correct=int(((p!=y)&oldcorrect).sum()),new_correct=int(((p==y)&mask&~oldcorrect).sum()),old_correct_to_wrong=int(((p>=0)&(p!=y)&oldcorrect).sum()),errors_on_good_chain=int(((p>=0)&(p!=y)&mask&(q==1)).sum()),errors_on_bad_chain=int(((p>=0)&(p!=y)&mask&(q==0)).sum()))
qp=np.array([r['predictions']['quality']for r in results]);qstats=dict(good_support=int(((q==1)&av).sum()),bad_support=int(((q==0)&av).sum()),good_false_bad=int(((qp==0)&(q==1)&av).sum()),bad_detected=int(((qp==0)&(q==0)&av).sum()),bad_false_good=int(((qp==1)&(q==0)&av).sum()))
write(U/f'{role}_RESULTS.json',dict(metrics=metrics,quality=qstats,rows=results,timing=dict(full_hand_sequence_seconds_median=float(np.median(elapsed)),scope='all research candidates plus old9models; excludesRTMW/baseGRU/DTW/web; not deployed latency')))
print(json.dumps(dict(role=role,metrics=metrics,quality=qstats)))
