import run as R
import numpy as np
from collections import Counter
def number(actual,expected):
 if expected is None:assert actual is None
 else:assert actual is not None and np.isfinite(actual) and abs(actual-expected)<=1e-10*max(1.,abs(expected))
def mean(arr):
 a=np.asarray(arr,dtype=np.float64);valid=np.isfinite(a);return float(np.sum(np.where(valid,a,0.),dtype=np.float64)/valid.sum()) if valid.any() else None
def intervals(rows,mask,times):
 rebuilt=np.zeros(len(mask),bool);last=-2
 for r in rows:
  a=r['first'];b=r['last'];assert isinstance(a,int) and isinstance(b,int) and 0<=a<=b<len(mask) and a>last+1
  assert r['samples']==b-a+1 and r['reference_start_s']==float(times[a]/1e6) and r['reference_end_s']==float(times[b]/1e6)
  rebuilt[a:b+1]=True;last=b
 assert np.array_equal(rebuilt,mask)
def main():
 f=R.read(R.U/'FROZEN.json');r=R.read(R.U/'RESULT.json');R.check(f['pins']);R.check(r['pins']);R.check(f['protected']);assert len(r['rows'])==4500 and not(R.U/'RUNNING.json').exists()
 inputs={x['clip_id']:x for x in f['input_rows']};scores=R.prediction_rows(f);count=Counter();ranks=Counter();groupnames=['finger_angles','palm_position','elbow_position','pose_wrist_position','hand_wrist_position'];bounds=[(0,15),(15,17),(17,19),(19,21),(21,23)]
 for ri,row in enumerate(r['rows']):
  receipt=R.read(row['receipt']);z=R.load(R.B/'cases'/(row['name']+'.npz'));q=R.load(R.Q/'cases'/(row['name']+'.npz'));original=R.load(inputs[row['clip_id']]['output']);time=original['timestamp_us'];domain=z['domain'];po=z['point_ok'];so=z['sequence_new_ok'];quality=q['sequence_quality'];pd=z['point_delta'];sd=q['sequence_delta'];src=q['source']
  assert receipt['query_id']==row['name'] and receipt['reference_id']==row['clip_id'] and receipt['condition']==row['kind'];assert receipt['whole_gesture_correct'] is None and receipt['skill_score'] is None and receipt['validated_correction'] is None
  assert receipt['research_source']['reference_type']=='same_source_control' and receipt['research_source']['split']=='development_only'
  if row['kind']=='identity':
   assert receipt['recognition']=={k:v[row['clip_id']] for k,v in scores.items()}
   for model in scores:ranks[model+'_correct']+=receipt['recognition'][model]['correct'];ranks[model+'_top5']+=receipt['recognition'][model]['top5_correct']
   count['joined_original_predictions']+=1
  else:assert receipt['recognition'] is None
  for h in (0,1):
   hand=receipt['hands'][h];assert hand['side']==('L','R')[h] and len(hand['angles'])==15
   for gi,(a,b) in enumerate(bounds):
    g=hand['groups'][gi];assert g['group']==groupnames[gi] and g['unit']==('projected_degrees' if gi==0 else 'shoulder_width')
    ex=domain[:,h,a:b];p=po[:,h,a:b];s=so[:,h,a:b];qq=quality[:,h,a:b];point=pd[:,h,a:b];seq=sd[:,h,a:b];source=src[:,h,a:b]
    counts=[int(ex.sum()),int(p.sum()),int(s.sum()),int(qq.sum())]
    for key,val in zip(['audit_cells','point_compared_cells','sequence_compared_cells','corroborated_sequence_cells'],counts):assert g[key]==val
    for key,num in zip(['point_coverage','sequence_coverage','corroborated_coverage'],counts[1:]):number(g[key],num/counts[0] if counts[0] else None)
    status=('outside_audit_scope' if counts[0]==0 else 'no_comparable_evidence' if counts[2]==0 else 'partial_evidence' if counts[2]<counts[0] else 'computable_not_fully_corroborated' if counts[3]<counts[2] else 'corroboration_available');assert g['evidence_state']==status
    for key,val in [('point_mean_difference',point[p]),('sequence_level_mean_difference',seq[s,:3]),('sequence_change_mean_difference',seq[s,3:]),('corroborated_level_mean_difference',seq[qq,:3]),('corroborated_change_mean_difference',seq[qq,3:])]:number(g[key],mean(val));count['scalar_differences_verified']+=1
    assert g['source_cells']==dict(refined=int((s&(source==1)).sum()),retained_original=int((s&(source==2)).sum()))
    for key,mask in [('missing_point_intervals',(ex&~p).any(-1)),('sequence_context_unavailable_intervals',(ex&p&~s).any(-1)),('uncorroborated_intervals',(s&~qq).any(-1)),('reference_missing_intervals',(z['missing_original'][:,h,a:b]&ex).any(-1))]:intervals(g[key],mask,time);count['interval_tracks_verified']+=1
    tl=g['timeline'];assert tl['reference_seconds']==(time/1e6).tolist()
    for key,arr in [('audit_channels',ex),('point_channels',p),('sequence_channels',s),('corroborated_channels',qq)]:assert tl[key]==arr.sum(-1).tolist()
    for j in range(len(time)):
     number(tl['point_difference'][j],mean(point[j,p[j]]));number(tl['sequence_level_difference'][j],mean(seq[j,s[j],:3]));number(tl['sequence_change_difference'][j],mean(seq[j,s[j],3:]));count['timeline_values_verified']+=3
    assert g['gesture_correct'] is None and g['correction_direction'] is None
    count['audit_cells']+=counts[0];count['point_cells']+=counts[1];count['sequence_cells']+=counts[2];count['quality_cells']+=counts[3]
   for a,angle in enumerate(hand['angles']):
    ex=domain[:,h,a];p=po[:,h,a];s=so[:,h,a];qq=quality[:,h,a]
    assert angle['angle_index']==a
    for key,arr in [('audit_cells',ex),('point_compared_cells',p),('sequence_compared_cells',s),('corroborated_cells',qq)]:assert angle[key]==int(arr.sum())
    number(angle['point_mean_degrees'],mean(pd[:,h,a][p]));number(angle['sequence_level_mean_degrees'],mean(sd[:,h,a,:3][s]));number(angle['sequence_change_mean_degrees'],mean(sd[:,h,a,3:][s]))
    for key,mask in [('missing_point_intervals',ex&~p),('sequence_context_unavailable_intervals',ex&p&~s),('uncorroborated_intervals',s&~qq)]:intervals(angle[key],mask,time);count['interval_tracks_verified']+=1
   if row['kind']=='whole'+('L','R')[h]:
    for gi in (0,1,4):
     g=hand['groups'][gi];assert g['point_compared_cells']==0 and g['sequence_compared_cells']==0 and g['point_mean_difference'] is None and g['sequence_level_mean_difference'] is None
    # Arms must still match stored body observations, independently of hand absence.
    assert hand['groups'][2]['point_compared_cells']==int(po[:,h,17:19].sum());count['wholehand_controls_verified']+=1
  count['cases']+=1
  if (ri+1)%500==0:print('VERIFIED_RECEIPTS',ri+1,4500,flush=True)
 qv=R.read(R.Q/'VERIFIED.json');assert count['sequence_cells']==qv['counts']['coordinate_sequence_cells']==6614784;assert count['quality_cells']==qv['counts']['quality_after']==2657822;assert count['wholehand_controls_verified']==600 and count['joined_original_predictions']==300
 assert ranks['C_correct']==294 and ranks['T3F_correct']==295
 R.check(f['pins']);R.check(r['pins']);R.check(f['protected']);R.write(R.U/'VERIFIED.json',dict(at=R.now(),status='PASS_ALL4500_CONSUMER_RECEIPTS',counts=dict(count),saved_original_masked_ranks=dict(ranks),original_predictions_unchanged=True,input_scope='Only public developedDEV300 plus synthetic variants;self-reference audit,not new real-user or cross-person accuracy.',method='Separate NumPy sum/count and interval reconstruction all4500;14 malformed/missing/locality fixtures. Same Codex arithmetic,not independent anatomical or semantic truth.',implementation_complete=True,semantic_feedback_validated=False,training=0,inference=0,production_changed=False,pins={str(R.U/n):R.sha(R.U/n) for n in ['consumer.py','run.py','verify.py','RESULT.json','FROZEN.json']}));print('VERIFIED_CONSUMER4500',dict(count),dict(ranks),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  p=R.U/'VERIFY_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
