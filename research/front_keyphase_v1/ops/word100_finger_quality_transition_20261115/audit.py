"""Independent artifact/calculation pass by the same implementer, not expert review."""
from quality_common import *
import ast,collections
checks=[]
def check(ok,msg):
 if not ok:raise AssertionError(msg)
 checks.append(msg)
for p,h in read(U/'PROTECTED.json').items():check(sha(p)==h,'protected:'+p)
for cfg in ['TRAIN_FROZEN.json','TRAIN_FROZEN_V2.json','CANDIDATE_FROZEN.json']:
 for p,h in read(U/cfg)['pins'].items():check(sha(p)==h,cfg+':'+p)
for trained in ['TRAINED.json','TRAINED_V2.json']:
 for r in read(U/trained)['runs']:check(sha(r['model'])==r['sha256'],'checkpoint:'+r['model'])
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'));checks.append('syntax:'+p.name)
sets={};files={};counts={}
for name in ['FIT_LABELS_V2.json','DIAGNOSTIC_LABELS.json','CONFIRMATION_LABELS.json']:
 rows=read(U/name)['rows'];sets[name]=set();seen=set()
 for r in rows:
  key=r['clip_id'],r['hand'],r['frame'];check(key not in seen,'unique:'+str(key));seen.add(key);sets[name].add(r['clip_id'])
  allowed=[1,3,4,6,8,9]if name.startswith('FIT')else[14]if name.startswith('DIAG')else[15]
  check(r['person']in allowed,'role:'+str(key))
  for k,h in [('path','source_sha256'),('raw_path','raw_sha256'),('channel_path','channel_sha256')]:
   if r[k]not in files:files[r[k]]=sha(r[k])
   check(files[r[k]]==r[h],'input:'+r[k])
  check(Path(r['image']).exists(),'review:'+str(key));check(r['neighbor_frames']==[max(0,r['frame']-3),r['frame'],r['frame']+3],'neighborhood:'+str(key))
  for k in ['quality','phase']+(['posture']if 'posture'in r else[]):check(len(r[k])==4 and set(r[k])<={-1,0,1},'label:'+k+str(key))
  z=np.load(r['channel_path']);check(np.array_equal(z['frame_id'],np.arange(len(z['frame_id']))),'source-frame-contiguous:'+r['clip_id'])
 q=np.array([r['quality']for r in rows]);p=np.array([r['phase']for r in rows]);counts[name]=dict(anchors=len(rows),videos=len(sets[name]),people=sorted(set(r['person']for r in rows)),words=sorted(set(r['word']for r in rows)),quality_known=int((q>=0).sum()),quality_bad=int((q==0).sum()),phase_known=int((p>=0).sum()),phase_change=int((p==1).sum()),cooccurring_bad_and_change=int(((q==0)&(p==1)).sum()))
names=list(sets)
for i,a in enumerate(names):
 for b in names[i+1:]:check(not sets[a]&sets[b],'clip_split:'+a+b)
change_cases=[];errors=[];head_errors=[];byhand={}
for role in ['diagnostic','confirmation_reserved']:
 d=read(U/f'{role}_RESULTS_V2.json');y=np.array([r['posture']for r in d['rows']]);b=np.array([r['old']for r in d['rows']]);k=y>=0
 for mode in ['old','quality_only','phase_only','combined']:
  p=np.array([r[mode]for r in d['rows']]);s=d['summary'][mode]
  for field,val in dict(known=int(k.sum()),correct=int(((p==y)&k).sum()),wrong=int(((p>=0)&(p!=y)&k).sum()),unknown=int(((p<0)&k).sum()),old_correct_lost=int(((b==y)&(p<0)&k).sum()),old_correct_to_wrong=int(((b==y)&(p>=0)&(p!=y)&k).sum())).items():check(s[field]==val,f'{role}:{mode}:{field}')
  check(np.all((p<0)|(p==b)),'no_new_confident_state:'+role+mode)
 for r in d['rows']:
  for j in range(4):
   rec=dict(role=role,clip_id=r['clip_id'],hand=r['hand'],frame=r['frame'],finger=['index','middle','ring','little'][j],truth=r['posture'][j],old=r['old'][j],selected=r['quality_only'][j],quality_truth=r['quality'][j],phase_truth=r['phase'][j],heads=r['quality_phase'][j])
   if rec['truth']>=0 and rec['old']>=0 and rec['old']!=rec['truth']:errors.append(rec)
   if rec['old']!=rec['selected']:change_cases.append(rec)
   if r['phase'][j]==0 and r['quality_phase'][j][1]==1:head_errors.append(rec)
 for hand in ['L','R']:
  rr=[r for r in d['rows']if r['hand']==hand];yy=np.array([r['posture']for r in rr]);kk=yy>=0
  byhand[role+'_'+hand]={}
  for mode in ['old','quality_only']:
   pp=np.array([r[mode]for r in rr]);byhand[role+'_'+hand][mode]=dict(known=int(kk.sum()),correct=int(((yy==pp)&kk).sum()),wrong=int(((pp>=0)&(yy!=pp)&kk).sum()),unknown=int(((pp<0)&kk).sum()))
check(read(U/'CONFIRMATION_LABELS.json')['candidate_sha256']==sha(U/'CANDIDATE_FROZEN.json'),'reserve_labels_after_fixed_candidate')
for role,lf in [('diagnostic','DIAGNOSTIC_LABELS.json'),('confirmation_reserved','CONFIRMATION_LABELS.json')]:check(read(U/f'{role}_RESULTS_V2.json')['labels_sha256']==sha(U/lf),'eval_label_hash:'+role)
r=read(U/'SHADOW_REPLAY.json');check(len(r['rows'])==122,'replay122');check(sum(x['normal']for x in r['rows'])==64,'normal64');check(all(x['verdicts_unchanged']for x in r['rows']),'coarse_verdicts_same');check(sum(x['new_details']for x in r['rows'])==20,'details20')
d=read(U/'diagnostic_RESULTS_V2.json')['summary'];h=read(U/'HISTORIC_RESULTS_V2.json')['summary'];c=read(U/'confirmation_reserved_RESULTS_V2.json')['summary']
gates=dict(no_new_confident_state=True,diagnostic_errors_reduced=d['quality_only']['wrong']<d['old']['wrong'],reserve_errors_reduced=c['quality_only']['wrong']<c['old']['wrong'],correct_loss_le5pct=all(s['quality_only']['old_correct_lost']/s['old']['correct']<=.05 for s in[d,h,c]),normal_coarse_regression=True,reserve_positive_quality_examples=counts['CONFIRMATION_LABELS.json']['quality_bad']>0)
decision=dict(status='COMPARISON_COMPLETE_NOT_ADMITTED',web_connected=False,passed=all(gates.values()),gates=gates,reason='Diagnostic2errors suppressed, but reserve error unchanged1 and zero positive wrong-chain anchors in reserve. Historic40correct observations additionally withheld. Phase still false change on rotation. No rollout or retuning on15.',selected='quality_only',old_web='G6-finger-20261115')
write(U/'DECISION.json',decision);write(U/'ERROR_REVIEW.json',dict(old_errors=errors,changed_finger_decisions=change_cases,stable_falsely_marked_change=head_errors,byhand=byhand,scope='Source observational labels; no claim of learner sign correctness or expert truth.'))
write(U/'VERIFIED.json',dict(artifact_checks_passed=True,checks=len(checks),checks_detail=checks,protected_files_unchanged=len(read(U/'PROTECTED.json')),counts=counts,release_gate_passed=decision['passed'],independent_expert=False,implementer='Codex',verifier='Same Codex, separate saved artifact pass',user_recordings_used=False))
print(json.dumps(dict(checks=len(checks),counts=counts,gates=gates,old_errors=errors),ensure_ascii=False))
