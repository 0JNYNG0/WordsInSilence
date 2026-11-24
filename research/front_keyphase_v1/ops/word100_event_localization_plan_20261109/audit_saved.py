"""Read-only saved-data review for next plan. No training, inference or rescoring."""
from pathlib import Path
import json,hashlib,numpy as np
V=Path(__file__).resolve().parent;O=V.parent;U=O/'word100_attempt_event_supervision_20261109';P=O/'word100_separate_phase_learning_20261109';B=O/'word100_shared_motion_learning_20261109'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(b,s):assert b,s;checks.append(s)
def main():
 rows=read(U/'INPUTS.json')['rows'];normals={r['clip_id']:r for r in rows if r['kind']=='full_normal'};ready={r['clip_id']:r for r in read(B/'READY.json')['rows']};labels=[];controls=[];failures=[]
 for cid,r in normals.items():
  ck(sha(r['path'])==r['sha256'],'unchanged input '+cid);z=np.load(r['path']);orig=np.load(ready[cid]['path'])
  labels.append(dict(clip_id=cid,role=r['role'],person=r['person'],direct_stage_frames=int(((z['sy']>0)&(z['sw']>=.99)).sum()),weak_stage_frames=int(((z['sy']>0)&(z['sw']>0)&(z['sw']<.99)).sum()),phase_weight_values=sorted(set(z['sw'].tolist())),direct_shape_cells=int(orig['direct_ym'].sum()),stage_gold_frames=int((orig['stage_gold']>=0).sum())))
 legacy=read(P/'EVAL_INPUTS.json')['rows'];ln={(r['clip_id'],r['kind']):r for r in legacy}
 for r in legacy:
  if r['kind']!='right_closed_hand_main':continue
  ck(sha(r['path'])==r['sha256'],'legacy input pinned');cid=r['clip_id'];z=np.load(r['path']);n=np.load(ln[(cid,'crop_normal')]['path'])
  a=z['raw_x'][:,:102].reshape(-1,51,2);b=n['raw_x'][:,:102].reshape(-1,51,2)
  body=float(abs(a[:,:9]-b[:,:9]).max());wrists=float(abs(a[:,[9,30]]-b[:,[9,30]]).max());palm=[30,35,39,43,47];palm_change=float(np.linalg.norm(a[:,palm].mean(1)-b[:,palm].mean(1),axis=1).max())
  same=next(q for q in rows if q['clip_id']==cid and q['kind']=='close_R');x=np.load(same['path']);orig=np.load(normals[cid]['path']);an=x['raw'][:,:102].reshape(-1,51,2);bn=orig['raw'][:,:102].reshape(-1,51,2)
  controls.append(dict(clip_id=cid,matched_normal='crop_normal, same image-extraction path',body_max_delta=body,hand_wrist_max_delta=wrists,legacy_palm_centre_max_shift_shoulder_units=palm_change,new_control_palm_centre_max_shift=float(abs(an[:,palm].mean(1)-bn[:,palm].mean(1)).max()),old_raw_shape_delta=float(abs(a[:,30:51]-b[:,30:51]).max())))
  ck(body==0 and wrists<1e-6,'legacy preserves body/wrists')
  d=read(U/'legacy_decisions/final_attempt_supervised_42'/f'{cid}_right_closed_hand_main.json')
  dc=read(U/'legacy_decisions/final_attempt_supervised_42'/f'{cid}_crop_normal.json')
  failures.append(dict(clip_id=cid,matched_normal_cards=[{k:c.get(k) for k in ['id','status','display_status','query_indices','phase_evidence']} for c in dc['cards']],cards=[{k:c.get(k) for k in ['id','required','status','display_status','query_indices','phase_evidence','event_presence_probability','uncertainty_reason']} for c in d['cards']]))
 fit=[r for r in labels if r['role']=='fit'];dev=[r for r in labels if r['role']=='development']
 sums=dict(fit_clips=len(fit),fit_direct_stage_frames=sum(r['direct_stage_frames'] for r in fit),fit_weak_stage_frames=sum(r['weak_stage_frames'] for r in fit),fit_direct_stage_clips=sum(r['direct_stage_frames']>0 for r in fit),fit_direct_shape_cells=sum(r['direct_shape_cells'] for r in fit),historical_dev_clips=len(dev),historical_dev_stage_frames=sum(r['stage_gold_frames'] for r in dev))
 # Saved normal withheld card, no reassessment with a changed threshold.
 normal=[]
 for name in ['final_normal_only_42','final_attempt_supervised_42']:
  d=read(U/'decisions'/name/'NIA_SL_WORD1133_REAL11_F_full_normal.json');c=d['cards'][0]
  normal.append(dict(model=name,span=c['query_indices'],status=c['display_status'],shape_evidence=c['shape_evidence']))
 sources=[P/'lib.py',U/'unit.py',U/'prepare.py',U/'train.py',O/'word100_action_feedback_demo_20261108/evidence.py',P/'followup_contract.py',U/'PROTOCOL_AUDIT.json']
 result=dict(status='REVIEW_COMPLETE_NO_NEW_MODEL',checks=checks,label_support=sums,labels=labels,legacy_shape_controls=controls,failed_stage_assignment=failures,normal_transition=normal,code_findings=[dict(path=str(P/'lib.py'),finding='Phase input contains64 learned observation features+7attributes+66geometry. Geometry includes palm centres. Difference inspection is downstream of selected phase span.'),dict(path=str(U/'unit.py'),finding='Presence has mean/max pooling; can represent absence but has no explicit event order loss. Phase softmax is restricted to expected word stages plus background; no explicit unknown-action class.'),dict(path=str(U/'prepare.py'),finding='Changed shape target cells masked rather than falsely relabelled normal. New close_hand preserves palm anchors; legacy whole-hand template also moves palm-centre geometry. Last-event omission has an abrupt edit seam.'),dict(path=str(P/'followup_contract.py'),finding='Guard checks other threshold-qualified candidate runs; it cannot restore a stage with no qualified runs.'),dict(path=str(U/'PROTOCOL_AUDIT.json'),finding='Inner07-09 already seen by initial models; not suitable as independent model-selection holdout.')],source_pins={str(p):sha(p) for p in sources},claims=['Body/wrist input invariance verified on3 historical handshape controls; this supports testing a wrist/body candidate route, not proof it will improve accuracy.','Labels are sparse direct anchors plus weak transferred intervals, not continuously verified event boundaries.','No new training, neural inference, prediction rewrite, threshold choice or private access.'])
 (V/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],support=sums,controls=controls),ensure_ascii=False))
if __name__=='__main__':main()
