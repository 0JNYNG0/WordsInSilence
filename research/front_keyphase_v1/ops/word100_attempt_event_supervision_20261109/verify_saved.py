"""Independent saved-array / arithmetic audit. No training or model inference."""
from pathlib import Path
import json,hashlib,re,traceback,time
import numpy as np
import torch
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(b,label):
 if not bool(b):raise AssertionError(label)
 checks.append(label)
def main():
 f=read(U/'FROZEN.json');inputs=read(U/'INPUTS.json');rows=inputs['rows'];normal={r['clip_id']:r for r in rows if r['kind']=='full_normal'}
 ck(len(rows)==552 and len(normal)==48,'552 conditions from48 originals')
 for p,h in {**f['protected'],**f['code_pins']}.items():
  if isinstance(h,str) and len(h)==64:ck(sha(p)==h,'pinned unchanged '+str(p))
 ck(sha(U/'INPUTS.json')==f['input_sha256'],'input manifest frozen')
 for filename,code in [('TRAIN_FROZEN.json','train.py'),('EVAL_FROZEN.json','evaluate.py'),('LEGACY_FROZEN.json','legacy_compare.py')]:ck(read(U/filename)['code_sha256']==sha(U/code),'code frozen '+code)
 fit={r['clip_id'] for r in rows if r['role']=='fit'};dev={r['clip_id'] for r in rows if r['role']=='development'}
 ck(len(fit)==36 and len(dev)==12 and not fit&dev,'original FIT36 DEV12 disjoint')
 ck(not {r['source_sha256'] for r in rows if r['role']=='fit'}&{r['source_sha256'] for r in rows if r['role']=='development'},'source digest split')
 ck({r['person'] for r in rows if r['role']=='fit'}==set(range(1,10)) and {r['person'] for r in rows if r['role']=='development'}=={11,12,13},'record ID split')
 norm=read(U/'NORMALIZATION.json');ck(norm['fit_record_ids']==list(range(1,7)),'normalization FIT01-06')
 by={}
 for r in rows:
  ck(sha(r['path'])==r['sha256'],'input digest '+r['clip_id']+' '+r['kind']);z=np.load(r['path']);by[(r['clip_id'],r['kind'])]=z
  ck(all(np.isfinite(z[k]).all() for k in z.files),'finite arrays')
  ck(np.all(np.diff(z['timestamp_us'])>0),'monotonic time')
  if r['kind'] in ['mask_final_event','all_hand_missing']:ck(not z['sw'].any() and not z['pmask'].any(),'missing sensor not taught absent')
 for r in rows:
  n=by[(r['clip_id'],'full_normal')];z=by[(r['clip_id'],r['kind'])];idx=z['source_indices']
  ck(np.array_equal(z['frame_id'],n['frame_id'][idx]),'source identity exact')
  if r['kind']=='omit_final_event':
   ck(np.all(z['present'][r['removed_groups']]==0),'absent target set')
   ck(np.array_equal(z['raw'][:,:-1],n['raw'][idx,:-1]),'omission no synthesized joints')
  if r['kind']=='mask_final_event':ck(np.array_equal(idx,np.arange(len(n['raw']))),'masked event preserves full timing')
  if r['kind'].startswith('close'):
   anchors=[1,2,3,4,5,6,9,14,18,22,26,30,35,39,43,47]
   ck(np.array_equal(z['raw'][:,:102].reshape(-1,51,2)[:,anchors],n['raw'][idx,:102].reshape(-1,51,2)[:,anchors]),'wrist palm arm preserved')
 names=read(U/'EVAL_FROZEN.json')['models'];ck(len(names)==8,'eight fixed runs')
 sums={}
 for name in names:
  c=read(U/'runs'/name/'COMPLETE.json');ck(c['epochs']==100 and len(c['history'])==100,'final100 '+name)
  ck(sha(U/'runs'/name/'model.pt')==c['sha256'],'checkpoint hash '+name)
  allowed={cid for cid in fit if int(re.search(r'REAL(\d+)',cid)[1])<=6} if name.startswith('inner') else fit
  ck(set(c['fit_clips'])==allowed,'no DEV training '+name)
  weights=torch.load(U/'runs'/name/'model.pt',weights_only=True,map_location='cpu');ck(all(torch.isfinite(v).all() for v in weights.values()),'finite checkpoint '+name)
  ck('presence.0.weight' in weights and not any(k.startswith('gru') for k in weights),'separate heads no backbone write')
  if 'attempt_supervised' in name:ck(all(c['seen'].get(k,0)>0 for k in ['close_R','close_L','close_both','reverse_main','close_reverse','omit_final_event']),'all counterfactual families trained')
  else:ck(not any(k in c['seen'] for k in ['close_R','reverse_main','omit_final_event']),'matched normal-only condition')
  cases=read(U/'scores'/f'{name}_cases.json');records=read(U/'scores'/f'{name}_records.json');expected=read(U/'scores'/f'{name}.json')['by_kind'];sums[name]={}
  selected=[r for r in rows if (7<=r['person']<=9 if name.startswith('inner') else r['role']=='development')]
  ck({(r['clip_id'],r['kind']) for r in selected}=={(c['clip_id'],c['kind']) for c in cases},'all cases retained '+name)
  ck(len(records)==len({(r['clip_id'],r['kind'],r['frame']) for r in records}),'original reviewed anchors not duplicated '+name)
  for c in cases:
   ck(sha(c['decision'])==c['sha256'],'decision hash');d=read(c['decision']);cards=d['cards']
   ck(c['display']==[q['display_status'] for q in cards] and c['status']==[q['status'] for q in cards],'stored labels match decision')
   for card in cards:
    ck(card['presentation']['segments']==[s for e in card['phase_evidence'] for s in e['available_runs']],'all candidate evidence exposed')
   if c['kind']=='all_hand_missing':ck('similar' not in c['display'] and 'missing_candidate' not in c['status'],'no all-hidden false green or absence')
  for kind,e in expected.items():
   cs=[c for c in cases if c['kind']==kind];rr=[r for r in records if r['kind']==kind]
   actual=dict(clips=len(cs),stage_correct=sum(r['pred']==r['target'] for r in rr),stage_total=len(rr),stage_confident_correct=sum(r['confident'] and r['pred']==r['target'] for r in rr),required_present_clips=sum(all(s for s,req in zip(c['phase_present'],c['required']) if req) for c in cs),difference_clips=sum('difference' in c['display'] for c in cs),missing_candidate_clips=sum('missing_candidate' in c['status'] for c in cs),all_required_similar=sum(all(s=='similar' for s,req in zip(c['display'],c['required']) if req) for c in cs),uncertain_clips=sum('uncertain' in c['display'] for c in cs),presence_absence_below_threshold=sum(all(c['presence'][g]<=.35 for g in c['omitted_groups']) for c in cs if c['omitted_groups']))
   ck(actual==e,'independent arithmetic '+name+' '+kind);sums[name][kind]=actual
  if name.endswith('42'):
   groups=read(O/'word100_gru_semantic_finetune_20261109/FROZEN.json')['semantic_stage_groups'];mapping={g:i for i,gs in enumerate(groups) for g in gs}
   for c in cases:
    pred=np.load(U/'predictions'/name/f"{c['clip_id']}_{c['kind']}.npz")['phase'];z=by[(c['clip_id'],c['kind'])];rr=[r for r in records if r['clip_id']==c['clip_id'] and r['kind']==c['kind']]
    for r in rr:
     ix=r['index'];p=int(pred[ix].argmax());ck(r['target']==mapping[int(z['stage_gold'][ix])] and r['pred']==mapping.get(p-1,-1),'prediction and original stage recompute')
 legacy=read(U/'LEGACY_COMPARE.json')['models']
 for name,groups in legacy.items():
  cs=read(U/'scores'/f'{name}_legacy_cases.json');ck(len(cs)==93,'all93 legacy '+name)
  for c in cs:ck(sha(c['decision'])==c['sha256'],'legacy hash')
  for kind,e in groups.items():
   cc=[c for c in cs if c['kind']==kind]
   a=dict(clips=len(cc),old_difference=sum('difference' in c['before'] for c in cc),new_difference=sum('difference' in c['after'] for c in cc),uncertain=sum('uncertain' in c['after'] for c in cc),all_required_similar=sum(all(s=='similar' for s,r in zip(c['after'],c['required']) if r) for c in cc),all_required_phase_present=sum(all(s for s,r in zip(c['phase_presence'],c['required']) if r) for c in cc))
   ck(a==e,'legacy arithmetic '+name+' '+kind)
 a=sums['final_attempt_supervised_42'];b=sums['final_normal_only_42'];l=legacy['final_attempt_supervised_42']
 gates=dict(normal_no_new_difference=a['full_normal']['difference_clips']==b['full_normal']['difference_clips']==0,normal_confirmation_preserved=a['full_normal']['all_required_similar']>=b['full_normal']['all_required_similar'],all_missing_kept_uncertain=a['all_hand_missing']['all_required_similar']==0,omitted_events_not_success=a['omit_final_event']['missing_candidate_clips']==6 and a['omit_final_event']['all_required_similar']==0,sensor_missing_not_omission=a['mask_final_event']['missing_candidate_clips']==0 and a['mask_final_event']['all_required_similar']==0,all_wrong_shape_attempts_retained=l['right_closed_hand_main']['all_required_phase_present']==l['right_closed_hand_main']['clips'])
 ck(not all(gates.values()),'failed full deployment gate honestly retained')
 result=dict(status='PASS_SAVED_ARTIFACT_AUDIT',checks=len(checks),items=checks,deployment_gates=gates,new_model_deployable=False,verification='Separate saved arithmetic by same Codex; not an independent expert or fresh evaluation',training_or_inference=0,primary=dict(normal=a['full_normal'],omission=a['omit_final_event'],mask=a['mask_final_event']),legacy_shape=l['right_closed_hand_main'])
 (U/'FINAL_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],checks=len(checks),gates=gates)))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'VERIFY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
