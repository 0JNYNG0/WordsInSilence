"""Check saved followup, with an independent calculation for omitted-run evidence."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;O=P.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(ok,label):
 checks.append(label)
 assert ok,label
def main():
 frozen=read(P/'FOLLOWUP_FROZEN.json');res=read(P/'FOLLOWUP_RESULT.json')
 ck(sha(P/'followup_contract.py')==frozen['implementation_sha256'],'followup code pinned')
 ck(sha(P/'EVAL_INPUTS.json')==frozen['inputs_sha256'],'followup inputs pinned')
 ck(sha(P/'EVALUATION_COMPLETE.json')==frozen['baseline_scores_sha256'],'original comparison unchanged')
 for name,meta in res['models'].items():
  ck(sha(meta['rows'])==meta['sha256'],'result hash '+name);rows=read(meta['rows']);ck(len(rows)==129,'129 controls '+name)
  ck(sum(bool(r['changes']) for r in rows)==meta['changed_clips'],'changes count '+name)
  for r in rows:
   for old,new in zip(r['before'],r['after']):ck(old==new or (old=='similar' and new=='uncertain'),'only withhold, never invent correctness '+name)
   if r['kind'] in ['full_normal','crop_normal','speed0.75','speed1.25','translation_scale','gap2','gap12','all_hand_missing']:
    ck(r['before']==r['after'],'normal/missing outputs preserved '+name+r['kind'])
  changed=[r for r in rows if r['changes']]
  ck(len(changed)==(1 if name.startswith('matched_relations') else 0),'expected scope '+name)
 # Directly verify that the retained, discarded interval has opposite shape,
 # visible finger chains and qualifying duration. No call to inference/refine.
 z=np.load(P/'controls/NIA_SL_WORD1133_REAL12_F_right_closed_hand_main.npz');cfg=read(O/'word100_positive_direction_20261109/FROZEN.json')['positive']
 interval=np.arange(75,83);raw=z['raw_x'];p=z['p'];points=raw[:,:102].reshape(-1,51,2);rat=[]
 for k in [5,9,13,17]:
  q=points[:,30+k:30+k+4];rat.append(np.linalg.norm(q[:,-1]-q[:,0],axis=1)/np.maximum(np.linalg.norm(np.diff(q,axis=1),axis=2).sum(1),1e-6))
 opposite=(p[:,1]<=1-cfg['threshold'])&(p[:,3]>=cfg['threshold'])
 observed=(raw[:,186:237][:,[30]+list(range(35,51))]>.5).all(1)
 corroborated=(np.stack(rat,1)<.85).sum(1)>=3
 good=opposite&observed&corroborated;idx=interval[good[interval]]
 ck(len(idx)>=2,'discarded run has opposite visible evidence')
 dt=float(np.median(np.diff(z['timestamp_us'])));duration=(z['timestamp_us'][idx[-1]]-z['timestamp_us'][idx[0]]+dt)/1000
 ck(np.all(np.diff(idx)==1) and duration>=cfg['min_evidence_ms'],'opposite evidence sustained with original threshold')
 result=dict(status='PASS',checks=len(checks),items=checks,opposite_interval=idx.tolist(),duration_ms=duration,
   scope='Same Codex independent arithmetic, not independent human ground truth; post-hoc regression only',no_model_forward=True)
 (P/'FOLLOWUP_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False))
if __name__=='__main__':main()
