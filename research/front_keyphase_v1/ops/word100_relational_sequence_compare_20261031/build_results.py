from pathlib import Path
from collections import Counter
import json,hashlib,html,numpy as np
from relational import features,GROUPS,NAMES
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_core_stage_evidence_20261031';EP=U.parent/'word100_semantic_episode_pilot_20261031'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']};phase=read(OLD/'PHASE_DATA.json');clips={c['alias']:c for c in phase['clips']};dev=read(U/'DEVELOPMENT.json');guide=read(U/'SOURCE_FEATURE_CONTRACT.json');by_word={e['word_id']:e for e in guide['expressions']};emp=read(EP/'EMPIRICAL_RANGE.json');oldpairs={(d['alias'],p['reference']):p for d in emp['development'] for p in d['pairs']}
 featuredata={}
 for a,r in rows.items():
  f=features(dict(np.load(r['input'],allow_pickle=False)));featuredata[a]={g:dict(x=x.tolist(),mask=m.tolist()) for g,(x,m) in f.items()}
 pairs=[]
 for d in dev['results']:
  raw=read(U/'pairs'/f"{d['condition']}_{d['query']}_{d['reference']}.json");raw['decisions']=d['decisions'];raw['counts']=d['counts'];raw['old_saved']=oldpairs[(d['query'],d['reference'])];raw.pop('changes',None);pairs.append(raw)
 controls=[]
 for p in sorted((U/'controls').glob('*.json')):
  d=read(p);controls.append(d)
 summary=[]
 for kind in sorted(set(d['control'] for d in controls)):
  ds=[d for d in controls if d['control']==kind];target='L_shape' if kind=='left_short_shape_change_5f' else 'R_shape';eligible=[d for d in ds if d['metadata'].get('actual_observed_changed_cells',1)>0]
  summary.append(dict(condition=kind,cases=len(ds),observed_edit_cases=len(eligible),target_group=target,nonzero_target_peak=sum((d['groups'][target]['peak_difference'] or 0)>1e-6 for d in eligible),insufficient=sum(not d['groups'][target]['sufficient'] for d in ds),order_reversal_candidates=sum(any(r['state']=='reverse' for r in d['free_order']['relations']) for d in ds),order_ambiguous=sum(any(r['state'] in ['ambiguous','insufficient'] for r in d['free_order']['relations']) for d in ds)))
 transition=dict(frame=89,seconds=2.966667,uncertain_context_frames=[85,96],uncertain_context_seconds=[2.833333,3.2],not_exact_one_frame_transition=True,not_detection_missing=True,not_entire_source_uncovered_action=True)
 ranges=read(U/'REFERENCE_RANGES.json');counts={k:dict(Counter(s['state'] for d in pairs if d['condition']==k for s in d['decisions'])) for k in ['B_same_path','C_relational_path']}
 times={a:sum(p['seconds'] for p in pairs if p['query']==a and p['condition']=='C_relational_path') for a in ['P036','P048','P066','P072']}
 data=dict(status='COMPLETE_DIAGNOSTIC_NOT_PRODUCTION_ADOPTED',groups=NAMES,clips=list(clips.values()),feature_data=featuredata,pairs=pairs,source_contract=guide,control_summary=summary,decision_counts=counts,incremental_cached_comparison_seconds=times,purple_explanation=transition,previous_url='http://127.0.0.1:8791/',new_neural_training=0,word_accuracy_measured=False)
 write(U/'REPORT_DATA.json',data)
 write(U/'RESULT_SUMMARY.json',dict(status=data['status'],decision_counts=counts,controls=summary,comparison_seconds=times,purple=transition,paired_comparisons=72,control_cases=len(controls),new_neural_training=0,adopted_alignment=False,main_result='Added separate left/right shape, arm/body relationship and observability evidence. Relational alignment not shown better; repeated-stage order ambiguous.'))
 print(json.dumps(dict(counts=counts,controls=summary,time=times),ensure_ascii=False))
if __name__=='__main__':main()
