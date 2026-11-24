"""Saved-output audit separate from training/evaluation, no model inference."""
from pathlib import Path
import json,hashlib,collections,statistics,math
U=Path(__file__).resolve().parent
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
checks=[]
def ck(name,ok):
 checks.append(dict(name=name,passed=bool(ok)))
 if not ok:raise AssertionError(name)
def compact(cards):
 displays=[r['display_status'] for r in cards]
 return dict(display=displays,warning='difference' in displays,similar_no_warning=all(r['display_status']=='similar' for r in cards if r.get('required',True)) and 'difference' not in displays)
for p,h in {**rd(U/'PROTECTIONS.json'),**rd(U/'FROZEN.json')['pins'],**rd(U/'EVAL_PINS.json')['pins']}.items():ck('frozen_file_unchanged',sha(p)==h)
tr=rd(U/'TRAIN_INPUTS_COMPLETE.json');ev=rd(U/'EVALUATION_INPUTS.json');ck('fit_evaluation_sha_disjoint',not {r['source_sha256'] for r in tr['rows']}&{r['source_sha256'] for r in ev['rows']})
ck('source_records_only_fit',all(1<=r['person']<=9 for r in tr['rows']))
rows=rd(U/'FULLPATH_COMPLETE.json')['rows'];ck('all_conditions_all_seeds_all_modes',len(rows)==78*3*3 and len({(r['seed'],r['clip_id'],r['kind'],r['mode']) for r in rows})==len(rows))
changed=[]
for r in rows:
 d=rd(r['result']);ck('output_sha',sha(r['result'])==r['sha256']);ck('baseline_sha',sha(d['baseline'])==d['baseline_sha256']);before=rd(d['baseline']);ck('before_arithmetic',compact(before['cards'])==r['before']);ck('after_arithmetic',compact(d['cards'])==r['after'])
 for a,b in zip(d['cards'],before['cards']):
  ck('shape_preserved',a.get('shape_evidence')==b.get('shape_evidence'));ck('direction_evidence_preserved',a.get('motion_evidence')==b.get('motion_evidence'));ck('old_warning_retained',b['display_status']!='difference' or a['display_status']=='difference')
 for e in d['ledger']:ck('no_desired_direction_filter',not e['expected_direction_used']);ck('no_forced_semantic_completion',not e['semantic_complete'])
 selected=d['selected'];ck('chronological_selection',selected==sorted(selected,key=lambda e:e['a']));ck('one_motion_not_reused',not any(a['axis']==b['axis'] and a['moving']==b['moving'] and max(a['a'],b['a'])<min(a['b'],b['b']) for i,a in enumerate(selected) for b in selected[i+1:]))
 if r['before']!=r['after']:changed.append({k:r[k] for k in ['seed','clip_id','kind','mode','before','after','result']})
summary={};scientific={}
for seed in [42,43,44]:
 for mode in ['role_only','quality_only','combined']:
  ss=[r for r in rows if r['seed']==seed and r['mode']==mode];by={}
  for kind in sorted({r['kind'] for r in ss}):
   rr=[r for r in ss if r['kind']==kind];by[kind]=dict(total=len(rr),before_warnings=sum(r['before']['warning'] for r in rr),after_warnings=sum(r['after']['warning'] for r in rr),before_false_similar=sum(r['before']['similar_no_warning'] for r in rr),after_similar=sum(r['after']['similar_no_warning'] for r in rr),omission_candidates=sum(r['omission_candidates'] for r in rr),changed=sum(r['before']!=r['after'] for r in rr))
  summary[f'{seed}/{mode}']=by
  normal=[r for r in ss if r['kind'] in ['full_normal','speed075','speed125']];unknown=[r for r in ss if r['kind'] in ['all_hand_missing','mask_final_event']];rev=by['reversed_core_auto_order'];omit=by['omit_final_event'];scientific[f'{seed}/{mode}']=dict(normal_no_extra_warning=all(not r['after']['warning'] for r in normal),unobserved_not_promoted_normal=all(not r['after']['similar_no_warning'] for r in unknown),unobserved_no_omission_claim=all(r['omission_candidates']==0 for r in unknown),reverse_warning_gain=rev['after_warnings']>rev['before_warnings'],reverse_false_similar_reduced=rev['after_similar']<rev['before_false_similar'],omission_improved=omit['omission_candidates']>0)
frame={};f=rd(U/'FRAME_ROLE_SCORES.json');cand=rd(U/'CANDIDATE_ROLE_SCORES.json')
for seed in [42,43,44]:
 rr=[r for r in f if r['seed']==seed];cc=[r for r in cand if r['seed']==seed];frame[str(seed)]=dict(total=len(rr),before_correct=sum(r['before']==r['target'] for r in rr),after_correct=sum(r['after']==r['target'] for r in rr),known_candidates=len(cc),before_candidate_correct=sum(r['before']==r['target'] for r in cc),after_candidate_correct=sum(r['after']==r['target'] for r in cc),per_class={str(k):dict(total=sum(r['target']==k for r in rr),before=sum(r['target']==k and r['before']==k for r in rr),after=sum(r['target']==k and r['after']==k for r in rr)) for k in range(8)})
cross={}
if (U/'CROSS_EXPRESSION_SCORES.json').exists():
 cr=rd(U/'CROSS_EXPRESSION_SCORES.json')
 for s in [42,43,44]:
  rr=[r for r in cr if r['seed']==s and r['known']];cross[str(s)]=dict(known=len(rr),unknown=sum(r['seed']==s and not r['known'] for r in cr),before_correct=sum(r['before']==r['target'] for r in rr),after_correct=sum(r['after']==r['target'] for r in rr),per_class={str(k):dict(n=sum(r['target']==k for r in rr),before=sum(r['target']==k and r['before']==k for r in rr),after=sum(r['target']==k and r['after']==k for r in rr)) for k in range(4)})
times=[r['role_seconds']+r['binding_seconds'] for r in rows if r['seed']==42 and r['mode']=='combined'];adopt=all(all(scientific[f'{s}/combined'].values()) for s in [42,43,44])
wr(U/'RESULT.json',dict(final_feedback=summary,scientific_gates=scientific,frame_role=frame,cross_expression_quality=cross,latency=dict(median_seconds=statistics.median(times),p95_seconds=sorted(times)[math.ceil(.95*len(times))-1],scope='Adapter inference and candidate binding only; cached frozen encoder, RTMW, TCN, DTW and UI excluded'),meets_predeclared_all_gates=adopt,web_adopted=False))
wr(U/'CHANGED_FEEDBACK.json',dict(rows=changed));wr(U/'VERIFICATION.json',dict(integrity_passed=True,checks=len(checks),scientific_gates_separate=scientific,independent_human_review=False,reviewer='Same Codex separate saved arithmetic audit; no training/inference repeated'));print(json.dumps(dict(checks=len(checks),frame=frame,cross=cross,gates=scientific,changed=len(changed)),ensure_ascii=False))
