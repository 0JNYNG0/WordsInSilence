"""Apply preregistered gates to verified saved results only; never rerun models."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
U=Path(__file__).resolve().parent
read=lambda n:json.loads((U/n).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 dest=U/'PROCESS_DECISION.json';assert not dest.exists()
 c=read('PROCESS_COMPARISON_FROZEN.json');r=read('PROCESS_EVALUATION.json');a=read('PROCESS_AUTO_EVALUATION.json')
 rv=read('PROCESS_EVALUATION_SAVED_VERIFIED.json');av=read('PROCESS_AUTO_EVALUATION_VERIFIED.json')
 assert rv['status'].startswith('PASS') and av['status'].startswith('PASS')
 for p,h in c['pins'].items():assert sha(p)==h,p
 g=c['fixed_gates'];ds=r['summaries']['development'];classes=ds['overall']['class_counts']
 qs=a['summaries']['development']['real_runtime_qualification'];auto=a['summaries']['development']['automatic']
 known0=[x for x in r['rows'] if x['role']=='development' and x['truth']==0]
 native=[x for x in r['rows'] if x['role']=='development' and x['kind']=='native' and x['truth']==1]
 tests=[]
 def minimum(name,actual,required):tests.append(dict(name=name,actual=actual,required_min=required,passed=actual>=required))
 def maximum(name,actual,required):tests.append(dict(name=name,actual=actual,required_max=required,passed=actual<=required))
 minimum('raw_macro_query_balanced_accuracy',ds['macro_query_balanced_accuracy'],g['raw_macro_query_balanced_accuracy_min'])
 minimum('known_positive_raw_confident_recovered',classes['1']['confident_correct'],g['known_positive_confident_required'])
 minimum('known_zero_raw_confident_recovered',classes['0']['confident_correct'],g['known_negative_confident_required'])
 maximum('known0_raw_confident_false_positive',sum(x['raw_display']==1 for x in known0),g['known0_confident_false_positive_max'])
 maximum('native1_raw_confident_false_negative',sum(x['raw_display']==0 for x in native),g['native1_confident_false_negative_max'])
 minimum('qualified_positive_recovered',qs['confirmed_positive_correct'],g['qualified_positive_required'])
 maximum('qualified_known0_false_confirmation',qs['confirmed_positive_wrong'],g['qualified_known0_false_confirmation_max'])
 maximum('qualified_unknown_false_confirmation',qs['unknown_confirmed_positive'],g['qualified_unknown_false_confirmation_max'])
 minimum('auto_native_recovered',auto['recovered'],g['automatic_native_positive_required'])
 for q,x in auto['per_query'].items():minimum('auto_per_query:'+q,x['recovered'],g['automatic_per_query_required'])
 for q in ds['per_query']:
  rows=[x for x in a['view_rows'] if x['role']=='development' and x['kind']=='native' and x['physical_query']==q and x['truth']==1]
  assert len(rows)==3
  minimum('qualified_native_per_query:'+q,sum(x['qualification']['qualified_display']==1 for x in rows),g['native_positive_per_query_required'])
 controls=[x for x in r['rows'] if x['role']=='fit' and x['truth']==1 and x['kind'] in ['prefix','suffix','middle_splice','speed_0_8','speed_1_25','intact_context_trim','missing_rgb']]
 maximum('FIT_positive_control_confident_reversal',sum(x['raw_display']==0 for x in controls),0)
 maximum('old_frozen_files_changed',0,0)
 passed=all(x['passed'] for x in tests)
 out=dict(at=datetime.now(timezone.utc).isoformat(),status='ELIGIBLE_FOR_SEPARATE_RUNTIME_VALIDATION' if passed else 'NOT_ADOPTED_FIXED_GATES_FAILED',
  gates=tests,passed=sum(t['passed'] for t in tests),failed=sum(not t['passed'] for t in tests),
  exact_existing_model_comparison='Old models have no full-process score; no fabricated previous completion accuracy. Their files and saved outputs stay unchanged.',
  new_training_runs=1,epochs=80,raw_query_views=297,automatic_queries=60,new_DTW=0,
  integration_done=False,whole_goal_complete=False,independent_human_validation=False,
  result_pins={str(U/n):sha(U/n) for n in ['PROCESS_COMPARISON_FROZEN.json','PROCESS_TRAIN_RESULT.json','PROCESS_EVALUATION.json',
    'PROCESS_AUTO_EVALUATION.json','PROCESS_EVALUATION_SAVED_VERIFIED.json','PROCESS_AUTO_EVALUATION_VERIFIED.json']},
  next='Diagnose saved process/point/interval outputs and manual-vs-automatic context mismatch before proposing one substantive next change. No threshold/epoch search.',
  current_models='C/T3F and current public feedback demos preserved',private_data_used=False)
 dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(out['status'],'passed',out['passed'],'failed',out['failed'])
if __name__=='__main__':main()
