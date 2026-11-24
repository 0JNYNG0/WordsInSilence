"""Interpret stored fit controls under already frozen C ranges; no tuning/replay."""
from pathlib import Path
from collections import Counter
import json
from run_compare import decisions
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']};ranges=read(U/'REFERENCE_RANGES.json')['ranges']['C_relational_path'];results=[]
 for p in sorted((U/'controls').glob('*.json')):
  d=read(p);a=d['alias']
  if rows[a]['role']!='fit':continue
  d['reference']=a;dec=decisions(d,ranges);target='L_shape' if d['control']=='left_short_shape_change_5f' else 'R_shape';relevant=[s for s in dec if s['group']==target]
  results.append(dict(alias=a,control=d['control'],target_group=target,original_control=p.name,observed_edit_cells=d['metadata'].get('actual_observed_changed_cells'),counts=dict(Counter(s['state'] for s in relevant)),larger_any=any(s['state']=='larger_than_fit_reference_range' for s in relevant),all_target_unavailable=all(s['state'] in ['reference_range_unavailable','observation_insufficient'] for s in relevant),decisions=relevant))
 summary=[]
 for name in sorted({x['control'] for x in results}):
  ds=[x for x in results if x['control']==name];summary.append(dict(control=name,fit_cases=len(ds),larger_than_range_cases=sum(x['larger_any'] for x in ds),all_target_unavailable=sum(x['all_target_unavailable'] for x in ds),observed_edit_cases=sum(x['observed_edit_cells'] is not None and x['observed_edit_cells']>0 for x in ds)))
 with (U/'CONTROL_RANGE_DIAGNOSTIC.json').open('x',encoding='utf8') as f:json.dump(dict(status='FIXED_RANGE_DIAGNOSTIC_NOT_KSL_ACCURACY',fit_only=True,development_excluded=4,new_inference=False,rows=results,summary=summary),f,ensure_ascii=False,indent=2)
 print(json.dumps(summary))
if __name__=='__main__':main()
