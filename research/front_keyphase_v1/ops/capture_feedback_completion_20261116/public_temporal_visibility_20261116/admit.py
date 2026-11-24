from pathlib import Path
import json,hashlib,ast
L=Path(__file__).resolve().parent;K=L.parent/'public_view_arrival_support_20261116';R=L.parents[4]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 assert read(L/'qualified_checks/RESULT.json')['regressions']==0
 q=read(L/'qualified_regression/RESULT.json');assert q['shape_new_wrong']==q['shape_lost_correct']==q['distinction_failed']==q['temporal_failed']==0 and not q['known_error_loss']
 assert read(L/'qualified_regression/confirmation/PATTERN_CHECK.json')['passed']and read(L/'verified_extra/RESULT.json')['passed']
 pins=dict(read(K/'web/ADMISSION.json')['pins']);assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
 paths=[K/'web/ADMISSION.json']+[L/n for n in ['diagnose.py','train_visible.py','candidate.py','endpoint_v2.py','final_temporal.py','temporal_presentation.py','temporal_web_adapter.py','runtime/worker.py','serve.py']]+list((L/'palm').glob('*.pt'))+[R/'temporal_demo_launcher.py']
 for p in paths:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'))
  pins[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
 (L/'web').mkdir(exist_ok=True);(L/'web/ADMISSION.json').write_text(json.dumps(dict(admitted=True,pins=pins,status='Public gates passed; verify runtime before activation.',private_data=False),indent=2),encoding='utf8');print('RUNTIME_ADMISSION',len(pins),flush=True)
if __name__=='__main__':main()
