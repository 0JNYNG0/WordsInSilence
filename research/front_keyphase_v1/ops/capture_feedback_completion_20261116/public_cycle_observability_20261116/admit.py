from pathlib import Path
import json,hashlib,ast
M=Path(__file__).resolve().parent;L=M.parent/'public_temporal_visibility_20261116';R=M.parents[4]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 assert read(M/'FASTPATH_CHECK.json')['passed']
 assert read(M/'learned_checks/RESULT.json')['failed']==0
 q=read(M/'final_regression/RESULT.json');assert q['shape_new_wrong']==q['shape_lost_correct']==q['distinction_failed']==q['temporal_failed']==0 and not q['known_error_loss']
 assert read(M/'final_regression/confirmation/PATTERN_CHECK.json')['passed']and read(M/'final_extra/RESULT.json')['passed']and read(M/'FINAL_EPISODE_CONTROLS.json')['passed']
 pins=dict(read(L/'web/ADMISSION.json')['pins']);assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
 paths=[L/'web/ADMISSION.json']+[M/n for n in ['cycle_common.py','cycle_candidate.py','final_cycle.py','train_motion.py','CALIBRATION.json','cycle_web_adapter.py','runtime/worker.py','serve.py']]+list((M/'motion').glob('*.pt'))+[R/'cycle_demo_launcher.py']
 for p in paths:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'))
  pins[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
 (M/'web/ADMISSION.json').write_text(json.dumps(dict(admitted=True,pins=pins,status='Public gates passed; runtime checks required before activation.',private_data=False),indent=2),encoding='utf8');print('ADMISSION',len(pins),flush=True)
if __name__=='__main__':main()
