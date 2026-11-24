from pathlib import Path
import json,hashlib,ast,datetime,shutil
D=Path(__file__).resolve().parent;R=D.parent/'selective_rtmw_anchor_repair_20261117'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 checks={}
 for rel in ['checks/SHORT_CHECK.json','checks/STOP_CHECK.json','checks/DEMO_CHECK.json','web/CONSUMER_CHECK.json','web/CARD_CHECK.json','web/CAPTURE_CHECK.json']:
  p=D/rel;assert read(p)['passed'],rel;checks[str(p)]=sha(p)
 stop=read(D/'checks/STOP_CHECK.json');assert any(r['before']['first_stop']is None and r['after']['first_stop']is not None for r in stop['rows'])
 assert read(D/'checks/SHORT_CHECK.json')['improved']>0
 pins=read(R/'web/ADMISSION.json')['pins'];assert all(sha(p)==h for p,h in pins.items());old=len(pins)
 for p in list(D.glob('*.py'))+list((D/'thanks_runtime').glob('*.py')):ast.parse(p.read_text('utf-8'));pins[str(p)]=sha(p)
 for p in [D/'CONTRACT.json',D/'ENGINEERING_NOTE.json']:pins[str(p)]=sha(p)
 for row in read(D/'thanks_runtime/LOADED_WEIGHTS.json'):
  assert sha(row['path'])==row['sha256'];pins[row['path']]=row['sha256']
 for r in read(D/'web/CAPTURE_CHECK.json')['rows']:
  folder=Path(r['directory']);out=D/'public_capture_artifacts'/f"{r['word']}_{r['kind']}";out.mkdir(parents=True,exist_ok=True)
  for src in [folder/'recorded.mp4',folder/'rtmw/observations.npz',folder/'auto_end_diagnostic.json']:
   if src.is_file():shutil.copy2(src,out/src.name)
 q=dict(admitted=True,at=datetime.datetime.now().astimezone().isoformat(),pins=pins,checks=checks,prior_pins_preserved=old,training_runs=0,private_data=False,physical_camera_verified=False,scope='Public frontal seated stop + learned multi-window movement evidence. No new weights or user-specific tuning.')
 (D/'web/ADMISSION.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');print('ADMITTED',len(pins),'prior',old)
if __name__=='__main__':main()
