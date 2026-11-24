from pathlib import Path
import hashlib,json,datetime
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins=read(O/'word100_local_event_demo_20261103/EXECUTION_FROZEN.json')['pins']
for p,h in pins.items():assert sha(p)==h,'Frozen prior resource drift '+p
pins.update(read(U/'DISPLAY_SOURCE_PINS.json'))
paths=list(U.glob('*.py'))+list(U.glob('*.js'))+list(U.glob('*.css'))+list(U.glob('*.html'))+[U/'reference_data.json',U/'SCOPE.json']
if (U/'public_tcn.json').exists():paths.append(U/'public_tcn.json')
for name in ['word100_candidate.py','word100_t3f.py','word100.py']:paths.append(O.parent/'src/wis_lab/service'/name)
paths.extend([O/'word100_body_boundary_20261017/training_C_20261013_best.pt',O/'word100_frozen_c_temporal_20261019/T3F_20261013_best.pt'])
for p in paths:pins[str(p.resolve())]=sha(p)
dest=U/'EXECUTION_FROZEN.json'
if dest.exists():
 old=read(dest)
 if old['pins']!=pins:
  archive=U/'freeze_history';archive.mkdir(exist_ok=True)
  dest.rename(archive/('freeze_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'))
 else:print('Unchanged freeze');raise SystemExit
dest.write_text(json.dumps(dict(status='ISOLATED_WEB_INTEGRATION_NO_TRAINING',pins=pins),indent=2,ensure_ascii=False),encoding='utf8')
print('Pinned',len(pins))
