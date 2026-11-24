from pathlib import Path
import json,hashlib,ast,datetime,shutil
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;P=G/'selective_rtmw_retry_20261117'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks={}
    for rel in ['PUBLIC_VALIDATION.json','GUARD_CHECK.json','web/CAPTURE_CHECK.json']:
        p=D/rel;assert read(p)['passed'];checks[str(p)]=sha(p)
    assert read(D/'VERIFICATION_PLAN.json')['frozen_candidate_sha256']==sha(D/'selective_retry.py')
    v=read(D/'PUBLIC_VALIDATION.json');assert v['recovered_cases']>0
    old=read(P/'PUBLIC_VALIDATION.json');oldmap={(r['id'],r['kind']):r for r in old['controls']}
    pairs=[(oldmap[r['id'],r['kind']],r) for r in v['controls'] if r['kind']in['missing_hand','missing_one_finger']and(r['id'],r['kind'])in oldmap]
    assert len(pairs)==16 and sum(b['unknown_after']>0 for a,b in pairs)<sum(a['unknown_after']>0 for a,b in pairs)
    assert read(D/'DECODE_PROFILE.json')['pixels_equal']
    capture=read(D/'web/CAPTURE_CHECK.json')
    assert any(r['retry'].get('accepted')for r in capture['rows'])
    for r in capture['rows']:
        f=Path(r['directory']);dest=D/'public_capture_artifacts'/f"{r['word']}_{r['kind']}";dest.mkdir(parents=True,exist_ok=True)
        for src in [f/'recorded.mp4',f/'selective_retry.json',f/'rtmw/observations.npz',f/'rtmw/SESSION.json']:
            if src.is_file():shutil.copy2(src,dest/src.name)
    pins=read(P/'web/ADMISSION.json')['pins'];assert all(sha(p)==h for p,h in pins.items());previous=len(pins)
    files=[D/x for x in['selective_retry.py','retry_runtime.py','serve.py','CONTRACT.json','ENGINEERING_NOTE.json']]
    files += [O/'word100_rtmw_stream_reuse_20261108/stream_runtime.py']
    modelroot=O/'word100_rtmw_mediapipe_compare_20261108'
    files += [next((modelroot/'models'/folder).rglob('*.onnx'))for folder in ['rtmw_l_384','person_yolox_m']]
    for p in D.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    for p in files:pins[str(p)]=sha(p)
    q=dict(admitted=True,at=datetime.datetime.now().astimezone().isoformat(),pins=pins,checks=checks,prior_pins_preserved=previous,training_runs=0,private_data=False,scope='Bounded selective re-observation of visible short joint losses at important-action boundaries; known public F controlled tests, not general recovery of hidden fingers or physical webcam acceptance')
    (D/'web/ADMISSION.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ADMITTED',len(pins),'PRIOR',previous)
if __name__=='__main__':main()
