"""Admit public-validated role separation and the explicitly selected smooth UI."""
from pathlib import Path
import ast, datetime, hashlib, json
D=Path(__file__).resolve().parent
R=D.parent/'rest_stability_endpoint_20261117'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    checks={}
    for rel in ['checks/ROLE_CHECK.json','checks/DIRECT_TRUTH_CHECK.json','checks/REVIEWED_PARTIAL_CHECK.json',
                'web/CONSUMER_CHECK.json','web/CARD_CHECK.json','web/CAPTURE_CHECK.json',
                'qa/BROWSER_CHECK.json','web/HTTP_RECORDING_CHECK.json','web/HTTP_INTEGRATION_CHECK.json']:
        p=D/rel;assert read(p)['passed'],rel;checks[str(p)]=sha(p)
    assert read(D/'checks/ROLE_CHECK.json')['improved']==4
    selected=read(D/'BOTTOM_UP_REVIEW.json')
    for name,h in selected['files'].items():assert sha(D/'site'/name)==h,'Selected animation changed'
    pins=read(R/'web/ADMISSION.json')['pins'].copy()
    assert all(sha(p)==h for p,h in pins.items()),'Frozen baseline changed'
    prior=len(pins)
    active=[D/'probe_endpoint_roles.py',D/'role_runtime.py',D/'serve.py',D/'thanks_runtime/worker.py',
            R/'endpoint_candidate.py',R/'probe_variants.py']
    for p in active:ast.parse(p.read_text('utf-8-sig'));pins[str(p)]=sha(p)
    for p in (D/'site').iterdir():
        if p.suffix in ['.html','.js','.css'] and not p.name.startswith('shape-study'):pins[str(p)]=sha(p)
    for row in read(D/'thanks_runtime/LOADED_WEIGHTS.json'):
        assert sha(row['path'])==row['sha256'],'Weight changed'
        pins[row['path']]=row['sha256']
    q=dict(admitted=True,at=datetime.datetime.now().astimezone().isoformat(),pins=pins,checks=checks,
           prior_pins_preserved=prior,training_runs=0,private_data_read=False,physical_camera_verified=False,
           user_ui_approval='그냥 이 느낌으로 해줘. 구름연기 질감도 넣지 마 그냥.',
           approved_reference=selected['animation'],endpoint_role_admitted=True,
           scope='Public frontal geometry/state integration. Four derived speed cases improved; not new-user accuracy. Frozen C/T3-F and all weights preserved. Smooth bottom-up UI explicitly approved.')
    (D/'web/ADMISSION.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ADMITTED',len(pins),'prior',prior)

if __name__=='__main__':main()
