from pathlib import Path
import json,hashlib,datetime,ast
D=Path(__file__).resolve().parent;P=D.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks={}
    for rel in ['qa/MEASURE_AFTER.json','qa/BROWSER_CHECK.json']:
        p=D/rel;assert read(p)['passed'];checks[str(p)]=sha(p)
    old=read(P/'web/ADMISSION.json');pins=old['pins'].copy()
    assert all(sha(p)==h for p,h in pins.items()),'Previous release changed'
    assert sha(D/'site/hologram.js')==sha(P/'site/hologram.js')
    a=(P/'site/scene.js').read_text('utf-8-sig');b=(D/'site/scene.js').read_text('utf-8-sig')
    assert a[a.index('async function openPractice()'):a.index('async function returnRoom()')]==b[b.index('async function openPractice()'):b.index('async function returnRoom()')],'Flight changed'
    for p in (D/'site').iterdir():
        if p.suffix in ['.html','.css','.js'] and not p.name.startswith('shape-study'):pins[str(p)]=sha(p)
    p=D/'serve.py';ast.parse(p.read_text('utf-8-sig'));pins[str(p)]=sha(p)
    out=dict(admitted=True,at=datetime.datetime.now().astimezone().isoformat(),pins=pins,checks=checks,prior_pins_preserved=len(old['pins']),
        scope='UI only: robot-anchored dialogue, larger angle panes, 30 fps demonstration with cached skin matrices. Capture/model/backend unchanged.',
        physical_camera_verified=False,private_data_read=False,training_runs=0)
    (D/'web/ADMISSION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('UI ADMITTED',len(pins),'previous',len(old['pins']))
if __name__=='__main__':main()
