from pathlib import Path
import json, urllib.request, hashlib, socket, datetime
D=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def main():
    base='http://127.0.0.1:8821'
    with urllib.request.urlopen(base+'/api/meta') as r:m=json.load(r)
    req=urllib.request.Request(base+'/api/status',headers={'X-Session-Token':m['token']})
    with urllib.request.urlopen(req) as r:s=json.load(r)
    assert s['state']=='ready' and not s['camera_thread_active']
    for k in ['frontal_scope_active','evidence_repair_active','selective_retry_active','stop_short_cycle_active','rest_stability_active','endpoint_role_active']:assert s[k],k
    selected=read(D/'BOTTOM_UP_REVIEW.json')
    for name,h in selected['files'].items():
        with urllib.request.urlopen(base+'/'+name) as r:assert hashlib.sha256(r.read()).hexdigest()==h
    a=read(D/'web/ADMISSION.json');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
    with socket.socket() as sock:preview_closed=sock.connect_ex(('127.0.0.1',8822))!=0
    assert preview_closed
    out=dict(passed=True,at=datetime.datetime.now().astimezone().isoformat(),url=base+'/',
             server=read(D/'SERVER_8821.json'),state=s['state'],camera_active=False,preview_8822_stopped=preview_closed,
             selected_effect_matches=True,frozen_pins=len(a['pins']),prior_pins_preserved=a['prior_pins_preserved'],
             ui=s['companion_ui_version'],physical_camera_verified=False,private_data_read=False,
             corrected_deployment_issue='First restart command matched no server because of PowerShell array grouping; old server remained, bind failed. Corrected explicit path array, verified idle ownership, stopped old trees, restarted and checked actual 8821 state.')
    (D/'web/DEPLOYMENT_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:out[k] for k in ['passed','url','state','camera_active','preview_8822_stopped','frozen_pins']},ensure_ascii=False))
if __name__=='__main__':main()
