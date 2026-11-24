from pathlib import Path
import importlib.util,json
E=Path(__file__).resolve().parent;D=E.parent/'public_shape_endpoint_learning_20261116'
s=importlib.util.spec_from_file_location('http_prior_fixture',D/'check_http.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.D=E
if __name__=='__main__':
 m.main();meta=m.request('/api/meta');q=m.request('/api/status',meta['token']);assert q['motion_evidence_active']and not q['camera_thread_active']
 p=E/'web/HTTP_CHECK.json';out=json.loads(p.read_text(encoding='utf-8'));out['motion_evidence_active']=q['motion_evidence_active'];out['server']=json.loads((E.parent.parent/'web_tutorial_live_integration_20261113/SERVER.json').read_text(encoding='utf-8'));p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 print('BODY_PALM_MODULE_ACTIVE_CAMERA_OFF',out['server']['pid'])
