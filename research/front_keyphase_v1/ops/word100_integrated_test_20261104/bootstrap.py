"""Create isolated integration from the frozen AM runtime; never alter AM."""
from pathlib import Path
import json, shutil
U=Path(__file__).resolve().parent
AM=U.parent/'word100_local_event_demo_20261103'
for name in ['run_demo.py','learner_adapter.py','order_event_worker.py','local_runtime.py','parallel_capture.py']:
    p=U/name
    if p.exists(): raise RuntimeError('Already bootstrapped: '+str(p))
    shutil.copy2(AM/name,p)
p=U/'run_demo.py';s=p.read_text(encoding='utf8').replace('PORT = 8810','PORT = 8812')
s=s.replace("U / 'learner_adapter.py'", "U / 'integrated_adapter.py'")
s=s.replace("write_new(U / 'DEMO_LAUNCH.json'", "write_new(U / ('DEMO_LAUNCH_' + str(os.getpid()) + '.json')")
s=s.replace("self.assets = public_assets", "self.preview = None\n        self.assets = public_assets")
s=s.replace("return result\n\n    def perform", "return self.adapter.enrich_saved(result, Path(record['result_path']).parent)\n\n    def perform")
s=s.replace("if route == '/api/state':", "if route == '/style.css':\n                    return self.send_file(app.unit / 'style.css', 'text/css; charset=utf-8')\n                if route == '/logic.js':\n                    return self.send_file(app.unit / 'logic.js', 'text/javascript; charset=utf-8')\n                match_ref = re.fullmatch(r'/api/reference/(WORD\\d{4})', route)\n                if match_ref and match_ref[1] in app.words:\n                    return self.send_json(read(app.unit / 'reference_data.json')[match_ref[1]])\n                if route == '/api/state':")
s=s.replace("if parsed.path != '/api/analyze':", "if parsed.path == '/api/preview':\n                    return self.preview_frame(parsed)\n                if parsed.path != '/api/analyze':")
marker='            def do_POST(self):'
preview='''            def preview_frame(self, parsed):
                if parsed.query or self.headers.get('X-Session-Token') != app.token:
                    return self.send_json({'error':'Invalid preview request'},403)
                if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length',[])) != 1:
                    return self.send_json({'error':'Invalid length'},400)
                try: length=int(self.headers['Content-Length'])
                except (TypeError,ValueError): return self.send_json({'error':'Invalid length'},400)
                if not 0 < length <= 300000: return self.send_json({'error':'Preview too large'},413)
                if app.busy(): return self.send_json({'error':'Analysis running'},409)
                self.connection.settimeout(5)
                body=self.rfile.read(length)
                if len(body)!=length: return self.send_json({'error':'Incomplete preview'},400)
                try:
                    return self.send_json(app.adapter.preview(body))
                except Exception:
                    app.first_failure('PREVIEW_FIRST_FAILURE.txt')
                    return self.send_json({'error':'손내림 확인이 어렵습니다. 수동 종료를 이용하세요.'},503)

'''
s=s.replace(marker,preview+marker)
p.write_text(s,encoding='utf8')
(U/'SCOPE.json').write_text(json.dumps(dict(
    request='2026-11-04 integrated live TCN + learned SignRep/DTW visual feedback page',
    author='Codex', port=8812, research_training='remains paused',
    allowed=['isolated web integration','explicit user capture for inference only','public integration tests'],
    preserved=['C','T3-F','frozen AM pipeline','AO paused research','all previous demos'],
    forbidden=['training','user/mentor tuning or model selection','new external models','external transmission'],
    acceptance=['3 second countdown','conservative hand-lowering auto stop','horizontal blind TCN top3',
       'existing single learned-prototype DTW','native-time visual evidence','uncertainty distinct from difference'],
    no_accuracy_claim=True),ensure_ascii=False,indent=2),encoding='utf8')
print('Isolated AM copy created')
