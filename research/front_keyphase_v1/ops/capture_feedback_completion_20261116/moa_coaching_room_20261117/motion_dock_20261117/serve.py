"""UI refinement only; reuse the admitted feedback/capture implementation."""
from pathlib import Path
import sys,json,threading,os
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent))
import serve as prior
from upgraded_runtime import admitted
prior.D=D
s=prior.server
class CaptureService(prior.CaptureService):
    def prepare(self):
        self.layout_admitted=False;super().prepare()
        if self.state!='ready':return
        self.state='preparing'
        try:
            self.layout_admitted=admitted(D/'web/ADMISSION.json')
            if not self.layout_admitted:raise RuntimeError('화면 검증 파일이 일치하지 않습니다.')
            self.state='ready'
        except Exception as exc:self.state='error';self.error=str(exc)
    def status(self):
        q=super().status();q.update(companion_ui_version='moa_coaching_room_v3',layout_admitted=getattr(self,'layout_admitted',False))
        if q['state']=='ready' and not q['layout_admitted']:q['state']='preparing'
        return q
if __name__=='__main__':
    s.PORT=8821;s.SERVICE=CaptureService()
    http=s.LocalServer(('127.0.0.1',s.PORT),prior.Handler)
    threading.Thread(target=s.SERVICE.prepare,daemon=True).start()
    (D/'SERVER.json').write_text(json.dumps(dict(pid=os.getpid(),port=s.PORT,entrypoint=str(Path(__file__).resolve()),camera_at_start=False)),encoding='utf-8')
    print('Local http://127.0.0.1:8821/ — UI v2, camera off at start',flush=True)
    http.serve_forever()
