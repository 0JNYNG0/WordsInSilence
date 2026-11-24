from pathlib import Path
import sys
D=Path(__file__).resolve().parent;P=D.parent/'public_frontal_evidence_repair_20261117';O=D.parent.parent
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import server
sys.path.insert(0,str(D))
from retry_runtime import CaptureService
class Handler(server.Handler):
    def do_GET(self):
        if server.urlsplit(self.path).path=='/live.js':
            if not self.valid_host():self.reply({'error':'Local host only'},403);return
            self.file(P/'site/live.js');return
        super().do_GET()
if __name__=='__main__':
    server.SERVICE=CaptureService()
    http=server.LocalServer(('127.0.0.1',server.PORT),Handler)
    server.threading.Thread(target=server.SERVICE.prepare,daemon=True).start()
    (D/'SERVER.json').write_text(server.json.dumps(dict(pid=server.os.getpid(),port=server.PORT,url=f'http://127.0.0.1:{server.PORT}/',camera_at_start=False,entrypoint=str(Path(__file__).resolve())),indent=2),encoding='utf-8')
    print(f'Local http://127.0.0.1:{server.PORT}/ camera starts only on user command',flush=True)
    http.serve_forever()
