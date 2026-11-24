from pathlib import Path
import sys
J=Path(__file__).resolve().parent;O=J.parent.parent
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import server
sys.path.insert(0,str(J))
from arrival_web_adapter import CaptureService
if __name__=='__main__':
 server.SERVICE=CaptureService()
 http=server.LocalServer(('127.0.0.1',server.PORT),server.Handler)
 server.threading.Thread(target=server.SERVICE.prepare,daemon=True).start()
 (server.U/'SERVER.json').write_text(server.json.dumps(dict(pid=server.os.getpid(),port=server.PORT,url=f'http://127.0.0.1:{server.PORT}/',camera_at_start=False,entrypoint=str(Path(__file__).resolve())),indent=2),encoding='utf-8')
 print(f'Local http://127.0.0.1:{server.PORT}/ camera starts only on user command',flush=True)
 http.serve_forever()
