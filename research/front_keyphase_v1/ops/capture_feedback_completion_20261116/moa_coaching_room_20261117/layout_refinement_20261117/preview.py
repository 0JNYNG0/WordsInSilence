"""UI-only preview: no inference engines or physical camera can start here."""
from pathlib import Path
import sys, threading
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent))
import serve as prior
prior.D=D
s=prior.server
class DisabledCamera:
    state='ready';result=None;camera_thread=None
    def status(self):return dict(state='ready',camera_thread_active=False)
class Handler(prior.Handler):
    def do_POST(self):self.reply({'error':'UI preview only; camera disabled'},403)
    def do_GET(self):
        if self.path.startswith('/preview.mjpeg'):self.send_error(404);return
        super().do_GET()
if __name__=='__main__':
    s.PORT=8822;s.SERVICE=DisabledCamera()
    http=s.LocalServer(('127.0.0.1',8822),Handler)
    print('UI preview 8822, physical camera disabled',flush=True)
    http.serve_forever()
