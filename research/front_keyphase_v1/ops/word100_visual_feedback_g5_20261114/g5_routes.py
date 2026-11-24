"""Read-only, explicit allowlist of public G4 media and result artifacts."""
from pathlib import Path
import json
U=Path(__file__).resolve().parent
ROUTES=json.loads((U/'ROUTES.json').read_text('utf-8'))
STATIC={'index.html','style.css','viewer.js','catalog.json'}
def handle_get(handler,route):
    if route=='/feedback-g5':
        handler.send_response(302);handler.send_header('Location','/feedback-g5/');handler.end_headers();return True
    if not route.startswith('/feedback-g5/'):return False
    name=route.removeprefix('/feedback-g5/') or 'index.html'
    if name in STATIC:handler.file(U/'site'/name)
    elif route in ROUTES:handler.file(Path(ROUTES[route]))
    else:handler.send_error(404)
    return True
