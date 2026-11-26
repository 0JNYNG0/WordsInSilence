"""Check tracked public source files without executing project/model code.

Run together with: gitleaks git --redact --log-opts=--all
This guard is not a complete privacy or licensing audit.
"""
from pathlib import Path
import re,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
DENIED={'.csv','.tsv','.jsonl','.npz','.npy','.pt','.pth','.ckpt','.onnx','.pkl',
        '.joblib','.mp4','.mov','.avi','.webm','.wav','.mp3','.png','.jpg','.jpeg',
        '.gif','.zip','.7z','.fbx','.glb','.blend','.exe','.dll','.pem','.key'}
SAFE_JSON={'Packages/manifest.json','Packages/packages-lock.json',
           'ProjectSettings/SceneTemplateSettings.json','docs/runtime-environment.json'}
paths=[x.decode('utf-8')for x in subprocess.check_output(
    ['git','ls-files','-z'],cwd=ROOT).split(b'\0')if x]
failures=[]
for rel in paths:
    p=ROOT/rel
    if p.suffix.lower()in DENIED or p.name.startswith('.env'):
        failures.append((rel,'data/media/model/credential file type'))
    if p.suffix.lower()=='.json' and rel not in SAFE_JSON:
        failures.append((rel,'JSON is not on the reviewed configuration allowlist'))
    if '/Fixtures/'in rel or rel.startswith(('data/','LocalMedia/','Validation/')):
        failures.append((rel,'local data folder'))
    actual=Path('\\\\?\\'+str(p))if sys.platform=='win32'else p
    text=actual.read_text('utf-8-sig')
    for m in re.finditer(r'(?i)[A-Z]:[\\/]+Users[\\/]+([^\\/\s\x22\x27<>]+)',text):
        if m[1]!='local-user':failures.append((rel,'non-generic user home path'))
    if re.search(r'data:(?:image|video|audio)/[^\s;]+;base64,[A-Za-z0-9+/=]{128,}',text):
        failures.append((rel,'embedded media'))
for path,reason in failures:print(path+': '+reason)
print(f'Checked {len(paths)} tracked files; {len(failures)} unresolved publication issues.')
raise SystemExit(1 if failures else 0)
