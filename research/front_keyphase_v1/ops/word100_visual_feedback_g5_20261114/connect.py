"""Minimal reversible route addition to current research UI; model code untouched."""
from prepare import U,O,write,sha
import shutil
base=O/'web_tutorial_live_integration_20261113'
paths=[base/'server.py',base/'site/index.html']
records=[]
for p in paths:
    backup=U/'before'/p.name
    if not backup.exists():shutil.copy2(p,backup)
    records.append(dict(path=str(p),backup=str(backup),before_sha256=sha(backup)))
p=paths[0];s=p.read_text('utf-8-sig')
if 'from g5_routes import handle_get' not in s:
    s=s.replace('PORT=8821;',"sys.path.insert(0,str(O/'word100_visual_feedback_g5_20261114'))\nfrom g5_routes import handle_get as handle_g5_get\nPORT=8821;")
    s=s.replace("        try:\n            if route=='/api/meta':","        try:\n            if handle_g5_get(self,route):return\n            if route=='/api/meta':",1)
p.write_text(s,encoding='utf-8')
p=paths[1];s=p.read_text('utf-8-sig')
if 'id="g5-review-link"' not in s:
    s=s.replace('<div class="top-controls">','<div class="top-controls"><a id="g5-review-link" href="/feedback-g5/" style="color:#203954;background:#fff;padding:9px 12px;border:1px solid #bcc9d7;border-radius:8px;text-decoration:none;font:13px sans-serif">새 피드백 · 공개 검증 결과 ↗</a>',1)
p.write_text(s,encoding='utf-8')
for r in records:r['after_sha256']=sha(r['path'])
write(U/'CONNECTION.json',dict(files=records,model_replaced=False,capture_logic_changed=False,route='/feedback-g5/',live_inference='existing live branch unchanged; new qualified model capture integration remains G6',rollback='Run rollback.py to restore only these two files; new artifacts remain for audit.'))
print('Current 8821 web: public evidence viewer route and link added; inference unchanged.')
