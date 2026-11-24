from pathlib import Path
import json
from html.parser import HTMLParser
root=Path(__file__).resolve().parent
repo=root.parents[3]
for name in ['FROZEN.json','FINAL_AUDIT.json','CURRENT_EXECUTION.json','VERIFICATION.json','END_TO_END.json','STREAMING.json','TRAINING_READINESS.json']:
    json.loads((root/name).read_text(encoding='utf-8'))
state=json.loads((root.parent/'STATE.json').read_text(encoding='utf-8'))
assert state['current_stage']=='SIGNREP_GPU_RUNTIME_VERIFIED_KSL_TRAINING_NOT_STARTED'
handoff=(repo/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf-8')
assert root.name in handoff and '3.23' in handoff
context=(repo/'WordsInSilence_PROJECT_CONTEXT.md').read_text(encoding='utf-8')
assert context.count('### 2026-10-29 SignRep 승인 실행')==2
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.extend(v for k,v in attrs if k=='href')
parser=Links();parser.feed((root/'RESULT_KO.html').read_text(encoding='utf-8'))
for href in parser.links:
    if not href.startswith('http'):assert (root/href).exists(),href
running=json.loads((root/'CURRENT_EXECUTION.json').read_text(encoding='utf-8'))['running']
assert not running
out=dict(status='PASS',json_and_document_sync=True,report_links=parser.links,running=running,html_scope='Static HTML syntax/links verified; browser panel queued, not screenshot-verified')
(root/'DELIVERY_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
