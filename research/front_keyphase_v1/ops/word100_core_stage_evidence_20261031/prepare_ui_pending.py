from pathlib import Path
import json,numpy as np
U=Path(__file__).resolve().parent
rows=json.loads((U/'AUTHORIZATION.json').read_text(encoding='utf-8-sig'))['rows'];clips=[]
for r in rows:
 with np.load(r['input'],allow_pickle=False) as a:seconds=(a['timestamp_us']/1e6).tolist()
 clips.append({k:r[k] for k in ['alias','label','word_id','person','role','frames']}|dict(seconds=seconds,segments=[],source_notes=['원본 공개 영상은 바로 볼 수 있습니다. 단계별 RGB 검토 초안은 준비 중입니다.']))
p=U/'PHASE_DATA.json'
if not p.exists():p.write_text(json.dumps(dict(status='RGB_REVIEW_RUNNING',clips=clips),ensure_ascii=False,indent=2),encoding='utf8')
p=U/'index.html';s=p.read_text(encoding='utf8').replace('.bar{display:flex;', '.bar{position:relative;').replace('.bar button{min-width:0;', '.bar button{position:absolute;height:100%;min-width:0;')
s=s.replace("b.style.flex=String(s.core[1]-s.core[0]+1);", "b.style.left=(100*s.core[0]/q.frames)+'%';b.style.width=(100*(s.core[1]-s.core[0]+1)/q.frames)+'%';")
s=s.replace('지원되는 31구간을 160회 학습했고','관절 정보가 충분한 31구간을 160번 반복해 학습했고')
p.write_text(s,encoding='utf8')
