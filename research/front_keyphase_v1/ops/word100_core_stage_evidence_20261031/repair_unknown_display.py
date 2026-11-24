from pathlib import Path
import json
U=Path(__file__).resolve().parent;p=U/'index.html';s=p.read_text(encoding='utf8')
old='for(let raw of q.rgb_review?.unknown||[])'
new="for(let raw of (Array.isArray(q.rgb_review?.unknown)?q.rgb_review.unknown:Object.entries(q.rgb_review?.unknown||{}).map(([k,v])=>({reason_ko:k+': '+(Array.isArray(v)?v.join(' / '):String(v))}))))"
assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf8')
print(json.dumps(json.loads((U/'PHASE_FIT.json').read_text(encoding='utf8'))['clips'][0]['unknown'],ensure_ascii=False))
