"""End-to-end cold/warm benchmark on existing public seated-camera crops."""
from pathlib import Path
import json,time
from live_adapter import U,O,analyze,warmup,write
def main():
 assert not (U/'LIVE_PUBLIC_EVALUATION.json').exists()
 cold=warmup();rows=[]
 for word in ('WORD0739','WORD1133','WORD1567','WORD2523'):
  source=O/'word100_seated_web_framing_20261102/public_crops'/f'NIA_SL_{word}_REAL12_F.mp4'
  dest=U/'live_public'/word
  if (dest/'RESULT.json').exists():r=json.loads((dest/'RESULT.json').read_text('utf8'))
  else:r=analyze(source,dest,word,progress=lambda e:print(word,json.dumps(e,ensure_ascii=False),flush=True))
  rows.append(dict(word_id=word,source=str(source),result_path=str(dest/'RESULT.json'),latency=r['latency'],device=r['device'],
   statuses=[s['status'] for s in r['stages']],source_sha256=r['source_sha256']))
  (U/'LIVE_PUBLIC_PROGRESS.json').write_text(json.dumps(dict(done=len(rows),last=word)),encoding='utf8')
 write(U/'LIVE_PUBLIC_EVALUATION.json',dict(rows=rows,cold_start=cold,private_inputs=0,training=False,scope='Existing public ID12 seated-view crops; development integration test, not private-user accuracy'))
if __name__=='__main__':main()
