"""Current web neural path on NEW public coordinate inputs only; no server/camera."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_action_feedback_demo_20261108'
sys.path.insert(0,str(D));sys.dont_write_bytecode=True
from consumer import FeedbackConsumer
def main():
 target=U/'FULL_REVERSE_WEB_COMPLETE.json';assert not target.exists()
 read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
 net=FeedbackConsumer();out=[]
 for s in read(U/'FULL_REVERSE_INPUT_COMPLETE.json')['rows']:
  z=np.load(s['path']);t=time.perf_counter();result=net.channels(z['x'],z['ts'],z['frame_id'],s['word_id']);p=U/'full_reverse_web'/f"{s['clip_id']}.json";p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
  out.append(dict(**s,result=str(p),result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seconds=time.perf_counter()-t));print(s['clip_id'],out[-1]['seconds'],flush=True)
 target.write_text(json.dumps(dict(rows=out,camera=False,server_started=False,normal_not_replayed=True),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
