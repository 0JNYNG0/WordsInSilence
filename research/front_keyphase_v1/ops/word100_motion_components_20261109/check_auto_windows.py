from common import *
from motion import *

def main():
 assert not (U/'AUTO_WINDOW_CHECK.json').exists()
 p=O/'word100_positive_direction_20261109';rows=read(p/'COMPARISON.json')['rows'];s=torch.load(U/'runs/motion_only_42/model.pt',map_location='cpu',weights_only=True);model=MotionNet();model.load_state_dict(s['model']);model.eval();out=[]
 with torch.inference_mode():
  for r in rows:
   old=read(r['decision'])['after'];z=np.load(r['prediction']);assert sha(r['prediction'])==r['sha256']
   for c in old['cards']:
    m=c.get('motion_evidence');span=m.get('span') if m else None
    if not span:continue
    f,eligible,info=features(z['x'],z['timestamp_us'],*span,'along_other_forearm',m['moving_hand'],m['anchor_arm']);prob=model((torch.tensor(f)-s['mean'])/s['std']).softmax(-1).numpy();pred=int(prob.argmax());conf=float(prob.max())
    status='unknown' if not eligible or conf<.65 or pred==1 else 'supported' if (pred-1)==m['expected_sign'] else 'contradicted'
    out.append(dict(clip_id=r['clip_id'],variant=r['kind'],card=c['id'],span=span,old_state=m['state'],state=status,pred=pred,confidence=conf,eligible=bool(eligible),evidence=info))
 write(U/'AUTO_WINDOW_CHECK.json',dict(rows=out,summary={v:{k:sum(q['variant']==v and q['state']==k for q in out) for k in ['supported','contradicted','unknown']} for v in sorted({r['variant'] for r in out})},scope='Saved actual automatic phase windows, two displayed cards share same stroke; not independent new examples',no_private=True))
 print(json.dumps(read(U/'AUTO_WINDOW_CHECK.json')['summary'],ensure_ascii=False))
if __name__=='__main__':main()
