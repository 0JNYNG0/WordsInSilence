from graph import *
from source_decoder import *
from collections import Counter
O=U.parent;R=O/'word100_candidate_role_learning_20261111';D=O/'word100_action_feedback_demo_20261108'
def main():
 assert not (U/'SOURCE_DECODER_RESULTS.json').exists()
 cards=read(D/'CARDS.json')['words'];families=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words'];cfg=read(D/'FROZEN.json')['phase'];src=read(R/'EVALUATION_INPUTS.json')['rows']
 write(U/'SOURCE_DECODER_FROZEN.json',dict(reason='Confirmed source-code prior expected-order selection bug; use unchanged source phase network with independent runs',threshold=cfg['threshold'],rank_or_time_prior=False,require_all_occurrences_order_consistent=True,new_training=False,old_model_inference=False,source_decoder_sha=hashlib.sha256((U/'source_decoder.py').read_bytes()).hexdigest()))
 rr=[]
 for s in src:
  p=Path(s['baseline']);assert hashlib.sha256(p.read_bytes()).hexdigest()==s['baseline_sha256'];b=read(p);word=s['clip_id'].split('_')[2]
  occ=phase_occurrences(b['phase_probability'],b['observed_hand_mask'],b['timestamp_us'],families[word],cards[word],cfg['threshold'])
  rr.append(dict(clip_id=s['clip_id'],kind=s['kind'],occurrences=occ,pairs=phase_pairs(occ,families[word]),source=str(p),source_sha=s['baseline_sha256']))
 write(U/'SOURCE_DECODER_RESULTS.json',rr);summary={k:dict(Counter(p['state'] for r in rr if r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in rr})};write(U/'SOURCE_DECODER_COMPLETE.json',dict(summary=summary,conditions=len(rr),observation_model_unchanged=True));print(summary)
if __name__=='__main__':main()
