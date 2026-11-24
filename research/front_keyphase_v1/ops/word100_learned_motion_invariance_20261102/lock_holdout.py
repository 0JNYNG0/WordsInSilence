"""Metadata-only lock. Does not load old classifier scores or new video predictions."""
from prepare_real_fit import N,O,P,M,read,sha,exclusive
def main():
 original=read(P/'INPUT_MANIFEST.json')['rows'];known={r['source_sha256'] for r in original};fresh=O/'word100_autonomous_recognition_20261026/fresh199';rows=[]
 for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
  for pid in [14,15]:
   p=fresh/'data/records'/f'NIA_SL_{word}_REAL{pid:02d}_F.json';r=read(p)
   assert r['signer_id']==pid and r['word_code']==word and r['source_sha256'] not in known
   rows.append(dict(clip_id=r['clip_id'],person=pid,word_id=word,source_sha256=r['source_sha256'],source=r['source'],record=str(p),record_sha256=sha(p),frames=r['frames']))
 assert len({r['source_sha256'] for r in rows})==8
 exclusive(N/'HELDOUT_LOCK.json',dict(rows=rows,role='Locked additional recorded IDs for feedback candidate; not tuning or model selection',
  past_exposure='Used for prior TCN ensemble evaluation2026-10-26. Not globally untouched. No previous scores read in this audit.',
  current_training_excluded=True,exact_duplicate_against48_sources=False,recorded_ids_not_biometric_identity=True,
  execute_after='Freeze candidate/criteria using FIT and developmental diagnosis. Independently annotate physical intervals before new predictions; run final comparison once.',
  quarantined_id16_preserved=True))
 print('8 records locked; no new video inference, no old score access')
if __name__=='__main__':main()
