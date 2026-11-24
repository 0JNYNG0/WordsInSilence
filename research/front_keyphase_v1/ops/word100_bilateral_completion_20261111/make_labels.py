from bilateral_core import *

def main():
 assert not (U/'LABELS.json').exists()
 reviews={r['clip_id']:r for r in read(U/'REVIEW_READY.json')['rows']}
 # Index ranges are uncertainty intervals from original RGB contact sheets,
 # not a claim of one-frame expert annotation. Left = performer's left.
 definitions={
 ('1567',1): (4,[(52,54,59,61,2),(59,61,67,69,0),(67,69,77,79,2)],[(83,94)],[(31,44)]),
 ('1567',3): (4,[(42,44,49,51,2),(49,51,59,61,0),(59,61,67,70,2)],[(72,83)],[(24,29)]),
 ('1567',5): (4,[(52,54,61,63,2),(61,63,72,74,0),(72,74,83,85,2)],[(87,90)],[(24,35)]),
 ('1567',11):(4,[(49,51,59,61,2),(59,61,69,71,0),(69,72,79,82,2)],[(85,95)],[(28,38)]),
 ('1567',12):(4,[(53,55,58,60,2),(58,60,68,70,0),(68,70,78,81,2)],[(84,89)],[(24,35)]),
 ('1567',13):(4,[(39,41,48,50,2),(49,51,61,63,0),(62,64,73,75,2)],[(78,89)],[(24,27)]),
 ('2523',1): (0,[(38,40,43,45,0)],[(46,53)],[(17,23)]),
 ('2523',3): (0,[(66,68,73,75,0)],[(76,84)],[(25,42)]),
 ('2523',5): (0,[(62,64,68,70,0)],[(72,77)],[(21,44)]),
 ('2523',11):(0,[(56,58,61,64,0)],[(66,76)],[(21,33)]),
 ('2523',12):(0,[(56,58,61,64,0)],[(66,76)],[(21,33)]),
 ('2523',13):(0,[(62,64,68,70,0)],[(72,80)],[(24,43)]),
 }
 added=[]
 for (w,p),(axis,evs,holds,bg) in definitions.items():
  clip=f'NIA_SL_WORD{w}_REAL{p:02d}_F';r=reviews[clip]
  assert sha(r['sheet'])==r['sheet_sha256']
  added.append(dict(clip_id=clip,person=p,role='fit' if p<10 else 'development',axis=axis,moving='L',anchor='R',events=[dict(onset=[a,b],offset=[c,d],direction=y) for a,b,c,d,y in evs],holds=holds,background=bg,source_sha256=r['source_sha256'],sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],reviewer='Codex RGB inspection, not independent human/expert',notes='Visible left wrist movement only; not finger closure/contact/semantic correctness. Interval boundary uncertainty retained. No synthetic left/right flip.'))
 rows=read(A/'FIT_LABELS.json')['rows']+[r for r in read(B/'OBSERVED_BOUNDARIES.json')['rows'] if not r['role'].startswith('fit')]+added
 for r in rows:
  r['original_role']=r['role'];r['role']='fit' if r['role'].startswith('fit') else 'development'
 fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']!='fit']
 assert not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev}
 assert not {r['person'] for r in fit}&{r['person'] for r in dev}
 out=dict(rows=rows,added_left_rows=len(added),label_meaning='Observed physical movement interval, NOT whole sign completion; fragments are derived supervised spans',fit_clips=len({r['clip_id'] for r in fit}),dev_clips=len({r['clip_id'] for r in dev}),fit_events=sum(len(r['events']) for r in fit),dev_events=sum(len(r['events']) for r in dev),new_independent_expert_labels=0,development_previously_exposed=True)
 write(U/'LABELS.json',out)
 write(U/'FIRST_READ_ERROR.json',dict(action='Read guessed DEV_LABELS.json',error='File did not exist; located actual OBSERVED_BOUNDARIES.json by file listing',data_changed=False))
 write(U/'FIRST_LABEL_ERROR.json',dict(error='AssertionError: disjoint person sets',cause='fit_additional was mistakenly treated as development by exact role string comparison',fix='Normalize existing fit-prefixed roles without changing their split',first_source='make_labels_first.py.txt',training_had_started=False))
 event('labels_frozen',**{k:v for k,v in out.items() if k!='rows'})
if __name__=='__main__':main()
