from targets import *
from collections import Counter

def main():
 assert not (U/'EVAL_FROZEN.json').exists()
 checks=[]
 def check(name,value):
  assert value,name
  checks.append(name)
 labels=read(U/'LABELS.json')['rows'];oldlabels=read(P/'LABELS.json')['rows']
 check('prior annotations unchanged',all(r in labels for r in oldlabels))
 fit=[r for r in labels if r['role']=='fit'];dev=[r for r in labels if r['role']!='fit']
 check('fit dev source disjoint',not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev})
 check('fit dev record IDs disjoint',not {r['person'] for r in fit}&{r['person'] for r in dev})
 samples=read(U/'TRAIN_SAMPLES.json');used=[r for r in samples if 'sample' in r]
 check('multiple intervals contain outer boundaries',all(r['boundaries']==[1.,1.] for r in used if r['kind']=='multiple'))
 check('nonmoving boundaries are ignored',all(r['boundary_mask']==[0.,0.] for r in used if r['kind'] in ['hold','background']))
 check('ambiguous quality is ignored',all(r['quality_mask']==0 for r in used if r['kind'].endswith('ambiguous')))
 check('sample indices contiguous',sorted(r['sample'] for r in used)==list(range(len(used))))
 extra=[];original={r['word_id']:r for r in read(A/'EVALUATION_LABELS.json')['rows']}
 specs=[('WORD0003',0,[([74,76],[80,82],0),([81,83],[90,92],2)],[], 'dense_0003_14.jpg'),('WORD0500',1,[([54,58],[63,66],2)],[[69,87]],'NIA_SL_WORD0500_REAL14_F.jpg')]
 for word,axis,events,holds,sheetname in specs:
  r=copy.deepcopy(original[word]);r.update(axis=axis,moving='L',anchor='R',events=[dict(onset=a,offset=b,direction=d) for a,b,d in events],holds=holds,background=[],role='additional_left_evaluation',sheet=str(A/'public_rgb'/sheetname))
  r['sheet_sha256']=sha(r['sheet']);r['notes']='Same Codex RGB review before new scores. Physical left wrist/forearm movement only; no handshape, contact or semantic completion truth. Previously exposed public record14 videos, not fresh evaluation. WORD0500 left raising is support-hand preparation, not a required sign action.'
  check('extra evaluation source excluded from fit',r['source_sha256'] not in {a['source_sha256'] for a in fit});extra.append(r)
 write(U/'EXTRA_EVAL_LABELS.json',dict(rows=extra,independent_human=False,new_source_videos=0))
 for f in ['targets.py','train.py','boundary_core.py','LABELS.json']:
  check('frozen '+f,sha(U/f)==read(U/'FROZEN.json')['pins'][str(U/f)])
 write(U/'PREFLIGHT.json',dict(status='PASS',checks=checks,training_rows=len(samples),used=len(used),excluded=len(samples)-len(used),quality_masked=sum(r['quality_mask']==0 for r in used),multiple_rows=sum(r['kind']=='multiple' for r in used),fit_videos=len({r['clip_id'] for r in fit}),fit_events=sum(len(r['events']) for r in fit),left_axes=dict(Counter(str(r['axis']) for r in fit if r['moving']=='L')),extra_left_events=3))
 event('preflight_passed',checks=len(checks),samples=len(used))

if __name__=='__main__':main()
