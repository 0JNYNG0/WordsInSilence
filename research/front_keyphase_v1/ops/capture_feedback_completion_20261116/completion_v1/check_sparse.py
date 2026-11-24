from observed_gap import *
sys.path.insert(0,str(V.parent));from diagnose_terminal import sources
def main():
 torch.set_num_threads(2);out=V/'sparse_checks';out.mkdir(exist_ok=True);assert not(out/'FROZEN.json').exists()
 write(out/'FROZEN.json',dict(cause='One missing hand frame invalidated seven motion windows, erasing a repeated movement.',
  repair='Use six actually observed samples in an otherwise valid seven-frame window; at most one internal gap <=100ms. Retain missing masks/fingers; require observed terminal state.',
  gate='Known normal6 unchanged; single internal missing frame recovers counted task when enough actual evidence remains; long gaps and partial/static remain incomplete.',
  code_sha=sha(V/'observed_gap.py'),thresholds_or_weights_changed=False,private=False))
 net=SparseVisible();rows=[]
 inputs=sources()+read(V/'holdout/EXTRACTED.json')['rows']
 for r in inputs:
  raw=dict(np.load(r['raw_path']));base=net.analyze(raw);a=base['events'][0]['start'];b=base['events'][-1]['end'];mid=(a+b)//2
  for kind in['normal','one_missing','seven_missing','first_only','constant']:
   z={k:v.copy()for k,v in raw.items()}
   if kind=='one_missing':z['xy_px'][mid,112:133]=np.nan
   if kind=='seven_missing':z['xy_px'][mid-3:mid+4,112:133]=np.nan
   if kind=='first_only':
    for k in['xy_px','score','timestamp_us','frame_id']:z[k]=z[k][:base['events'][0]['end']+2]
   if kind=='constant':z['xy_px']=np.repeat(z['xy_px'][a:a+1],len(z['frame_id']),axis=0)
   q=net.analyze(z);supported=all(c['status']=='supported'for c in q['criteria']);expected=kind in['normal','one_missing']
   rows.append(dict(person=r['person'],kind=kind,supported=supported,check=supported==expected,criteria=q['criteria'],events=q['events']))
 write(out/'RESULT.json',dict(rows=rows,passed=sum(r['check']for r in rows),total=len(rows)))
 print('SPARSE',sum(r['check']for r in rows),len(rows))
 for r in rows:
  if not r['check']:print(r['person'],r['kind'],{c['id']:c['status']for c in r['criteria']},len(r['events']))
if __name__=='__main__':main()
