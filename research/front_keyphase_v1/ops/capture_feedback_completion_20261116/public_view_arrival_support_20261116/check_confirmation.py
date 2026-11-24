from preserved_arrival import *
def brief(q):return dict(events=len(q['events']),criteria={c['id']:c['status']for c in q['criteria']})
def main():
 torch.set_num_threads(2);old,new=PlaneViewLesson(),PreservedArrivalLesson();out=K/'confirmation';rows=[];controls=[]
 write(out/'REVIEW.json',dict(reviewer='Codex original RGB contact-sheet review before candidate outputs',observation='Right hand upright over left supporting palm; two visible approach cycles. No hidden finger or physical3Dcontact label.',scope='Unselected camera views of knownFITperson6, not unseen-person accuracy.'))
 for r in read(out/'INPUTS.json')['rows']:
  z=dict(np.load(r['raw']));a=old.analyze(z);b=new.analyze(z);aa,bb=brief(a),brief(b);passed=all(bb['criteria'][k]=='supported'for k,v in aa['criteria'].items()if v=='supported')and not any(v=='difference'and aa['criteria'][k]!='difference'for k,v in bb['criteria'].items())and all(e in b['events']for e in a['events']);rows.append(dict(id=r['id'],before=aa,after=bb,passed=passed));write(out/(r['id']+'.json'),dict(before=a,after=b));print('NEW_VIEW',rows[-1],flush=True)
 for r in sources():
  if r['id']not in[f'NIA_SL_WORD1290_REAL{p:02d}_{v}'for p in[12,18]for v in['L','R']]:continue
  z=dict(np.load(r['raw']));cut=48 if 'REAL12_'in r['id']else 62;hold=48 if 'REAL12_'in r['id']else 53;n=len(z['frame_id'])
  for kind in['partial','constant']:
   raw={k:v.copy()for k,v in z.items()}
   if kind=='partial':
    for k in['xy_px','score','timestamp_us','frame_id']:raw[k]=raw[k][:cut]
   else:raw['xy_px']=np.repeat(raw['xy_px'][hold:hold+1],n,axis=0)
   q=new.analyze(raw);controls.append(dict(id=r['id'],kind=kind,after=brief(q),passed=len(q['events'])<2))
 write(out/'RESULT.json',dict(rows=rows,controls=controls,passed=all(r['passed']for r in rows+controls),private_data=False));print('CONFIRMED',all(r['passed']for r in rows+controls),flush=True)
if __name__=='__main__':main()
