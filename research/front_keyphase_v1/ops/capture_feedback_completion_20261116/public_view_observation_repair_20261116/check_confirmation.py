from view_common import *
from view_candidate import ViewLesson

def brief(q):return dict(events=len(q['events']),criteria={c['id']:c['status']for c in q['criteria']})
def main():
 torch.set_num_threads(2);out=J/'confirmation';old,new=PalmMotionLesson(),ViewLesson();rows=[];controls=[]
 write(out/'REVIEW.json',dict(reviewer='Codex source-image review; not clinical or linguistic expert validation',source='Native L/R contact sheets, reviewed before candidate prediction',observation='Visible upright right hand, supporting left palm approximately horizontal. Individual occluded joints and physical contact not labelled.',training=False,private_data=False))
 write(out/'MODEL_FREEZE.json',dict(pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[J/'view_candidate.py']+list((J/'plane').glob('*.pt'))+list((J/'palm').glob('*.pt'))}))
 for r in read(out/'INPUTS.json')['rows']:
  raw=dict(np.load(r['raw']));a=old.analyze(raw);b=new.analyze(raw);aa,bb=brief(a),brief(b);lost=[k for k,v in aa['criteria'].items()if v=='supported'and bb['criteria'][k]!='supported'];warn=[k for k,v in bb['criteria'].items()if v=='difference'and aa['criteria'][k]!='difference'];rows.append(dict(id=r['id'],before=aa,after=bb,lost=lost,new_warning=warn,passed=not lost and not warn and bb['events']>=aa['events']));write(out/(r['id']+'.json'),dict(before=a,after=b));print('UNSELECTED_VIEW',rows[-1],flush=True)
 for r in sources():
  if r['id']not in[f'NIA_SL_WORD1290_REAL{p:02d}_{v}'for p in[12,18]for v in['L','R']]:continue
  raw=dict(np.load(r['raw']));cut=48 if 'REAL12_'in r['id']else 62;hold=48 if 'REAL12_'in r['id']else 53;n=len(raw['frame_id'])
  for kind in['partial','constant']:
   z={k:v.copy()for k,v in raw.items()}
   if kind=='partial':
    for k in['xy_px','score','timestamp_us','frame_id']:z[k]=z[k][:cut]
   else:z['xy_px']=np.repeat(z['xy_px'][hold:hold+1],n,axis=0)
   a=old.analyze(z);b=new.analyze(z);controls.append(dict(id=r['id'],kind=kind,before=brief(a),after=brief(b),passed=len(b['events'])<2))
 write(out/'RESULT.json',dict(rows=rows,controls=controls,passed=all(r['passed']for r in rows+controls),candidate_selection=False,private_data=False));print('CONFIRMATION',all(r['passed']for r in rows+controls),flush=True)
if __name__=='__main__':main()
