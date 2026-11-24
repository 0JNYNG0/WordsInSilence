from form_candidate import *
from form_v5 import QualifiedFormLesson
def main():
 torch.set_num_threads(2);old,new=ThanksCandidate(),QualifiedFormLesson();checks=[]
 write(D/'final_confirmation/REVIEW.json',dict(reviewer='Codex native RGB review, not expert annotation',source='SOURCE_REVIEW.jpg',observation='Right open hand and left supporting hand around frames84-112, followed by lowering. Initial43-78 panels were pre-action and not used as gesture truth.',physical_contact=None,source_never_used_for_fitting=True))
 for r in read(D/'final_confirmation/INPUTS.json')['rows']:
  z=dict(np.load(r['raw']));a=old.analyze(z);b=new.analyze(z);aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']};passed=all(bb[k]==v for k,v in aa.items()if v!='unknown')and not any(v=='difference'and aa[k]!='difference'for k,v in bb.items())and len(a['events'])==len(b['events']);checks.append(dict(id=r['id'],before=aa,after=bb,events_before=len(a['events']),events_after=len(b['events']),passed=passed));write(D/'final_confirmation'/(r['id']+'.json'),dict(before=a,after=b))
 write(D/'final_confirmation/RESULT.json',dict(rows=checks,passed=all(r['passed']for r in checks),scope='First evaluation of these cameras after final candidate fixed. Front-person data known; not unseen-signer accuracy.'));assert all(r['passed']for r in checks),checks
 q=read(D/'qualified_checks/RESULT.json');assert q['new_wrong']==q['lost_correct']==0 and all(r['passed']for r in q['normal'])
 assert read(D/'qualified_checks/distinctions/RESULT.json')['failed']==0
 assert read(D/'qualified_confirmation/confirmation/PATTERN_CHECK.json')['passed']
 assert read(D/'safety/RESULT.json')['passed']and read(D/'safety/PRESENTATION.json')['passed']
 parent=read(P/'web/ADMISSION.json');pins=dict(parent['pins'])
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
 for p in[P/'web/ADMISSION.json']+[D/n for n in['audit.py','form_candidate.py','form_v2.py','form_v4.py','form_v5.py','train_relation.py','form_presentation.py','runtime/worker.py','form_web_adapter.py','serve.py']]+list((D/'relation').glob('*.pt')):
  pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 write(D/'web/ADMISSION.json',dict(admitted=True,scope='Scientific gates passed; local runtime validation before launch.',pins=pins,new_relation_role='Inspection proposal only; original completion/repetition untouched',private_training=False))
 print('FINAL_VIEWS',checks,'PINS',len(pins),flush=True)
if __name__=='__main__':main()
