from view_common import *

def main():
 q=read(J/'final_checks/RESULT.json');assert not any(r['lost']or r['new_warning']for r in q['rows'])
 q=read(J/'final_regression/RESULT.json');assert q['shape_new_wrong']==q['shape_lost_correct']==q['distinction_failed']==q['temporal_failed']==0 and not q['known_error_loss']
 assert read(J/'final_regression/confirmation/PATTERN_CHECK.json')['passed']
 assert read(J/'confirmation/RESULT.json')['passed']and read(J/'final_checks/CONFIRMATION.json')['passed']
 pins=dict(read(E/'web/ADMISSION.json')['pins']);assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
 names=['view_common.py','view_candidate.py','final_candidate.py','view_web_adapter.py','runtime/worker.py','serve.py']
 for p in[E/'web/ADMISSION.json']+[J/n for n in names]+list((J/'plane').glob('*.pt'))+[O.parents[2]/'view_demo_launcher.py']:
  pins[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
 write(J/'web/ADMISSION.json',dict(admitted=True,status='Scientific checks passed; isolated runtime checks required before activation',pins=pins,scope='Only projected hand-plane head fine-tuned with public FIT side views. Previously deployed fingers, palm, endpoint/motion, C-T3F and TCN unchanged.',private_data=False,physical_contact=None))
 print('ADMITTED_FOR_RUNTIME_CHECK',len(pins),flush=True)
if __name__=='__main__':main()
