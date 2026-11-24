from arrival_candidate import *
def main():
 q=read(K/'preserved_checks/RESULT.json');assert q['new_warning']==q['lost']==q['control_failed']==0
 assert read(K/'preserved_checks/BOUNDARY_PRESERVATION.json')['passed']and read(K/'confirmation/RESULT.json')['passed']
 q=read(K/'regression/RESULT.json');assert q['shape_new_wrong']==q['shape_lost_correct']==q['distinction_failed']==q['temporal_failed']==0 and not q['known_error_loss']
 assert read(K/'regression/confirmation/PATTERN_CHECK.json')['passed']
 pins=dict(read(J/'web/ADMISSION.json')['pins']);assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())
 for p in[J/'web/ADMISSION.json']+[K/n for n in['arrival_candidate.py','preserved_arrival.py','arrival_web_adapter.py','runtime/worker.py','serve.py']]+list((K/'runs').glob('*.pt'))+[O.parents[2]/'arrival_demo_launcher.py']:
  pins[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
 write(K/'web/ADMISSION.json',dict(admitted=True,pins=pins,status='Public gates passed; runtime/capture verification required before activation',scope='Preserve all deployed cycles; add independently observed and source-trained projected near endpoints only.',private_data=False,physical_contact=None));print('ADMITTED_FOR_RUNTIME',len(pins),flush=True)
if __name__=='__main__':main()
