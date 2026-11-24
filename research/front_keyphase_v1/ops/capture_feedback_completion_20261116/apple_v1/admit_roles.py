from apple_io import *
import shutil,admit_web
assert all(r['full']for r in read(U/'required_role_checks/NORMAL.json')['rows'])
assert read(U/'required_role_checks/CONTROLS.json')['passed']==70
assert not(U/'web/ADMISSION_v1.json').exists();shutil.copyfile(U/'web/ADMISSION.json',U/'web/ADMISSION_v1.json')
admit_web.main();q=read(U/'web/ADMISSION.json');q['supersedes']='ADMISSION_v1.json';q['evidence']['role_repair']='15 normal=8FIT+7development/confirmation; 70 controlled checks preserved. Pair-head replacement regressed FIT9 and was not admitted. Only one-hand mouth-region dependency repaired.'
q['evidence']['candidate_held_people']=[];q['evidence']['previously_held_now_diagnostic_people']=[16,17];q['limitations'].append('Crop repair developed after observing person16 capture; persons16/17 are no longer fresh evaluation for this repair.');write(U/'web/ADMISSION.json',q)
