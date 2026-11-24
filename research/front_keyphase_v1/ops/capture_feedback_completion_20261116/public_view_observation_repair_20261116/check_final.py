from view_common import *
from final_candidate import PlaneViewLesson
from check_views import main as view_check
import importlib.util
spec=importlib.util.spec_from_file_location('current_view_regression',J/'check_regression.py')
reg=importlib.util.module_from_spec(spec);spec.loader.exec_module(reg)

def main():
 if not(J/'final_checks/RESULT.json').exists():view_check(PlaneViewLesson,'final_checks')
 reg.main(PlaneViewLesson,'final_regression')
 torch.set_num_threads(2);net=PlaneViewLesson();rows=[]
 for r in read(J/'confirmation/INPUTS.json')['rows']:
  previous=read(J/'confirmation'/(r['id']+'.json'));q=net.analyze(dict(np.load(r['raw'])));aa={c['id']:c['status']for c in previous['after']['criteria']};bb={c['id']:c['status']for c in q['criteria']};rows.append(dict(id=r['id'],same_criteria=aa==bb,same_events=previous['after']['events']==q['events']));write(J/'final_checks'/(r['id']+'.json'),dict(before=previous['before'],after=q))
 write(J/'final_checks/CONFIRMATION.json',dict(rows=rows,passed=all(r['same_criteria']and r['same_events']for r in rows),note='Confirmation clips now known; this verifies removing unhelpful new palm/relation models, not fresh performance.'))
 print('FINAL_CONFIRMATION',rows,flush=True)
if __name__=='__main__':main()
