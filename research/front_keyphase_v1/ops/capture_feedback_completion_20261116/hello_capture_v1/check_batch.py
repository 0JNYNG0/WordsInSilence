from batch_observer import *
import importlib.util,hashlib
def main():
 assert not(U/'BATCH_COMPARISON.json').exists();torch.set_num_threads(2)
 pins=[LIVE/'coach.py',LIVE/'action_runtime.py',LIVE/'action_integration/ADMITTED.json',LIVE/'seated_diagnosis_20261116/VERTICAL_ADMITTED.json']
 write(U/'FROZEN.json',dict(goal='Reuse recorded RTMW in one batch instead of repeated live encoders; existing gesture requirements unchanged.',pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in pins},gates='Known public normal, partial, reverse, shapes, gaps; preserve expected states. Failures stay isolated.',training=False,private_data=False,reference_people=1,independent_people=0))
 s=importlib.util.spec_from_file_location('hello_public_cases',LIVE/'usability_repair_20261116/public_compare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 raw=dict(np.load(LIVE/'source/reference.npz'));net=BatchLesson();rows=[]
 for name,z,expected,_ in m.cases(raw):
  q=net.analyze(z);row=dict(case=name,expected=expected,actual=q['completed'],passed=q['completed']==expected,seconds=q['seconds'],encoder_calls=q['shared_encoder_calls']);rows.append(row)
  if name in['native_reference','reverse_sweep_frames','closed_sweep_hand','open_hands_in_fists_motion','one_downstroke_then_hold']:write(U/(name+'.json'),q)
 write(U/'BATCH_COMPARISON.json',dict(rows=rows,passed=sum(r['passed']for r in rows),total=len(rows),training=False,candidate_admitted=False));print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
