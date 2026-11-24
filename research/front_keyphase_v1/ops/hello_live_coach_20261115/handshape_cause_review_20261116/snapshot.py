from diagnose import *
import shutil
def main():
 b=V/'before';b.mkdir(exist_ok=True)
 files=[O/'word100_finger_repair_20261115/finger_runtime.py',O/'basic_demo_feedback_20261115/shape_consensus.py',U/'coach.py',U/'diagnostic_ledger.py']
 for p in files:
  assert not(b/p.name).exists();shutil.copyfile(p,b/p.name)
 weights=[]
 for folder in [O/'word100_finger_repair_20261115',O/'basic_demo_feedback_20261115/local_shape',O/'word100_clause_finger_diagnostics_20261114/finger_models_v2',O/'word100_broad_factorized_learning_20261110/runs/coupled_42']:
  for p in folder.rglob('*.pt'):weights.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(V/'PROTECTED_WEIGHTS.json',weights)
 torch.set_num_threads(2);net=ShapeConsensus();lab=json.loads((O/'word100_finger_error_split_20261115/LABELS.json').read_text(encoding='utf-8'))['rows'];paths=sorted({r['raw_path']for r in lab})
 cache=V/'baseline';cache.mkdir(exist_ok=True);manifest=[]
 for path in paths:
  z=load(path);x=channel(z);arr={}
  for h in ['L','R']:
   q=net.sequence(x,h);arr.update({h+'_'+k:v for k,v in q.items()})
  out=cache/(Path(path).stem+'.npz');np.savez_compressed(out,**arr);manifest.append(dict(raw=path,baseline=str(out),frames=len(x)))
 write(V/'BASELINE_MANIFEST.json',manifest);print('snapshots',len(manifest),'frames',sum(r['frames']for r in manifest),'weights',len(weights))
if __name__=='__main__':main()
