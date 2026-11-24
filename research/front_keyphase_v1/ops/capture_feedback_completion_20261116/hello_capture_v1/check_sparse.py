from sparse_observer import *
import importlib.util
def main():
 assert not(U/'SPARSE_COMPARISON.json').exists();torch.set_num_threads(2)
 s=importlib.util.spec_from_file_location('hello_public_cases',LIVE/'usability_repair_20261116/public_compare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 raw=dict(np.load(LIVE/'source/reference.npz'));net=SparseLesson();rows=[];dest=U/'sparse';dest.mkdir(exist_ok=True)
 for name,z,expected,_ in m.cases(raw):
  q=net.analyze(z);rows.append(dict(case=name,expected=expected,actual=q['completed'],passed=q['completed']==expected,seconds=q['seconds'],encoder_calls=q['shared_encoder_calls']));write(dest/(name+'.json'),q)
 write(U/'SPARSE_COMPARISON.json',dict(rows=rows,passed=sum(r['passed']for r in rows),total=len(rows),training=False,candidate_admitted=False));print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
