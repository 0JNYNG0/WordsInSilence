from feedback_lesson import *
import importlib.util
def main():
 assert not(U/'FEEDBACK_COMPARISON_v3.json').exists();torch.set_num_threads(2)
 s=importlib.util.spec_from_file_location('hello_public_cases',LIVE/'usability_repair_20261116/public_compare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 raw=dict(np.load(LIVE/'source/reference.npz'));net=FeedbackLesson();rows=[];dest=U/'feedback_v3';dest.mkdir(exist_ok=True)
 for name,z,expected,_ in m.cases(raw):
  q=net.analyze(z);cs={c['id']:c['status']for c in q['criteria']};full=all(v=='supported'for v in cs.values());rows.append(dict(case=name,expected=all(expected),completed=q['completed'],full=full,criteria=cs,seconds=q['seconds'],check=full==all(expected)));write(dest/(name+'.json'),q)
 write(U/'FEEDBACK_COMPARISON_v3.json',dict(rows=rows,passed=sum(r['check']for r in rows),total=len(rows),admitted=False));print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()


