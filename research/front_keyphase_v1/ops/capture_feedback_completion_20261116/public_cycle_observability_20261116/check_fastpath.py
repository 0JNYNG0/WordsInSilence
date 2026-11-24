from final_cycle import *
def main():
 torch.set_num_threads(2);net=LearnedCycleLesson();rows=[]
 for r in all_sources()+read(M/'confirmation/INPUTS.json')['rows']:
  p=M/'learned_checks'/(r['id']+'.json')
  if not p.exists():p=M/'final_extra'/(r['id']+'.json')
  a=read(p)['after'];b=net.analyze(dict(np.load(r['raw'])));clean=lambda q:[{k:v for k,v in c.items()if k!='scale_basis'}for c in q['criteria']];passed=clean(a)==clean(b)and a['events']==b['events'];rows.append(dict(id=r['id'],passed=passed,before_seconds=a['seconds'],after_seconds=b['seconds']));
  if not passed:print('FASTPATH_CHANGED',r['id'],flush=True)
 write(M/'FASTPATH_CHECK.json',dict(rows=rows,passed=all(r['passed']for r in rows),scope='Final candidate output-equivalence after conditional fallback execution; no threshold or decision change.'));print('FASTPATH',all(r['passed']for r in rows),flush=True)
if __name__=='__main__':main()
