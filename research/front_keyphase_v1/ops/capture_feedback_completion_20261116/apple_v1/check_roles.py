from role_candidate import *
import check_controls
def main():
 torch.set_num_threads(2);net=RoleCandidate();rows=[];dest=U/'role_checks';dest.mkdir(exist_ok=True)
 for r in read(U/'INPUTS.json')['rows']+read(U/'holdout/EXTRACTED.json')['rows']:
  q=net.analyze(dict(np.load(r.get('raw',r.get('raw_path')))));cs={c['id']:c['status']for c in q['criteria']};rows.append(dict(person=r['person'],criteria=cs,full=all(v=='supported'for v in cs.values())));write(dest/f'p{r["person"]}.json',q)
 write(dest/'NORMAL.json',dict(rows=rows));print(json.dumps(rows,ensure_ascii=False))
 check_controls.main(RoleCandidate,dest,True)
if __name__=='__main__':main()
