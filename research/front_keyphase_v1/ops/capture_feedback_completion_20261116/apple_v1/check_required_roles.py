from required_role_candidate import *
import check_controls
def main():
 torch.set_num_threads(2);net=RequiredRoleCandidate();dest=U/'required_role_checks';dest.mkdir(exist_ok=True);rows=[]
 for r in read(U/'INPUTS.json')['rows']+read(U/'holdout/EXTRACTED.json')['rows']:
  q=net.analyze(dict(np.load(r.get('raw',r.get('raw_path')))));cs={c['id']:c['status']for c in q['criteria']};rows.append(dict(person=r['person'],criteria=cs,full=all(v=='supported'for v in cs.values())));write(dest/f'p{r["person"]}.json',q)
 assert all(r['full']for r in rows);write(dest/'NORMAL.json',dict(rows=rows));print('NORMAL',len(rows))
 check_controls.main(RequiredRoleCandidate,dest,True)
if __name__=='__main__':main()
