from pathlib import Path
import sys,json,hashlib,time
N=Path(__file__).resolve().parent;G=N.parent;O=G.parent;V=G/'completion_v1';H=G/'public_shape_arrival_repair_20261116';M=G/'public_cycle_observability_20261116'
sys.path.insert(0,str(V))
from observed_gap import SparseVisible,np,torch
sys.path.insert(0,str(H))
from repair import RepairedVisible
from source_policy import require_front
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 torch.set_num_threads(2);loaded=[];original=torch.load
 def traced(path,*a,**kw):
  p=Path(path);loaded.append(str(p.resolve()));return original(path,*a,**kw)
 torch.load=traced
 for name,cls in [('front_sparse',SparseVisible),('front_linked',RepairedVisible)]:
  loaded.clear();net=cls();models=list(dict.fromkeys(loaded));rows=[];out=N/name;out.mkdir(exist_ok=True)
  for r in read(N/'INPUTS.json')['rows']:
   if r['word']!='WORD1290':continue
   require_front(r);a=read(r['baseline'])['after'];b=net.analyze(dict(np.load(r['raw'])));aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']};row=dict(id=r['id'],before=aa,after=bb,events_before=len(a['events']),events_after=len(b['events']),lost=[k for k,v in aa.items()if v=='supported'and bb[k]!='supported'],new_warning=[k for k,v in bb.items()if v=='difference'and aa[k]!='difference'],seconds=b['seconds']);rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b))
   if row['lost']or row['new_warning']:print(name,row,flush=True)
  write(out/'RESULT.json',dict(rows=rows,loaded_weights=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for p in models],full=sum(all(v=='supported'for v in r['after'].values())for r in rows),failed=sum(bool(r['lost']or r['new_warning'])for r in rows)));print(name,'FULL',read(out/'RESULT.json')['full'],'FAILED',read(out/'RESULT.json')['failed'],flush=True)
if __name__=='__main__':main()
