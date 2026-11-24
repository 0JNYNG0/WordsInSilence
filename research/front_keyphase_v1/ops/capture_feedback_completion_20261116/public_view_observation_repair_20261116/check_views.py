from view_common import *
from view_candidate import ViewLesson

def brief(q):return dict(events=len(q['events']),criteria={c['id']:c['status']for c in q['criteria']})
def main(factory=ViewLesson,output='view_checks'):
 torch.set_num_threads(2);net=factory();out=J/output;out.mkdir(exist_ok=True);rows=[]
 for r in sources():
  a=baseline(r['id']);b=net.analyze(dict(np.load(r['raw'])));aa,bb=brief(a),brief(b);lost=[k for k,v in aa['criteria'].items()if v=='supported'and bb['criteria'][k]!='supported'];new=[k for k,v in bb['criteria'].items()if v=='difference'and aa['criteria'][k]!='difference'];removed=[k for k,v in aa['criteria'].items()if v=='difference'and bb['criteria'][k]!='difference']
  row=dict(id=r['id'],before=aa,after=bb,lost=lost,new_warning=new,removed_normal_warning=removed);rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b))
  if aa!=bb:print(row,flush=True)
 write(out/'RESULT.json',dict(rows=rows,lost_supported=sum(bool(r['lost'])for r in rows),new_warning=sum(bool(r['new_warning'])for r in rows),removed_warning_conditions=sum(bool(r['removed_normal_warning'])for r in rows),private_data=False));print('CHECK',len(rows),'lost',sum(bool(r['lost'])for r in rows),'newwarning',sum(bool(r['new_warning'])for r in rows),flush=True)
if __name__=='__main__':main()
