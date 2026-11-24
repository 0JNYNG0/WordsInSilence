from common import *
from collections import defaultdict

def main():
    dest=HERE/'axis2_sign_audit';dest.mkdir(exist_ok=True);assert not(dest/'RESULTS.json').exists()
    m=read(HERE/'INPUTS_COMPLETE.json');assert sha(m['path'])==m['sha256'];z=np.load(m['path']);out=[]
    for i,r in enumerate(m['rows']):
        if r['split']!='train'or r['axis']!=2:continue
        d=float(z['projected_geometry'][i,40]);assert int(z['y'][i])==r['target']
        out.append(dict(index=i,clip_id=r['clip_id'],person=r['person'],view=r['view'],reverse=r['reverse'],origin=r['origin'],target=r['target'],span=[r['a'],r['b']],signed_body_projection_delta=d))
    groups=defaultdict(list)
    for r in out:groups[r['origin'],r['target']].append(r['signed_body_projection_delta'])
    summary=[dict(origin=o,target=t,rows=len(v),negative=sum(x<0 for x in v),zero=sum(x==0 for x in v),positive=sum(x>0 for x in v),minimum=min(v),median=float(np.median(v)),maximum=max(v))for(o,t),v in sorted(groups.items())]
    write(dest/'RESULTS.json',dict(at=stamp(),scope='FIT-only measurement-label consistency audit; sign counts at mathematical zero, not a tuned movement threshold or model. No probe scoring.',rows=out,summary=summary,pins={str(p):sha(p)for p in[Path(__file__),HERE/'INPUTS_COMPLETE.json',Path(m['path'])]},web_changed=False))
    for r in summary:print(r,flush=True)

if __name__=='__main__':main()
