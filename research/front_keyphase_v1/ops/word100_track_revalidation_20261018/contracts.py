import json
from pathlib import Path
import numpy as np
from candidate import Revalidator
U=Path(__file__).resolve().parent
def main():
    checks=[]
    h=np.array([[[70+(i%4),50+(i//4)] for i in range(21)]],float)
    def geo(n=1,pref=None,arms=True):
        return {'valid':[True]*n,'preferred':[0]*n if pref is None else pref,'arms_visible':[arms,arms],'arm_cost':np.array([[.1,1.]]*n),'scale':100.}
    def ok(name,test):assert test,name;checks.append(name)
    m=Revalidator();g=geo()
    out,d=m.step(h,[.9],0,[1],['provisional'],g);ok('first independent conflict waits',out==[None] and d['confirmation_count']==[1])
    out,d=m.step(h+1,[.9],33333,[1],['provisional'],g);ok('second bounded observation relabels',out==[0] and d['recheck_reason']==['revalidated_current_arm'])
    ambiguous=geo(pref=[-1]);ambiguous['arm_cost'][:]=.1
    out,d=m.step(h+2,[.9],66666,[1],['provisional'],ambiguous);ok('short physical carry',out==[0])
    out,d=m.step(h+3,[.9],100000,[1],['provisional'],ambiguous);ok('history does not itself refresh arm evidence',out==[0] and d['corrected_arm_time']=={'0':.033333})
    out,d=m.step(h+4,[.9],150000,[1],['provisional'],ambiguous);ok('evidence expires despite ongoing detections',out==[1])
    for name,ts,points in [('gap',200000,h),('physical jump',33333,h+100)]:
        m=Revalidator();m.step(h,[.9],0,[1],['provisional'],g);out,d=m.step(points,[.9],ts,[1],['provisional'],g);ok(name+' cannot confirm',out==[None] and d['confirmation_count']==[1])
    m=Revalidator();out,_=m.step(h,[.9],0,[1],['confirmed'],g);ok('confirmed identity protected',out==[1])
    m=Revalidator();out,_=m.step(h,[.9],0,[0],['provisional'],g);ok('normal identity protected',out==[0])
    m=Revalidator();out,_=m.step(h,[.9],0,[1],['provisional'],geo(arms=False));ok('elbow only cannot change',out==[1])
    m=Revalidator();out,_=m.step(h,[.9],0,[None],['withheld_conflict'],g);ok('prior duplicate guard protected',out==[None])
    m=Revalidator();out,_=m.step(h,[.4],0,[1],['provisional'],g);ok('low confidence protected',out==[1])
    m=Revalidator();out,_=m.step(np.empty((0,21,2)),[],0,[],[],geo(0));ok('missing hand never invented',out==[])
    m=Revalidator();m.step(h,[.9],0,[1],['provisional'],g);m.step(np.empty((0,21,2)),[],10000,[],[],geo(0));out,d=m.step(h,[.9],33333,[1],['provisional'],g);ok('blank frame breaks vote chain',out==[None] and d['confirmation_count']==[1])
    m=Revalidator();m.step(h,[.9],33333,[1],['provisional'],g);out,d=m.step(h,[.9],33333,[1],['provisional'],g);ok('time reset breaks votes',out==[None] and d['confirmation_count']==[1])
    hh=np.concatenate([h,h+70]);gg=geo(2,[0,1]);gg['arm_cost']=np.array([[.1,1.],[1.,.1]])
    m=Revalidator();m.step(hh,[.9,.9],0,[1,0],['provisional']*2,gg);out,_=m.step(hh,[.9,.9],33333,[1,0],['provisional']*2,gg);ok('two wrong identities jointly swap without duplicate slot',out==[0,1])
    m=Revalidator();out,_=m.step(hh,[.9,.9],0,[1,0],['provisional','confirmed'],gg);ok('confirmed destination not displaced',out==[1,0])
    m=Revalidator();cross=geo(2,[1,0]);cross['arm_cost']=np.array([[1.,.1],[.1,1.]])
    out,_=m.step(hh,[.9,.9],0,[1,0],['provisional']*2,cross);ok('normal crossing preserved',out==[1,0])
    m=Revalidator();perm=Revalidator()
    for t in [0,33333,66666]:
        out,_=m.step(hh,[.9,.9],t,[1,0],['provisional']*2,gg)
        gp={**gg,'preferred':gg['preferred'][::-1],'arm_cost':gg['arm_cost'][::-1]}
        rr,_=perm.step(hh[::-1],[.9,.9],t,[0,1],['provisional']*2,gp)
        assert out==rr[::-1]
    ok('raw candidate order invariant',True)
    (U/'CONTRACTS.json').write_text(json.dumps({'status':'PASS','count':len(checks),'checks':checks,'public_data_used':False},indent=2),encoding='utf-8');print('PASS',len(checks))
if __name__=='__main__':main()
