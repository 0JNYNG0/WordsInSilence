import json
from pathlib import Path
import numpy as np
from candidate import duplicate_guard,ObservationLedger
def main():
    checks=[]
    def ck(n,t):assert t,n;checks.append(n)
    params=dict(sides=[1,0],states=[2,1],current_pose=[False,False],origin_us=[33333,100000],timestamp_us=100000,confirmed_distance=np.array([[np.nan,.01],[np.nan,.15]]),coverage=np.array([[0,1],[.85,0.]]),valid=[True,True])
    out,b,a=duplicate_guard(**params);ck('confirmed recent opposite anchor +covered duplicate withheld',out==[1,None] and b==[1] and a==[-1,0])
    for name,patch in [('old evidence',{'timestamp_us':200000}),('no anatomical anchor',{'states':[1,1]}),('partial hand',{'valid':[True,False]}),('insufficient region evidence',{'coverage':np.array([[0,1],[.7,0]])}),('own history protects crossing',{'confirmed_distance':np.array([[np.nan,.01],[.3,.15]])}),('unrelated history',{'confirmed_distance':np.array([[np.nan,.01],[np.nan,.5]])}),('confirmed own side protected',{'states':[2,2]})]:
        ck(name,duplicate_guard(**{**params,**patch})[0]==[1,0])
    out,_,_=duplicate_guard(**{**params,'sides':[0,1],'states':[1,2],'current_pose':[False,False],'origin_us':[100000,33333],'confirmed_distance':params['confirmed_distance'][::-1],'coverage':params['coverage'][::-1,::-1]});ck('raw-order guard equivariant',out==[None,1])
    h=np.arange(42,dtype=float).reshape(1,21,2)+20;p=np.zeros((33,2));pu=np.zeros(33,bool)
    m=ObservationLedger();o=m.step(h,[.9],p,pu,0,200,200,[1],[1],[1],[0],[],100)
    ck('low Pose wrist preserves all current Hand points',o['point_mask'].all() and np.array_equal(o['observed_xy'],h))
    ck('provisional anatomy not falsely certain',o['identity_side'].tolist()==[-1] and o['suggested_side'].tolist()==[1])
    ck('local shape without anatomical side',np.array_equal(o['local_xy'],h-h[:,:1]))
    o=m.step(h+1,[.9],p,pu,33333,200,200,[1],[1],[1],[0],[],100);ck('visible physical link is only a possibility',bool(o['possible_links'][0,0]) and o['identity_side'][0]<0)
    o=m.step(h+2,[.9],p,pu,200000,200,200,[1],[1],[1],[0],[],100);ck('long gap creates no physical link',not o['possible_links'].any())
    h2=h.copy();h2[0,0]=[-20,9999];o=ObservationLedger().step(h2,[.9],p,pu,0,200,200,[1],[1],[1],[0],[],100);ck('outside Hand wrist no false local shape',not o['local_mask'].any() and o['point_mask'].sum()==20)
    h3=h2.copy();h3[0,0]=[-99999,123456];q=ObservationLedger().step(h3,[.9],p,pu,0,200,200,[1],[1],[1],[0],[],100);ck('outside coordinates do not leak',np.array_equal(o['observed_xy'],q['observed_xy'],equal_nan=True))
    o=ObservationLedger().step(h,[.4],p,pu,0,200,200,[1],[1],[1],[0],[],100);ck('low Hand confidence marks unusable without fake points',not o['point_mask'].any() and np.isnan(o['observed_xy']).all())
    o=ObservationLedger().step(h,[.9],p,pu,0,200,200,[1],[None],[1],[0],[0],100);ck('duplicate conflict keeps observation',o['point_mask'].all() and o['identity_state'][0]==2 and o['identity_side'][0]<0)
    m=ObservationLedger();m.step(h,[.9],p,pu,0,200,200,[1],[1],[1],[0],[],100);o=m.step(np.empty((0,21,2)),[],p,pu,33333,200,200,[],[],[],[],[],100);ck('no observation never invented',o['point_mask'].shape==(0,21))
    o=m.step(h,[.9],p,pu,66666,200,200,[1],[1],[1],[0],[],100);ck('blank frame cannot bridge physical link',not o['possible_links'].any())
    o=ObservationLedger().step(h,[.9],p,pu,0,200,200,[0],[0],[2],[0],[],100);ck('established identity retained',o['identity_side'].tolist()==[0])
    Path(__file__).with_name('CONTRACTS.json').write_text(json.dumps({'status':'PASS','checks':checks,'count':len(checks),'public_data_used':False},indent=2),encoding='utf8');print('PASS',len(checks))
if __name__=='__main__':main()
