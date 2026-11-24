import numpy as np
from candidate import geometry,decide,HandAssigner
from run import write,U

def main():
    checks=[]
    def ok(name,value):
        assert value,name
        checks.append(name)
    # State-machine controls independent of geometry implementation.
    g={'arms_visible':[True,True],'preferred':[0,0],'coverage':np.array([[0,1],[1,0.]])}
    ok('weak duplicate held',decide([0,1],['confirmed','provisional'],g)==([0,None],[1]))
    ok('normal distinct arms retained',decide([0,1],['confirmed','provisional'],{**g,'preferred':[0,1]})==([0,1],[]))
    ok('confirmed crossing retained',decide([0,1],['confirmed','confirmed'],g)==([0,1],[]))
    ok('elbow-only cannot reject',decide([0,1],['confirmed','provisional'],{**g,'arms_visible':[True,False]})==([0,1],[]))
    ok('two weak candidates cannot reject',decide([0,1],['provisional','provisional'],g)==([0,1],[]))
    ok('coverage below threshold retained',decide([0,1],['confirmed','provisional'],{**g,'coverage':np.array([[0,0],[.799,0]])})==([0,1],[]))
    ok('no force-fill unassigned',decide([0,None],['confirmed','unassigned'],g)==([0,None],[]))
    p=np.zeros((33,2));pu=np.zeros(33,bool)
    p[[11,13,15]]=[[700,200],[700,350],[700,500]];p[[12,14,16]]=[[300,200],[300,350],[300,500]];pu[[11,12,13,14,15,16]]=True
    hand=np.array([[700,500]]+[[690+(i%4)*7,505+(i//4)*7] for i in range(20)],float)
    warped=hand.copy();warped[0]=[700,450];h=np.array([hand,warped]);gg=geometry(h,p,pu,1000,1000)
    ok('warped wrist region both same arm',gg['preferred']==[0,0] and gg['coverage'][1,0]>=.8)
    ok('geometry positive guard',decide([0,1],['confirmed','provisional'],gg)==([0,None],[1]))
    rev=geometry(h[::-1],p,pu,1000,1000)
    ok('candidate order invariant',np.allclose(gg['arm_cost'],rev['arm_cost'][::-1]) and decide([1,0],['provisional','confirmed'],rev)==([None,0],[0]))
    trans=geometry(h*2+30,p*2+30,pu,2030,2030)
    ok('translation scale invariant',np.allclose(gg['arm_cost'],trans['arm_cost']) and np.array_equal(gg['coverage'],trans['coverage']))
    mirrorh=h.copy();mirrorh[:,:,0]=1000-mirrorh[:,:,0];mirrorp=p.copy();mirrorp[:,0]=1000-mirrorp[:,0]
    for a,b in [(11,12),(13,14),(15,16),(17,18),(19,20),(21,22)]:mirrorp[[a,b]]=mirrorp[[b,a]]
    mir=geometry(mirrorh,mirrorp,pu,1000,1000)
    ok('mirror anatomical swap',mir['preferred']==[1,1] and np.allclose(gg['arm_cost'],mir['arm_cost'][:,::-1]))
    bad=pu.copy();bad[[15,16]]=False;q=p.copy();q[[15,16]]=[[-1e9,1e9],[1e9,-1e9]]
    z=geometry(h,p,bad,1000,1000);zz=geometry(h,q,bad,1000,1000)
    ok('unusable wrist coordinates cannot leak',np.array_equal(z['arm_cost'],zz['arm_cost']) and z['preferred']==[-1,-1])
    outside=p.copy();outside[15]=[700,1001]
    ok('outside usable wrist not evidence',geometry(h,outside,pu,1000,1000)['arms_visible']==[False,True])
    outsideh=h.copy();outsideh[1,0,1]=1001
    ok('partial palm not hard corrected',geometry(outsideh,p,pu,1000,1000)['preferred'][1]==-1)
    empty=geometry(np.empty((0,21,2)),p,pu,1000,1000)
    ok('no hand remains empty',decide([],[],empty)==([],[]))
    raw=h.copy();pp=p.copy();vv=pu.copy();geometry(h,p,pu,1000,1000)
    ok('inputs immutable',np.array_equal(h,raw) and np.array_equal(p,pp) and np.array_equal(pu,vv))
    model=HandAssigner();out,_=model.assign(h,[0,1],[.99,.99],p,pu,100,1000,1000)
    restarted,_=model.assign(h,[0,1],[.99,.99],p,pu,100,1000,1000)
    ok('timestamp reset',out==restarted)
    write(U/'CONTRACTS.json',{'status':'PASS','checks':checks,'count':len(checks),'public_data_used':False})
    print('CONTRACTS',len(checks),'PASS')
if __name__=='__main__':main()
