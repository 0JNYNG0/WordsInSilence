from audit import *
from copy import deepcopy

def fixture():
    n=5
    z={'raw_count':np.ones(n,dtype=int),'timestamp_us':np.arange(n)*33333}
    p={'identity_side':np.full((n,2),-1),'sides':np.full((n,2),-1),'point_mask':np.zeros((n,2,21),bool),'possible_links':np.zeros((n,2,2),bool)}
    p['sides'][:,0]=1;p['point_mask'][:,0]=True;p['possible_links'][1:,0,0]=True;p['identity_side'][2:4,0]=0
    pr={'sides':p['sides'].copy(),'current_pose':np.zeros((n,2),bool)};pr['sides'][2:4,0]=0;pr['current_pose'][2:4,0]=True
    s={'action':np.zeros((n,2),int)};g={'coverage':np.zeros((n,2,2))}
    return z,p,pr,s,g

def main():
    checks=[]
    def test(name,edit,expected):
        args=fixture();edit(*args);r,why,path=probe(*args,0,0)
        assert (r is not None)==expected,(name,why)
        if r:assert r['delay_us']<=100000 and r['side']==0
        checks.append({'test':name,'status':'PASS','reason':why})
    test('two direct anchors, separate delayed output',lambda *x:None,True)
    test('single anchor insufficient',lambda z,p,pr,s,g:pr['current_pose'].__setitem__((3,0),False),False)
    test('history not current anchor',lambda z,p,pr,s,g:pr['current_pose'].__setitem__(slice(None),False),False)
    test('blank frame breaks',lambda z,p,pr,s,g:z['raw_count'].__setitem__(1,0),False)
    test('nonpositive time breaks',lambda z,p,pr,s,g:z['timestamp_us'].__setitem__(1,0),False)
    test('100ms timeout',lambda z,p,pr,s,g:z['timestamp_us'].__setitem__(3,100001),False)
    test('missing physical link',lambda z,p,pr,s,g:p['possible_links'].__setitem__((1,0,0),False),False)
    def branch(z,p,pr,s,g):z['raw_count'][1]=2;p['possible_links'][1,1,0]=True
    test('branch rejected',branch,False)
    def merge(z,p,pr,s,g):z['raw_count'][0]=2;p['possible_links'][1,0,1]=True
    test('merge rejected',merge,False)
    def dup(z,p,pr,s,g):z['raw_count'][0]=2;g['coverage'][0,0,1]=.8
    test('overlap without anchor rejected',dup,False)
    test('previous withheld stays withheld',lambda z,p,pr,s,g:p['sides'].__setitem__((0,0),-1),False)
    test('previous confirmed stays unchanged',lambda z,p,pr,s,g:p['identity_side'].__setitem__((0,0),1),False)
    test('unobserved not invented',lambda z,p,pr,s,g:p['point_mask'].__setitem__((0,0),False),False)
    test('contrary evidence on path rejected',lambda z,p,pr,s,g:p['identity_side'].__setitem__((1,0),1),False)
    # A future anchor never becomes a present-time answer; same prefix cannot
    # see the two anchors and must not emit a retrospective confirmation.
    args=fixture();short=tuple({k:v[:2] for k,v in a.items()} for a in args);assert probe(*short,0,0)[0] is None
    checks.append({'test':'prefix cannot use unseen future frames','status':'PASS'})
    args=fixture();snap=[{k:a.copy() for k,a in d.items()} for d in args]
    for d in args:
        for a in d.values():a.flags.writeable=False
    probe(*args,0,0)
    assert all(np.array_equal(d[k],s[k]) for d,s in zip(args,snap) for k in d)
    checks.append({'test':'read-only inputs unchanged','status':'PASS'})
    write(U/'CONTRACTS.json',{'status':'PASS','count':len(checks),'checks':checks});print('PASS',len(checks))
if __name__=='__main__':main()
