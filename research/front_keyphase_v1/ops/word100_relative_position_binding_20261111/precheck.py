from relative_core import *
from evaluate_relative import bind,assign_ledger

def main():
 checks=[]
 def check(n,v):
  assert v,n
  checks.append(n)
 cfg=read(U/'FROZEN.json')
 for p,h in cfg['pins'].items():check('frozen '+p,sha(p)==h)
 for p,h in read(U/'PROTECTIONS.json').items():check('preserved '+p,sha(p)==h)
 # Real cached FIT example: scale/translation in screen coordinates must cancel.
 src=read(P/'PUBLIC_CACHE.json')['rows'][0];z=np.load(src['path']);x=z['x'];ts=z['ts'];e=dict(a=50,b=65,axis=1,moving='R')
 p=position_features(x,ts,e);xx=x.copy();pts=xx[:,:102].reshape(-1,51,2);pts[:]=pts*1.7+np.array([.2,-.3]);p2=position_features(xx,ts,e)
 check('screen scale and translation invariance',np.allclose(p,p2,atol=2e-5))
 xx=x.copy();xx[:,186:237]=0;p2=position_features(xx,ts,e);check('missing not filled with desired position',np.all(p2==0))
 # Same candidate family remains the same irrespective of requested direction.
 ledger=[dict(a=1,b=5,axis=2,moving='R',event=0,family=4,role_coverage=1.,valid=1.)];prob=np.zeros((8,11));prob[:,4]=1;aa=assign_ledger(ledger,prob)
 check('candidate kept',len(aa)==1 and aa[0]['assignment_eligible'])
 saved=dict(ledger=ledger,cards=[dict(id='test',display_status='similar',required=True)]);cp=[dict(physical_event_complete=False)];ds={(2,'R',1,5):dict(direction=-1,direction_readable=True)};s=dict(id='test',axis=2,movement_subjects=['R'],families=[4],sign=1,instruction='팔을 따라 이동',required=True)
 a=bind(saved,cp,ds,[s],prob);s['sign']=-1;b=bind(saved,cp,ds,[s],prob)
 check('assignment ignores desired direction',a[1][0]['events']==b[1][0]['events'])
 check('partial contrary attempt preserved',a[1][0]['events'][0]['direction']==-1 and not a[1][0]['events'][0]['physical_complete'])
 check('partial contradiction warns',a[0][0]['display_status']=='difference')
 prob[:]=0;prob[:,9]=1;cc=assign_ledger(ledger,prob);check('return retained not deleted',len(cc)==1 and cc[0]['family']==9 and not cc[0]['assignment_eligible'])
 write(U/'PRECHECK.json',dict(status='PASS',checks=len(checks),names=checks,scope='Integrity, input invariance, missing mask and direction-independent binding checks. Not accuracy evidence.'));event('precheck_pass',checks=len(checks))

if __name__=='__main__':main()
