from diagnose import *
sys.path.insert(0,str(U))
from diagnostic_ledger import DiagnosticLedger
from coach import Coach
from observation_quality import classification_reasons

def main():
 checks=[]
 def ck(name,q):checks.append(dict(name=name,passed=bool(q)));assert q,name
 l=DiagnosticLedger();l.begin('public_synthetic',1)
 for i,cause in enumerate(['model_disagreement','model_confidence_insufficient','finger_chain_degenerate']):
  category='missing_keypoints'if i==2 else'shape_uncertain'
  ob=dict(assessed_timestamp_us=i*100000,stage_index=0,diagnostic=dict(category=category,code=cause),shape_diagnostic={'R':dict(observation_reason=cause,finger_reasons=[cause]*4,raw_pixels='SECRET_PIXELS',raw_points=[9999999],unavailable_fingers=[])})
  l.observe(ob,{})
 snap=l.snapshot();text=json.dumps(snap);ck('diagnostic contains no private extra fields',all(s not in text for s in ['SECRET_PIXELS','9999999','raw_pixels','raw_points']))
 keys=snap['session']['subcause_counts'];ck('causes distinct',all('stage0.shape.R.index.'+c in keys for c in ['model_disagreement','model_confidence_insufficient','finger_chain_degenerate']))
 p=np.array([[[0,1,0,-1]],[[1,1,0,-1]],[[0,1,0,-1]]]);states=np.array([[-1,1,0,-1]]);usable=np.ones((1,4),bool)
 q=classification_reasons({'a':p,'b':p},states,usable,np.full((1,4),'geometry_present',dtype='<U40'))
 ck('model confidence vs disagreement',q.tolist()==[['model_disagreement','model_decided','model_decided','model_confidence_insufficient']])
 # Existing service ownership, stopped-session and bounded-retention regressions.
 path=U/'usability_repair_20261116/check_ledger.py';source=path.read_text(encoding='utf-8')
 source=source.replace("(V/'LEDGER_CHECK.json').write_text", "(Path(r'"+str(V)+"')/'LEDGER_REGRESSION.json').write_text")
 exec(compile(source,str(path),'exec'),{'__file__':str(path),'__name__':'public_regression'})
 ck('prior ledger contracts',json.loads((V/'LEDGER_REGRESSION.json').read_text(encoding='utf-8'))['passed'])
 write(V/'DIAGNOSTIC_CHECKS.json',dict(passed=True,checks=checks,physical_camera=False,private_data=False));print('diagnostics',len(checks),'PASS')
if __name__=='__main__':main()
