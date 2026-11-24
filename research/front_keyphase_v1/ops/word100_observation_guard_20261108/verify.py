"""Saved-output arithmetic and preservation audit; no model re-inference."""
from prepare import *
def main():
    checks=[]
    def ck(name,v):checks.append(dict(name=name,passed=bool(v)))
    f=read(J/'FROZEN.json');summary={}
    for view in ['full','crop']:
        e=read(U/f'EVAL_{view}.json');rs=e['rows'];s=e['summary']
        ck(view+' denominator',len(rs)==s['n'] and len({r['clip_id'] for r in rs})==len(rs))
        for key,field in [('before','baseline'),('after','guarded')]:ck(view+' '+key,sum(r[field]['rank']==1 for r in rs)==s[key])
        ck(view+' regressions',s['regressions']==[r['clip_id'] for r in rs if r['baseline']['rank']==1 and r['guarded']['rank']!=1])
        for r in rs:
            truth=f['classes'].index(r['word'])
            for key in ['baseline','guarded']:
                if 'logits' in r[key]:
                    a=np.asarray(r[key]['logits']);ck(view+' '+r['clip_id']+' '+key,int((a>a[truth]).sum()+1)==r[key]['rank'])
        summary[view]=s
    frozen=read(U/'GUARD_FROZEN.json');plan=read(U/'INPUT_PLAN.json')
    for p,h in frozen['pins'].items():ck('frozen '+p,sha(p)==h)
    pins={**plan['protected'],**f['pins']}
    for p,h in pins.items():ck('protected '+p,sha(p)==h)
    people={k:{r['person'] for r in plan['rows'] if (r['role']=='calibration')==k} for k in [True,False]}
    ck('observation calibration/regression recorded IDs separated',not people[True]&people[False])
    for r in plan['rows']:
        z=np.load(r['cache'] or U/'crops'/(r['clip_id']+'.npz'));old=np.load(r['raw'])
        ck(r['clip_id']+' exact frame/time preservation',np.array_equal(z['frame_id'],old['frame_id']) and np.array_equal(z['timestamp_us'],old['timestamp_us']))
    crop=read(U/'EVAL_crop.json')['rows'];before=sum(s['false_inframe_before'] for r in crop for s in r['geometric_proxy']);after=sum(s['false_inframe_after'] for r in crop for s in r['geometric_proxy'])
    engineering=read(U/'ENGINEERING_CHECKS.json');integration=read(U/'INTEGRATION_CHECKS.json')
    ck('engineering contracts',engineering['status']=='PASS');ck('integration contracts',integration['status']=='PASS')
    ck('no full correct-to-wrong regression',not summary['full']['regressions'])
    ck('net crop recognition improvement',summary['crop']['after']>summary['crop']['before'] and not summary['crop']['regressions'])
    ck('less outside-hand contamination proxy',after<before)
    out=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',n=len(checks),checks=checks,protected_count=len(pins),summary=summary,
        proxy_before=before,proxy_after=after,verifier='Same Codex separate saved-logit/arithmetic/hash verification, not independent expert truth',new_inference=False)
    write(U/'VERIFICATION.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['checks','summary']},ensure_ascii=False));assert out['status']=='PASS'
if __name__=='__main__':main()
