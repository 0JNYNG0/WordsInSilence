import json,hashlib,sys,copy,time
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;B=U.parents[1];sys.path.insert(0,str(B/'src'))
from wis_lab.service import word100_role_dtw as R,word100_multiref as M,word100_analysis as A
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text('utf-8'))
def write(name,d): (U/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def erase(s,side):
    s=copy.deepcopy(s)
    for i,g in enumerate(A.F.GROUP_ORDER):
        if g.endswith('_'+side) or g=='bimanual':s['present'][:,i]=False;s['features'][:,A.OFFSETS[g]]=np.nan
    return s
def summarized(pairs,required,baseline):return R.relative(R.aggregate(pairs,required),baseline)
def run():
    assert not (U/'ROLE_BANK.json').exists(),'Preserve frozen bank, do not repeat'
    old,oldpin=M.bank(); evidence=read(U/'PUBLIC_EVIDENCE.json')
    roles={}
    for code,item in old['words'].items():
        single=code in ('WORD0949','WORD2177','WORD2551')
        public=[x for x in evidence if x['word_id']==code]
        if single: assert [x['signer_id'] for x in public]==list(range(1,10))
        roles[code]={'required_sides':['R'] if single else ['L','R'],
          'basis':'Public TRAIN01-09 RGB: right hand performs, left remains lowered without supporting contact; observational annotation by Codex, not linguistic expert certification' if single else 'Conservative both-hands requirement; stationary support is retained; no absence-based exclusion',
          'source_sha256':{x['clip_id']:x['source_sha256'] for x in public}}
    write('HAND_ROLES.json',{'schema':'wis_reference_hand_roles_v1','annotator':'Codex visual review','user_data_used':False,'unreviewed_policy':'both required','words':roles})
    bank={'schema':R.SCHEMA,'legacy_bank_sha256':oldpin,'roles_path':str(U/'HAND_ROLES.json'),'roles_sha256':sha(U/'HAND_ROLES.json'),
          'code_sha256':{str(Path(x)):sha(x) for x in [R.__file__,A.__file__,M.__file__]},'user_data_used':False,'words':{}}
    tests=[];train_pairs=[]
    for code,item in old['words'].items():
        required=roles[code]['required_sides'];target=copy.deepcopy(item)
        target.update(required_sides=required,role_basis=roles[code]['basis'])
        refs=target['references']; samples={r['clip_id']:M.sample(r['sample_path'],r['sample_sha256']) for r in refs}
        if len(required)==1:
            loo=[]
            for q in refs:
                pairs=[]
                for ref in refs:
                    if q['clip_id']==ref['clip_id']:continue
                    pair={'query_clip':q['clip_id'],'reference_clip':ref['clip_id'],'result':R.compare_samples(samples[q['clip_id']],samples[ref['clip_id']],required)}
                    pairs.append(pair);train_pairs.append(pair)
                loo.append(dict(clip_id=q['clip_id'],signer_id=q['signer_id'],metrics=R.aggregate(pairs,required)))
            baseline={}
            for key in M.AXES:
                values=[next(m for m in row['metrics'] if m['key']==key) for row in loo]
                baseline[key]=M.distribution([m['value'] for m in values])
                baseline[key]['sides']={s:M.distribution([next(d for d in m['sides'] if d['side']==s)['value'] for m in values]) for s in ('L','R')}
                assert baseline[key]['n']==9 and baseline[key]['median']>M.EPS
            target.update(leave_one_out=loo,baseline=baseline)
            q=samples[refs[0]['clip_id']];ref=samples[refs[1]['clip_id']]
            original=R.compare_samples(q,ref,required)
            invisible=R.compare_samples(erase(q,'L'),ref,required)
            assert original==invisible
            missing=R.compare_samples(erase(q,'R'),ref,required);assert missing['status']=='INSUFFICIENT'
            selfpair=R.compare_samples(q,q,required)
            assert all(m['value'] is not None and abs(m['value'])<1e-6 for m in selfpair['metrics'])
            tests.extend([{'word':code,'test':name,'status':'PASS'} for name in ['unused_hand_invariance','required_hand_missing_rejected','self_difference_zero']])
        else:
            q=samples[refs[0]['clip_id']];ref=samples[refs[1]['clip_id']]
            assert R.compare_samples(q,ref,required)==A.compare_samples(q,ref)
            tests.append({'word':code,'test':'both_hands_exact_legacy_pair_parity','status':'PASS'})
        bank['words'][code]=target
        print('bank',code,flush=True)
    # All36 existing public DEV clips are evaluated in the new deterministic layer.
    # No conditions are selected or adjusted using these outcomes.
    inputs=read(B/'ops/word100_multiref_20261014/INPUTS.json'); dev=[]; probes=[]
    for info in inputs:
        if info['role']!='development_only':continue
        code=info['clip_id'].split('_')[2];item=bank['words'][code];required=item['required_sides'];q=M.sample(info['sample_path'],info['sample_sha256'])
        refs=item['references']
        def compare(sample):
            pairs=[{'result':R.compare_samples(sample,M.sample(x['sample_path'],x['sample_sha256']),required)} for x in refs]
            return summarized(pairs,required,item['baseline'])
        m=compare(q);dev.append({'clip_id':info['clip_id'],'word_id':code,'required_sides':required,'metrics':m})
        if len(required)==1:
            m2=compare(erase(q,'L'));assert m==m2
            m3=compare(erase(q,'R'));assert all(x['value'] is None for x in m3)
            probes.append({'clip':info['clip_id'],'optional_hand_missing_invariant':True,'required_hand_missing_held':True})
        else:
            for side in ('L','R'):
                hidden=compare(erase(q,side));assert all(x['value'] is None for x in hidden)
            probes.append({'clip':info['clip_id'],'each_required_hand_missing_held':True})
    assert len(dev)==36
    # Entire old normal DEV numeric metrics must match for unchanged two-hand roles.
    olddev={x['query_clip']:x for x in read(B/'ops/word100_multiref_20261014/DEVELOPMENT.json')}
    for x in dev:
        if len(x['required_sides'])==2:
            for a,b in zip(x['metrics'],olddev[x['clip_id']]['metrics']):
                assert a['value']==b['value'] and a['ratio']==b['ratio'] and a['baseline']==b['baseline']
    write('TRAIN_PAIRS.json',train_pairs);write('DEVELOPMENT.json',dev)
    write('PUBLIC_VALIDATION.json',{'status':'PASS','synthetic_and_fixed_train_checks':tests,'public_DEV_probes':probes,'public_DEV_count':36,'both_hands_unchanged_DEV_count':27,'single_hand_invariance_DEV_count':9,'thresholds_changed':False,'new_training':0})
    write('ROLE_BANK.json',bank)
    cfg={'schema':R.SCHEMA,'bank_path':str(U/'ROLE_BANK.json'),'bank_sha256':sha(U/'ROLE_BANK.json')}
    (B/'config/word100_role_dtw.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n','utf-8')
    write('FROZEN_READY.json',{'status':'PUBLIC_VALIDATION_PASS_BEFORE_USER_REPLAY','bank_sha256':cfg['bank_sha256'],'roles_sha256':bank['roles_sha256'],'code_sha256':bank['code_sha256'],'new_TCN_training':0,'user_in_rules_or_scales':False})
    print(json.dumps({'status':'PASS','dev':36,'train_pairs_new':len(train_pairs),'bank_pin':cfg['bank_sha256']},ensure_ascii=False))
if __name__=='__main__':run()
