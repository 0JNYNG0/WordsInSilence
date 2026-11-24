"""Stored blend/selection/timestamp arithmetic, separately implemented, no models."""
from pathlib import Path
from collections import Counter
import itertools,json,hashlib,datetime,time,traceback
import numpy as np
S=Path(__file__).resolve().parent;J=S.parent/'splice_rank_learning';MODE='equal_rank_blend';AX={2:0,3:1,4:2,5:4,7:3}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):return r['seed'],r['clip_id'],r['kind']

def main():
    dest=S/'VERIFICATION.json';assert not dest.exists();checks=0
    def ck(v,msg):
        nonlocal checks
        assert v,msg;checks+=1
    f=read(S/'FROZEN.json');new=read(S/'RESULTS.json');old=read(J/'evaluation/COMPARISON.json')
    for path,h in f['pins'].items():ck(sha(path)==h,('preservedpin',path))
    ck(new['alpha']==f['alpha']==.5 and not new['new_training']and not new['new_inference'],'onefixednoinferencecondition')
    om={key(r):r for r in old['rows']};nm={key(r):r for r in new['rows']};pm={key(r):r for r in new['predictions']}
    ck(set(om)==set(nm)==set(pm)and len(nm)==246,'same246')
    sources={};protected=0;ptsrows=[]
    for k,row in nm.items():
        prior=om[k];pr=pm[k];ck(sha(pr['path'])==pr['sha256'],'newarrayhash');z=dict(np.load(pr['path']));o=dict(np.load(pr['original_prediction']));j=dict(np.load(pr['J_prediction']))
        ck(all(row['conditions'][m]==prior['conditions'][m]for m in prior['conditions']),'oldconditionsimmutable')
        if pr['source']not in sources:sources[pr['source']]=dict(np.load(pr['source']))
        source=sources[pr['source']];ts=source['ts'];q=source['z'][:,292:392].reshape(-1,10,10)
        cards=row['conditions'][MODE]['cards'];oc=prior['conditions']['local_full']['cards']
        for i,(c,orig)in enumerate(zip(cards,oc)):
            sp=z[f'spans_{i}'];gate=z[f'presence_{i}'];score=z[f'{MODE}_logits_{i}']
            ck(np.array_equal(sp,o[f'spans_{i}'])and np.array_equal(sp,j[f'spans_{i}']),'spanssame')
            ck(np.array_equal(gate,o[f'presence_{i}'])and np.array_equal(gate,j[f'presence_{i}']),'gatesame')
            if c['family']not in[2,3]:
                ck(c==orig and np.array_equal(score,o[f'local_full_logits_{i}']),'otherfamilyexact');protected+=1;continue
            expected=np.asarray([o[f'local_full_logits_{i}'],j[f'splice_rank_logits_{i}']],np.float64).mean(axis=0)
            ck(np.array_equal(score,expected),'exact0.5scoremean')
            pick=np.flatnonzero(gate>=0)
            if not len(pick):ck(c['chosen'] is None,'missingadmissionunchanged');continue
            at=int(pick[score[pick].argmax()]);a,b=map(int,sp[at]);chosen=c['chosen'];ax=2*AX[c['family']]
            obs=[x for x in range(a,b+1)if q[x,ax,8]>.5];hq=[x for x in obs if q[x,ax,9]>.5]
            ck(chosen['span']==[a,b]and chosen['score']==float(score[at]),'selectorrawargmaxsame')
            ck(chosen['body_observed']==obs and chosen['hand_qualified']==hq,'qualifierssame')
            ck(c['semantic_completion'] is None and chosen['semantic_completion'] is None,'nocompletion')
            if 'truth'in c:
                tr=c['truth'];ck(tr['all_witness']==(set(tr['witness'])<=set(obs))and tr['wrong_points']==len(set(tr['wrong_witness'])&set(obs)),'witnessarithmetic')
        for mode in ['local_full','splice_rank',MODE]:
            cc=row['conditions'][mode]['cards'];actual=row['pts_order'][mode];pairs=[(i,j)for i,j in itertools.combinations(range(len(cc)),2)if sorted([cc[i]['family'],cc[j]['family']])==[2,3]]
            ck(len(actual)==len(pairs),'PTSpaircount')
            for (i,j),stored in zip(pairs,actual):
                spans=[cc[q]['chosen']['span']if cc[q]['chosen']else None for q in [i,j]];times=[[int(ts[x])for x in s]if s else None for s in spans]
                if any(v is None for v in times):state='unobserved';overlap=None
                else:
                    a,b=times;overlap=max(0,min(a[1],b[1])-max(a[0],b[0]));state='forward'if a[1]<=b[0]else'reverse'if b[1]<=a[0]else'ambiguous_overlap'
                ck(stored['spans']==spans and stored['time_extents_us']==times and stored['state']==state and stored['overlap_duration_us']==overlap,'independentPTSarithmetic')
                ptsrows.append(dict(seed=k[0],clip_id=k[1],kind=k[2],condition=mode,spans=spans,times=times,state=state,overlap_us=overlap))
    ck(protected==new['protected_other_family_cards']==174,'protected174')
    for summary in new['summary']:
        rr=[r for r in new['rows']if r['seed']==summary['seed']and r['kind']==summary['kind']];mode=summary['condition'];cards=[c for r in rr for c in r['conditions'][mode]['cards']];truth=[c['truth']for c in cards if'truth'in c]
        ck(summary['scenarios']==len(rr)and summary['cards']==len(cards),'summarydenominator')
        ck(summary['missing']==sum(c['chosen']is None for c in cards),'summarymissing')
        ck(summary['truth_cards']==len(truth)and summary['all_witness']==sum(t['all_witness']for t in truth)and summary['wrong_points']==sum(t['wrong_points']for t in truth),'summarywitness')
        ck(summary['PTS_order']==dict(Counter(o['state']for r in rr for o in r['pts_order'][mode])),'summaryPTS')
    fast=[r for r in ptsrows if r['condition']==MODE and r['kind']=='speed125'and r['state']!='forward']
    changes=[]
    oldpts={(r['seed'],r['clip_id'],r['kind'],r['condition']):r for r in ptsrows}
    for row in ptsrows:
        if row['condition']!=MODE:continue
        before=oldpts[row['seed'],row['clip_id'],row['kind'],'local_full'];jr=oldpts[row['seed'],row['clip_id'],row['kind'],'splice_rank']
        if row['state']!=before['state']or row['state']!=jr['state']:
            changes.append(dict(new=row,old=before,J=jr))
    summary=[r for r in new['summary']if r['condition']==MODE]
    verdict=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='SAVED_ARITHMETIC_VERIFIED_REMAINING_FAST_REGRESSION_NOT_G2_PASS',checks=checks,
        rows=246,source_scenarios=82,seeds=[42,43,44],protected_other_family_cards=174,alpha=.5,
        dynamic_witness=sum(r['all_witness']for r in summary if r['kind']=='dynamic_complete_blocks'),
        dynamic_wrong_points=sum(r['wrong_points']for r in summary if r['kind']=='dynamic_complete_blocks'),
        fast_unresolved=fast,PTS_relation_changes=changes,
        conclusion=['Fixedblend retainedJdynamicwitness18/18 anddynamicreverse9/9,normal9/9andslow9/9.',
            'Fastnormalorder8/9,withoneactualpositiveoverlap remaining; no allseednormalpreservation. No adoptionclaim.',
            'This experiment changes onlyrankselection. Correcthandshape/direction/semanticcompletion were not recomputed or newlyvalidated.',
            'No secondalpha, seedselection, thresholdsearch, newtraining ormodelinference. Preserve result and return rootG2decision.',
            'In deploymenttwo smallrankheads couldshareoneencoderpass; this savedarithmetic experimentdidnotmeasureelapseduserlatency.'],
        inference=False,training=False,web_changed=False,root_owns_G2=True,
        verification_scope='Separate arithmetic routine bysameCodeximplementer; parent independentreview remains needed.',
        pins={str(p):sha(p)for p in[Path(__file__),S/'evaluate_saved.py',S/'FROZEN.json',S/'RESULTS.json']})
    dest.write_text(json.dumps(verdict,ensure_ascii=False,indent=2),encoding='utf-8');print('Verifiedfixedblend',checks,'checks,remainingfast',fast,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (S/f'VERIFICATION_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
