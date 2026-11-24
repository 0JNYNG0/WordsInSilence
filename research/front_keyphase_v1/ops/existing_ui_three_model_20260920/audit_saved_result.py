"""Independent recount from original metadata and saved predictions; no replayer/model import."""
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path

ROLES=('parent','control','treatment')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def lines(path):
    p=Path(path)
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []

def audit(manifest_path,result_path):
    m=read(manifest_path);r=read(result_path);cfg=m['config'];checks=[]
    def check(name,ok):checks.append({'name':name,'passed':bool(ok)})
    def near(a,b):return a is None if b is None else isinstance(a,(float,int)) and math.isclose(a,b,abs_tol=1e-9)
    check('result tied to original manifest',r['manifest_sha256']==sha(manifest_path))
    check('input and implementation bytes remain pinned',all(Path(p).is_file() and sha(p)==h for p,h in m['input_hashes'].items()))
    check('missing files remain absent',all(not Path(p).exists() for p in m['absent_paths']))
    session_dir=Path(m['roots']['session_dir'])
    session=read(session_dir/'session.json')
    order=session['trial_ids'];rows={x['trial_id']:x for x in r['rows']}
    check('all original attempts retained in order',len(rows)==len(order)==len(r['rows']) and [x['trial_id'] for x in r['rows']]==order and r['rows']==m['rows'])
    check('original participant closed before comparison',session['state']=='CLOSED' and session['camera'] is False and session['participant_id']==cfg['participant_id'] and session['session_id']==cfg['session_id'])
    check('development limits present',set(('NOT_BLINDED','NOT_EXECUTION_CORRECTNESS'))<=set(r['study_labels']) and r['training_steps']==0 and r['model_promoted'] is False)
    check('three frozen identities and steps',set(r['models'])==set(ROLES) and all(r['models'][role]['sha256']==cfg['models'][role]['sha256'] and r['models'][role]['step']==(220 if role=='parent' else 660) for role in ROLES))
    grouped=defaultdict(dict)
    for rev in lines(session_dir/'trial_reviews.jsonl'):
        tid=rev['trial_id'];number=rev['revision']
        if tid not in rows or rev['session_id']!=session['session_id'] or rev['source']!='participant_self_report':raise ValueError('review provenance mismatch')
        prior=grouped[tid].get(number)
        if prior is not None and prior!=rev:raise ValueError('conflicting original review revision')
        grouped[tid][number]=rev
    latest={tid:rs[max(rs)] for tid,rs in grouped.items()}
    truth={tid:rev['actual_word_id'] for tid,rev in latest.items() if rev.get('actual_word_id') not in (None,'','UNKNOWN')}
    check('original known truths not replaced by requested word',all(rows[tid]['actual_word_id']==truth.get(tid) and rows[tid]['review_revision']==(latest[tid]['revision'] if tid in latest else None) for tid in order))
    eligible=set();artifact_counts=Counter();terminal_counts=Counter()
    failures={}
    for event in lines(session_dir/'timing_events.jsonl'):
        if event.get('event')=='trial_closed' or event.get('type')=='trial_closed':
            failures[event['trial_id']]=event.get('outcome',event.get('state'))
    for tid in order:
        stem=session_dir/'trials'/tid
        paths=[stem.with_suffix(x) for x in ('.result.json','.sidecar.json','.npz')]
        present=[p.is_file() for p in paths]
        if not any(present):status='MISSING_ALL'
        elif not present[0]:status='MISSING_RESULT'
        elif not present[1]:status='MISSING_SIDECAR'
        elif not present[2]:status='MISSING_NPZ'
        elif read(paths[0])['observation_sha256']==read(paths[1])['observation_sha256']==sha(paths[2]):status='COMPLETE'
        else:status='OBSERVATION_SHA_MISMATCH'
        if status=='COMPLETE':eligible.add(tid)
        terminal='FEEDBACK' if present[0] else failures.get(tid,'UNVERIFIED_TERMINAL')
        check(tid+' artifact and terminal evidence',rows[tid]['eligible']==(tid in eligible) and rows[tid]['artifact_status']==status and rows[tid]['terminal']==terminal)
        artifact_counts[status]+=1;terminal_counts[terminal]+=1
    summary=r['summary'];pred=r['predictions'];labels={w:d['label'] for w,d in cfg['words'].items()}
    check('all cohort denominators from original session/reviews',summary['all_attempts']==len(order) and summary['confirmed_n']==len(truth) and summary['unconfirmed_n']==len(order)-len(truth) and summary['eligible_n']==len(eligible))
    check('terminal/artifact tallies independent',summary['terminal_counts']==dict(terminal_counts) and summary['artifact_status_counts']==dict(artifact_counts))
    check('all18 actual-word supports',summary['coverage_by_actual']=={w:sum(t==w for t in truth.values()) for w in labels})
    check('predictions only for eligible trials',set(pred)==eligible)
    load_failures={e['trial_id'] for e in r['load_failures']}
    calls=lines(Path(result_path).parent/'inference_calls.jsonl')
    called=[e for e in calls if e.get('event')=='recognise']
    call_counts=Counter((e['trial_id'],e['role']) for e in called)
    wanted={(tid,role) for tid in eligible-load_failures for role in ROLES}
    check('actual call log unique and complete',set(call_counts)==wanted and all(n==1 for n in call_counts.values()) and len(called)==r['inference_calls']==3*len(eligible-load_failures))
    check('observation failures separately logged',{e['trial_id'] for e in calls if e.get('event')=='observation_load_error'}==load_failures)
    check('saved predictions agree with per-call evidence',all(e['project18_top1']==pred[e['trial_id']][e['role']]['project18'] and e['full93_top1']==pred[e['trial_id']][e['role']]['full93'] and e['status']==pred[e['trial_id']][e['role']]['status'] for e in called))
    independent={}
    for scope in ('project18','full93'):
        outcomes={};correct_sets={}
        for role in ROLES:
            pairs=[];answered={}
            for tid,word in truth.items():
                expected=word if scope=='project18' else labels[word]
                p=pred.get(tid,{}).get(role,{})
                answer=p.get(scope) if p.get('status')=='READY' else None
                answered[tid]=answer;pairs.append((expected,answer))
            hits={tid for tid,word in truth.items() if answered[tid]==(word if scope=='project18' else labels[word])}
            correct_sets[role]=hits;outcomes[role]=answered
            cells=Counter(pairs);support=Counter(t for t,p in pairs)
            macro=[];confusion=defaultdict(dict)
            for (expected,answer),n in cells.items():confusion[expected]['__NO_PREDICTION__' if answer is None else answer]=n
            metrics=summary['scopes'][scope]['per_role'][role]
            for wid,label in labels.items():
                value=wid if scope=='project18' else label
                n=support[value];tp=cells[(value,value)];fp=sum(count for (a,p),count in cells.items() if p==value and a!=value)
                if n:macro.append(2*tp/(n+tp+fp))
                check(scope+'/'+role+'/'+wid+' per-word support',metrics['per_class'][wid]['support']==n and metrics['per_class'][wid]['correct']==tp and near(metrics['per_class'][wid]['recall'],tp/n if n else None))
            check(scope+'/'+role+' accuracy/failures/confusion/F1',metrics['correct']==len(hits) and metrics['confirmed_n']==len(truth) and near(metrics['accuracy'],len(hits)/len(truth) if truth else None) and metrics['inference_failures_on_confirmed']==sum(p is None for a,p in pairs) and metrics['confusion']==dict(confusion) and metrics['macro_f1_class_count']==len(macro) and near(metrics['macro_f1'],sum(macro)/len(macro) if macro else None))
        for candidate,base in [('treatment','control'),('treatment','parent'),('control','parent')]:
            reported=summary['scopes'][scope]['paired'][candidate+'_vs_'+base]
            check(scope+'/'+candidate+'_vs_'+base+' paired outcomes',set(reported['gains'])==correct_sets[candidate]-correct_sets[base] and set(reported['regressions'])==correct_sets[base]-correct_sets[candidate] and set(reported['prediction_changes'])=={tid for tid in truth if outcomes[candidate][tid]!=outcomes[base][tid]} and reported['both_answered']==sum(outcomes[candidate][tid] is not None and outcomes[base][tid] is not None for tid in truth))
        independent[scope]={role:len(correct_sets[role]) for role in ROLES}
    return {'status':'PASS_SAVED_OUTPUT_RECOUNT' if all(c['passed'] for c in checks) else 'FAIL','passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'result_sha256':sha(result_path),'manifest_sha256':sha(manifest_path),'independent_correct':independent,'confirmed_n':len(truth),'all_attempts':len(order),'model_inference_calls_by_auditor':0,'limits':'One visible-feedback development participant; classification only, no performance-quality or population-generalization conclusion'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--result',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=audit(args.manifest,args.result)
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('status','passed','total','all_attempts','confirmed_n')}))
    raise SystemExit(0 if report['status'].startswith('PASS') else 1)
