"""Recount all fixed cohort/native gains and losses from saved outputs only."""
from pathlib import Path
import json
import hashlib
from collections import Counter

OPS=Path(__file__).resolve().parent;WEB=OPS.parent.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[];comparisons={}
def check(name,passed):checks.append(dict(name=name,passed=bool(passed)))
def pred(row,arm,scope):return row['arms'][arm]['scopes']['project18' if scope=='project18' else 'full']['top1']
for cohort in ('old','new'):
    p=read(OPS/f'web_{cohort}_plan_LOCAL_ONLY.json');r=read(OPS/f'web_{cohort}_result_LOCAL_ONLY.json')
    s=WEB/'runs'/p['session_id'];session=read(s/'session.json')
    reviews={}
    for line in (s/'trial_reviews.jsonl').read_text(encoding='utf-8').splitlines():
        v=json.loads(line);reviews[v['trial_id']]=v
    ids=[x['trial_id'] for x in r['rows']]
    check(cohort+' all18 closed-session trials preserved',session['state']=='CLOSED' and len(ids)==len(set(ids))==18 and set(ids)==set(session['trial_ids']) and ids==[x['trial_id'] for x in p['rows']])
    check(cohort+' actual labels originate from participant confirmations',all(x['actual_word_id']==reviews[x['trial_id']]['actual_word_id'] and not reviews[x['trial_id']]['is_expert_verification'] for x in r['rows']))
    check(cohort+' exact original observation bytes',all(sha(Path(x['observation']))==x['observation_sha256'] for x in r['rows']) and all({k:x[k] for k in p['rows'][i]}==p['rows'][i] for i,x in enumerate(r['rows'])))
    check(cohort+' same model pins and protected files',r['models']==p['models'] and all(sha(Path(v['path']))==v['sha256'] for v in p['models'].values()) and all(sha(Path(k))==v for k,v in p['protected_sha256'].items()))
    check(cohort+'36 forward calls zero training or promotion',r['inference_calls']==36 and r['optimizer_steps']==0 and not r['model_promoted'] and not r['production_registry_or_default_changed'])
    check(cohort+' all arm inference ready same frozen checkpoint',all(x['arms'][a]['model_status']=='READY' and x['arms'][a]['checkpoint_sha256']==p['models'][a]['sha256'] and x['arms'][a]['checkpoint_step']==660 for x in r['rows'] for a in ('control','treatment')))
    prior_quota=read(WEB/f'ops/public_replay_quota_20260920/web_{cohort}_result_LOCAL_ONLY.json')
    prior_head=read(WEB/'ops/head_only_native_pair_20260919/web_result_LOCAL_ONLY.json') if cohort=='old' else read(WEB/'ops/existing_ui_three_model_20260920/ses_3cec3d73b407/comparison/result_LOCAL_ONLY.json')
    comparisons[cohort]={}
    for scope in ('project18','full93'):
        truth='actual_word_id' if scope=='project18' else 'actual_label'
        summary=r['summary'][scope];now={};comparisons[cohort][scope]={}
        for arm in ('control','treatment'):
            now[arm]={x['trial_id']:pred(x,arm,scope) for x in r['rows']}
            correct=sum(now[arm][x['trial_id']]==x[truth] for x in r['rows'])
            check(cohort+' '+scope+' '+arm+' independent correctness',correct==summary['arms'][arm]['correct'] and summary['arms'][arm]['observation_or_inference_failures']==0)
            for bname,bdata in [('no_anchor',prior_head),('quota',prior_quota)]:
                before=({x['trial_id']:pred(x,arm,scope) for x in bdata['rows']} if bname=='quota' or cohort=='old' else {k:v[arm][scope] for k,v in bdata['predictions'].items()})
                gains=[x['actual_word_id'] for x in r['rows'] if before[x['trial_id']]!=x[truth] and now[arm][x['trial_id']]==x[truth]]
                losses=[x['actual_word_id'] for x in r['rows'] if before[x['trial_id']]==x[truth] and now[arm][x['trial_id']]!=x[truth]]
                comparisons[cohort][scope][arm+'_vs_'+bname]={'correct':correct,'previous_correct':sum(before[x['trial_id']]==x[truth] for x in r['rows']),'gains':gains,'regressions':losses}
        gains=[x['actual_word_id'] for x in r['rows'] if now['control'][x['trial_id']]!=x[truth] and now['treatment'][x['trial_id']]==x[truth]]
        losses=[x['actual_word_id'] for x in r['rows'] if now['control'][x['trial_id']]==x[truth] and now['treatment'][x['trial_id']]!=x[truth]]
        check(cohort+' '+scope+' paired gains/regressions',sorted(gains)==sorted(summary['gains']) and sorted(losses)==sorted(summary['regressions']))
        if cohort=='new':
            before={k:v['parent'][scope] for k,v in prior_head['predictions'].items()}
            comparisons[cohort][scope]['treatment_vs_parent']={'correct':sum(now['treatment'][x['trial_id']]==x[truth] for x in r['rows']),'previous_correct':sum(before[x['trial_id']]==x[truth] for x in r['rows']),'gains':[x['actual_word_id'] for x in r['rows'] if before[x['trial_id']]!=x[truth] and now['treatment'][x['trial_id']]==x[truth]],'regressions':[x['actual_word_id'] for x in r['rows'] if before[x['trial_id']]==x[truth] and now['treatment'][x['trial_id']]!=x[truth]]}
public=read(OPS/'public_reference_result.json');previous=read(WEB/'ops/public_replay_quota_20260920/public_reference_result.json')
check('same public metadata and approved five conditions',public['reference_meta_sha256']==previous['reference_meta_sha256']==sha(Path(public['reference_meta'])) and public['eligible']==previous['eligible'])
for arm in ('control','treatment'):
    a=public['results'][arm];b=previous['results'][arm]
    check(arm+' public exact IDs truths conditions',[(x['id'],x['true'],x['condition']) for x in a['rows']]==[(x['id'],x['true'],x['condition']) for x in b['rows']])
    for scope in ('project18','full93'):
        check(arm+' '+scope+' public counts',all(a[scope][c]['n']==sum(x['condition']==c for x in a['rows']) and a[scope][c]['correct']==sum(x['condition']==c and x[scope]['pred']==x['true'] for x in a['rows']) for c in ('intact','reviewed_masked')))
check('46 public forwards zero training',public['inference_calls']==46 and public['training_steps']==0)
native=read(WEB/'runs/parent_anchor_pair_v1_recovery/results.json');old=read(WEB/'runs/head_only_native_pair_v1/results.json');native_changes={}
for arm in ('control','treatment'):
    native_changes[arm]={}
    for key in ('full','source696','additional3','target_nikl18'):
        rows=native['arms'][arm]['evaluations'][key]['rows'];base=old['arms'][arm]['evaluations'][key]['rows'];prior={x['id']:x for x in base}
        check(arm+' '+key+' same native IDs and truths as unanchored',[(x['id'],x['true']) for x in rows]==[(x['id'],x['true']) for x in base])
        gains=[];losses=[];changes=[]
        for x in rows:
            b=prior[x['id']]
            if b['pred']!=x['pred']:
                item={'id':x['id'],'true':x['true'],'before':b['pred'],'after':x['pred']};changes.append(item)
                if b['pred']!=x['true'] and x['pred']==x['true']:gains.append(item)
                if b['pred']==x['true'] and x['pred']!=x['true']:losses.append(item)
        native_changes[arm][key]={'correct':sum(x['true']==x['pred'] for x in rows),'previous_correct':sum(x['true']==x['pred'] for x in base),'gains':gains,'regressions':losses,'prediction_changes':changes}
report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,'web_comparisons':comparisons,'native_comparisons':native_changes,'inference_calls_after_training':118,'training_in_audit':0,'new_inference_in_audit':0,'user_records_in_training':False,'model_promoted':False}
with (OPS/'FOLLOWUP_INDEPENDENT_AUDIT_LOCAL_ONLY.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'native_changes':native_changes,'web_new':comparisons['new']},ensure_ascii=False))
raise SystemExit(0 if report['status']=='PASS' else 1)
