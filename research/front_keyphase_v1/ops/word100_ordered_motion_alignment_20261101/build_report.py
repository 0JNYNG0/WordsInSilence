from pathlib import Path
from collections import Counter
import json,hashlib
U=Path(__file__).resolve().parent;AUD=U.parent/'word100_crossperson_stage_audit_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
    data=read(AUD/'REPORT_DATA.json');sources={}
    for r in read(U/'RESULT_MANIFEST.json')['results']:
        if r['kind']=='person_pair':
            assert hashlib.sha256(Path(r['file']).read_bytes()).hexdigest()==r['sha256']
            d=read(r['file']);sources[d['name']]=d
    counts={};details=[]
    for role in ['fit','development']:
        c=Counter()
        for case in data['cases']:
            if case['role']!=role:continue
            rs={a['segment_id']:a for a in sources[case['id']]['anchors']}
            for e in case['events']:
                a=rs[e['segment_id']];local=e['methods']['core_first'];held=bool(a['flags'])
                local_category=('different_kind' if local['state']=='different_kind_core'
                    else 'different_ordinal' if local['occurrence']=='different_documented_ordinal'
                    else 'same_kind_same_ordinal' if local['state']=='same_kind_core' and local['occurrence']=='same_documented_ordinal'
                    else 'same_kind_ordinal_unknown' if local['state']=='same_kind_core' else local['state'])
                c['all']+=1;c[local_category]+=1;c[local_category+('_held' if held else '_compatible')]+=1
                c['global_observation_insufficient']+=bool(e['methods']['global_dtw']['insufficient_groups'])
                details.append(dict(case=case['id'],segment=e['segment_id'],role=role,
                    local_category=local_category,local_research_supported=local.get('saved_state')=='supported_core_candidate',
                    flags=a['flags'],policy_status=a['status'],original_global_state=e['methods']['global_dtw']['state']))
        counts[role]=dict(c)
    for c in data['cases']:
        rs={a['segment_id']:a for a in sources[c['id']]['anchors']}
        for e in c['events']:
            e['guard']={k:v for k,v in rs[e['segment_id']].items() if k not in ['unchanged_global_groups','saved_physical_audit_global','saved_physical_audit_local']}
            e['methods']={k:v for k,v in e['methods'].items() if k in ['global_dtw','core_first']}
    data['policy_summary']=read(U/'SUMMARY.json');data['evidence_counts']=counts
    (U/'POLICY_EVIDENCE_AUDIT.json').write_text(json.dumps(dict(status='DEVELOPMENT_DIAGNOSTIC_NOT_EDUCATIONAL_ACCURACY',counts=counts,rows=details),ensure_ascii=False,indent=2),encoding='utf8')
    (U/'REPORT_DATA.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    print(json.dumps(counts,ensure_ascii=False))
if __name__=='__main__':main()
