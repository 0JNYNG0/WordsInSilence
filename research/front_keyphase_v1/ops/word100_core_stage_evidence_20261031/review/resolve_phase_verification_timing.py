"""Correct a reviewer snapshot-time error without changing implementation evidence."""
import ast, json
from pathlib import Path
from verify_phase_projection import UNIT, read, digest

target=UNIT/'review/PHASE_VERIFICATION.json'
report=read(target)
before=UNIT/'review/first_projection_uncertain_role'
initial=read(UNIT/'review/PHASE_INITIAL_FINDINGS.json')
assert len(report['failures'])==3
assert all(x['name'].startswith('initial_preserved/') for x in report['failures'])
backup=UNIT/'review/PHASE_VERIFICATION_REVIEWER_TIMING_ERROR.json'
with backup.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
correction={
 'type':'REVIEWER_SNAPSHOT_TIMING_ERROR',
 'cause':'The reviewer recorded PHASE_INITIAL_FINDINGS input hashes after root had already applied the fixes. Those hashes are the final artifacts, not the originally observed pre-fix artifacts. Comparing the earlier preserved snapshot to them was an invalid expectation.',
 'initial_failed_report':str(backup),'initial_failed_report_sha256':digest(backup),
 'original_artifacts_modified':False,'labels_or_paths_changed':False,
 'verification_limit':'The pre-fix artifacts are checked for correspondence with the directly observed initial behavior and code. No claim of equality to a hash captured before that first read is made, because such a reviewer hash was not captured.',
 'invalid_checks_preserved':report['failures']
}
with (UNIT/'review/PHASE_REVIEWER_TIMING_CORRECTION.json').open('x',encoding='utf-8') as f:json.dump(correction,f,ensure_ascii=False,indent=2);f.write('\n')
def ck(name,actual,expected=True):
    report['supplement_checks'].append(dict(name=name,passed=actual==expected,detail=None if actual==expected else dict(actual=actual,expected=expected)))
report['supplement_checks']=[x for x in report['supplement_checks'] if not x['name'].startswith('initial_preserved/')]
for name in ['integrate_evidence.py','PHASE_DATA.json','STAGE_PROJECTION.json']:
    ck('review_hash_was_final/'+name,digest(UNIT/name),initial['input_hashes'][name])
    ck('preserved_before_distinct/'+name,digest(before/name)!=digest(UNIT/name))
first=read(before/'STAGE_PROJECTION.json');last=read(UNIT/'STAGE_PROJECTION.json')
ck('before_denominator445',first['summary']['draft_comparable_frames'],445)
ck('before_agreement442',first['summary']['draft_agreement_frames'],442)
first_p36=next(c for c in first['clips'] if c['alias']=='P036')
ck('before_exact_uncertain_mismatch',first_p36['draft_disagreement'],[dict(frame=89,projected='core_candidate',rgb_draft='uncertain')])
ck('after_denominator444',last['summary']['draft_comparable_frames'],444)
ck('after_agreement442',last['summary']['draft_agreement_frames'],442)
ck('after_two_known_role_mismatches',sum(len(c['draft_disagreement']) for c in last['clips']),2)
tree=ast.parse((before/'integrate_evidence.py').read_text(encoding='utf-8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='roles')
namespace={}
exec(compile(ast.Module(body=[function],type_ignores=[]),'preserved_before_roles_function','exec'),namespace)
fixture=dict(alias='test',frames=2,segments=[dict(role='uncertain',segment_id='u',core=[1,1])])
ck('before_roles_reproduces_uncertain_as_known',namespace['roles'](fixture),[None,('u','uncertain')])
first_data={c['alias']:c for c in read(before/'PHASE_DATA.json')['clips']}
last_data={c['alias']:c for c in read(UNIT/'PHASE_DATA.json')['clips']}
ck('before_rgb_metadata_omitted',all('rgb_review' not in c for c in first_data.values()))
for a,c in first_data.items():
    ck(a+'/RGB_segments_unchanged_by_fix',c['segments'],last_data[a]['segments'])
report['verification_process_correction']=correction
all_checks=report['check_details']+report['supplement_checks']
report['failures']=[x for x in all_checks if not x['passed']]
report['checks']=dict(total=len(all_checks),passed=sum(x['passed'] for x in all_checks),failed=len(report['failures']))
report['status']='PASS_STORED_ARITHMETIC_AND_BOUNDED_CONTRACT' if not report['failures'] else 'FAIL_REVIEW_FINDINGS_REMAIN'
report['files'].append(dict(path=str(Path(__file__).resolve()),sha256=digest(Path(__file__))))
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
textpath=UNIT/'review/PHASE_VERIFICATION.txt'
text=textpath.read_text(encoding='utf-8')
lines=text.splitlines()
lines[1]=f"상태: {report['status']}. 검사{report['checks']['total']}개, 불일치{report['checks']['failed']}개."
lines.append('검토자가 최초이슈 로그에 수정후 해시를 기록한 시점 오류는 별도 보존·정정했다. 최초코드의 uncertain처리와442/445보존 내용을 직접 확인했으며 보존본은 수정하지 않았다.')
textpath.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(status=report['status'],checks=report['checks'],failures=report['failures'],sha256=digest(target)),ensure_ascii=False))
