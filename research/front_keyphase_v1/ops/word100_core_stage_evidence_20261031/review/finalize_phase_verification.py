"""Attach static review, fixed-findings lineage and stored role-table checks."""
import json
from pathlib import Path
from verify_phase_projection import UNIT, OLD, read, digest

target=UNIT/'review/PHASE_VERIFICATION.json'
report=read(target)
assert 'supplement_checks' not in report, 'Finalization already performed'
checks=[]
def ck(name, actual, expected=True):
    checks.append(dict(name=name,passed=actual==expected,detail=None if actual==expected else dict(actual=actual,expected=expected)))
for pin in report['files']:
    ck('inputs_unchanged/'+Path(pin['path']).name,digest(pin['path']),pin['sha256'])
ck('original_completed_EMPIRICAL_hash',digest(OLD/'EMPIRICAL_RANGE.json'),read(OLD/'COMPLETION.json')['evidence']['EMPIRICAL_RANGE.json'])
raw={c['alias']:c for n in ['PHASE_FIT.json','PHASE_DEV.json'] for c in read(UNIT/n)['clips']}
data=read(UNIT/'PHASE_DATA.json')
for c in data['clips']:
    ck(c['alias']+'/entire_rgb_metadata',c['rgb_review'],{k:v for k,v in raw[c['alias']].items() if k!='segments'})
diag={c['alias']:c for c in report['per_clip_diagnostics']}
for c in read(UNIT/'STAGE_PROJECTION.json')['clips']:
    expected={}
    for role,d in diag[c['alias']]['per_draft_role'].items():
        expected[role]=dict(draft_frames=d['draft_frames'],comparable_frames=d['match']+d['wrong_known_role'],agreement_frames=d['match'])
    ck(c['alias']+'/stored_by_draft_role',c['by_draft_role'],expected)
initial=read(UNIT/'review/PHASE_INITIAL_FINDINGS.json')
before=UNIT/'review/first_projection_uncertain_role'
for name in ['integrate_evidence.py','PHASE_DATA.json','STAGE_PROJECTION.json']:
    ck('initial_preserved/'+name,digest(before/name),initial['input_hashes'][name])
first=read(before/'STAGE_PROJECTION.json')
last=read(UNIT/'STAGE_PROJECTION.json')
for a,b in zip(first['clips'],last['clips']):
    ck(a['alias']+'/projection_unchanged_by_diagnostic_fix',a['reference_projections'],b['reference_projections'])
    ck(a['alias']+'/consensus_unchanged_by_diagnostic_fix',a['consensus_roles'],b['consensus_roles'])
ck('initial_comparable',first['summary']['draft_comparable_frames'],445)
ck('final_comparable',last['summary']['draft_comparable_frames'],444)
ck('final_agreement',last['summary']['draft_agreement_frames'],442)
role=report['per_draft_role_diagnostics']
active={k:sum(d[k] for r,d in role.items() if r!='rest') for k in ['draft_frames','match','projected_unknown','wrong_known_role']}
report['diagnostic_denominators']={
 'all_native_dev_frames':624,
 'projected_known':470,'projected_unknown':154,
 'dev_draft_known':sum(d['known_dev_draft'] for d in diag.values()),
 'dev_draft_unknown':sum(d['unknown_dev_draft'] for d in diag.values()),
 'both_known':444,'same_role_on_both_known':442,'different_role_on_both_known':2,
 'projected_known_on_dev_draft_unknown':sum(d['projected_known_on_draft_unknown'] for d in diag.values()),
 'projected_unknown_on_dev_draft_known':sum(d['known_draft_predicted_unknown'] for d in diag.values()),
 'rest_only':role['rest'],'active_roles_excluding_rest':active,'core_candidate_only':role['core_candidate'],
 'interpretation':'442/444 is conditional on both drafts being known. All 624 frames remain; the known DEV-draft denominator is552, with108 projected unknown and2different roles. Rest contributes262 of442 agreements. Active roles are180same/2different/68unknown across250draft frames. Core candidate roles are135same/1different/43unknown across179draft frames. None is recognition or linguistic accuracy.'
}
report['static_review']=[
 {'topic':'DEV leakage','result':'PASS','evidence':'Projection uses fixed saved paths, query frame count from AUTHORIZATION, and fit segments only. DEV segments enter observed=roles(q) after consensus construction. No DEV labels affect reference selection, path, mask or role vote.'},
 {'topic':'single core and3references','result':'PASS_FOR_PINNED_INPUTS','evidence':'Each query groups every saved reference-frame mapping; set values retain segment_id and role. Any None or multiple segment IDs yields unknown. Exact expected fit-reference set plus length3 guards the current3-reference-per-word AUTHORIZATION; independent verifier also asserts3distinct IDs.'},
 {'topic':'source text isolation','result':'PASS','evidence':'Source clauses add anchor-scoped metadata only. Original RGB segment fields remain identical, and full clip-level RGB limitations survive in rgb_review. Source text does not alter input arrays, role labels or projection.'},
 {'topic':'coordinate support','result':'PASS_ARITHMETIC_ONLY','evidence':'Counts recomputed by feature names, bool(mask) and finite coordinate checks. Expected angle denominator is14values per hand per included draft-core frame, including2thumb and12nonthumb angles. Presence is not coordinate correctness or handshape truth.'},
 {'topic':'semantic/temporal claims','result':'LIMITED_DIAGNOSTIC','evidence':'Coarse physical role agreement cannot prove shared stage identity, exact automatic boundaries, required order/count, completion or permissible variation. Fit and DEV segmentation resolutions differ, and spans are supported by sampled RGB, not dense independent human GT.'},
 {'topic':'bounded input robustness','result':'LIMITATION','evidence':'This review validates the pinned16clip inputs and all stored12paths. It is not a guarantee for arbitrary malformed or changed labels, missing files or changed AUTHORIZATION.'}
]
report['resolved_findings']=[
 {'id':'P1_UNCERTAIN_INCLUDED_IN_DIAGNOSTIC','resolution':'roles skips uncertain; P036f89 now remains unknown in DEV comparison.442/444 and2different roles independently confirmed. All reference projections unchanged.'},
 {'id':'P2_FEATURE_UNKNOWN_NOT_PROPAGATED','resolution':'rgb_review retains entire source clip metadata including unknown, notes and reviewer_note; deep equality confirmed for16clips.'},
 {'id':'P2_REFERENCE_IDENTITY_GUARD','resolution':'Implementation checks expected fit-reference set and length3; independently checked3distinct expected identities for all4queries.'}
]
report['initial_artifacts_preserved']=[dict(path=str(before/n),sha256=digest(before/n)) for n in ['integrate_evidence.py','PHASE_DATA.json','STAGE_PROJECTION.json','CAUSE.json']]
report['supplement_checks']=checks
report['checks']['total']+=len(checks)
report['checks']['passed']+=sum(c['passed'] for c in checks)
report['checks']['failed']+=sum(not c['passed'] for c in checks)
report['failures'] += [c for c in checks if not c['passed']]
report['files'].append(dict(path=str(Path(__file__).resolve()),sha256=digest(Path(__file__))))
report['status']='PASS_STORED_ARITHMETIC_AND_BOUNDED_CONTRACT' if not report['failures'] else 'FAIL_REVIEW_FINDINGS_REMAIN'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=(
 '공개16영상의 단계 근거 파생 및 저장 DTW 역할 투영 검산\n'
 f"상태: {report['status']}. 검사{report['checks']['total']}개, 불일치{report['checks']['failed']}개.\n"
 '16영상2084원본프레임의 frame/PTS,110구간의 원시RGB필드·관측지원 수, 출처 연결, 저장12DTW경로를 확인했다.\n'
 'DEV624프레임:3참고 역할 합의470,unknown154. DEV초안이 알려진552중 같은역할442, 다른역할2, 투영unknown108이다.\n'
 '양쪽 알려진 조건부 비교만 쓰면442/444이다. 휴식의 일치262가 포함되므로 인식정확도로 해석하지 않는다.\n'
 '핵심후보만179중135같음/1다름/43unknown. 휴식 제외 전체활동은250중180같음/2다름/68unknown이다.\n'
 '처음에는 uncertain1프레임을 비교해442/445로 표시했다. 최초파일 보존을 확인했고 수정 후 투영배열 자체는 바뀌지 않았다.\n'
 '손 겹침·깊이 등 unknown 및 전체RGB메타데이터 보존,3개별참고 검사를 확인했다.\n'
 '출처 연결19구간/48개 구간-문구 연결/158개 anchor출현이며 서로 다른 단위다.\n'
 '새학습·모델추론·DTW재탐색·과거모델재채점은 없다. 같은Codex초안이며 독립 사람 수어정답/정확경계/필수순서/교육정오 검증이 아니다.\n'
)
with (UNIT/'review/PHASE_VERIFICATION.txt').open('x',encoding='utf-8') as f:f.write(text)
print(json.dumps(dict(status=report['status'],checks=report['checks'],sha256=digest(target)),ensure_ascii=False))
