"""Report saved verified results; never calls a detector/model/DTW/comparator."""
import gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
U=Path(__file__).resolve().parent
def read(p):
    p=Path(p);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
b=read(U/'BRIDGE_MANIFEST.json');r=read(U/'RECOVERY_RESULT_MANIFEST.json')
checks={k:read(U/'review'/k) for k in ['LINEAGE_DATA_AUDIT.json','RECOVERY_SOURCE_VERIFICATION.json','RECOVERY_MATH_VERIFICATION_FINAL.json','BRIDGE_VERIFICATION.json']}
assert all(v['status'].startswith('PASS') for v in checks.values())
old=b['baseline_counts'];now=b['candidate_counts'];n=old['possible'];assert n==now['possible']==30560
cases=defaultdict(lambda:{'old':defaultdict(int),'new':defaultdict(int),'parents':0,'changed':0})
for row in b['records']:
    d=cases[row['query']];d['parents']+=1
    geo=read(row['geometry']['path']);base=read(geo['baseline_geometry']['path']) if row['mode'].startswith('NEW_') else geo
    d['changed']+=int(row['mode'].startswith('NEW_'))
    for which,obj in [('old',base),('new',geo)]:
        for group in obj['evidence']['groups'].values():
            for side in ('query','reference'):
                for k,v in group[side]['counts'].items():d[which][k]+=v
facts={'scope':{'public_clips_selected':8,'source_frames':303,'selected_windows':10,'evaluation_cases':4,'pairs':12,'events':39},
       'recovery_counts':r['counts'],'baseline':old,'candidate':now,
       'comparison_support_gain':now['matched_support']-old['matched_support'],
       'comparison_supported_percent_before':old['matched_support']/n*100,
       'comparison_supported_percent_after':now['matched_support']/n*100,
       'comparison_supported_percentage_points':(now['matched_support']-old['matched_support'])/n*100,
       'unavailable_relative_reduction_percent':(old['not_comparable']-now['not_comparable'])/old['not_comparable']*100,
       'cases':cases,'classification_or_instructional_accuracy':None,'new_training':0,
       'evidence':{str(U/'review'/k):sha(U/'review'/k) for k in checks},
       'visual_review':{'path':str(U/'review/RECOVERY_VISUAL_REVIEW.json'),'sha256':sha(U/'review/RECOVERY_VISUAL_REVIEW.json')},
       'candidate_outputs_not_deployed':True}
new(U/'FINAL_REPORT_FACTS.json',facts)
lines=['WordsInSilence — 비어 있던 손 관측 보완과 동작 비교 연결 결과',
'2026-11-01 / 공개 자료만 사용한 별도 연구 후보 / 기존 C·T3-F와 운영 웹·DTW 보존','',
'쉽게 말하면',
'영상에 손이 있는데 처음 관절 검출이 놓친 순간을 다시 살펴, 쓸 수 있는 관측을 추가했다. 그 정보를 저장만 하지 않고 이전 동작 비교에도 연결했다. 원래 보이던 손의 값과 비교할 전체 시간 범위는 그대로 지켰다.',
'이번에는 관찰 가능한 정보가 늘었다. 단어 정답률이나 잘못된 수어를 지적하는 정확도가 올랐다는 시험은 아니다.','',
'1. 무엇이 문제였나',
'고정한 공개 8영상의 10시간창, 원본 303프레임을 검출→좌우 배정→상반신 범위→각도 계산 순서로 추적했다.',
'잘하다(P048)의 72~74프레임은 처음 저장한 검출에 왼손 후보 자체가 없었다. 반면 순종(P036)의 몸 옆으로 내린 왼손은 관측은 있었지만 정해둔 상반신 범위 밖이었다. 둘을 같은 누락으로 취급해 복원하면 안 된다.',
'이 범위에서 원래 규칙과 다르게 값을 지운 계산 버그는 발견되지 않았다. 그래서 마스크를 풀지 않고, 실제 검출이 없는 순간만 다시 관측하도록 연결했다.','',
'2. 실제로 바꾼 방법',
'현재 저장한 손이 통째로 없고, 팔꿈치·손목 근거가 있으며, 손목이 원래 비교 범위 안인 순간에만 기존 로컬 MediaPipe로 다시 찾는다. 해당 원본 전체 화면과 같은 장면의 손 주변을 확인한다.',
'좌표는 원본 화면 크기로 합치며, 현재 팔 연결과 기존 모든 손 후보를 확인해 같은 손을 두 번 넣지 않는다. 기존 손·부분 관측은 덮어쓰지 않는다. 서로 일치하지 않거나 좌우를 확인할 수 없는 점은 보류한다.',
'새로운 모델·다운로드·단어별 예외·결과에 맞춘 임계값 변경은 없다. 같은 공통 규칙이 적용되지만, 모든 수어에서의 성능이 검증된 것은 아니다.','',
'3. 이전 → 현재',
'재검출 대상: 20프레임, 비어 있던 손 22회 중 18회에 추가 관측. 여기서 1회는 한 프레임의 왼손 또는 오른손 한 개다.',
'잘하다의 72·73프레임 왼손은 다시 찾았고, 74프레임은 충분한 근거가 없어 그대로 보류했다.',
'입력: 원래 지원됐던 특징값 16,154개 모두 동일; 새 지원 604개 추가. 전체 20,907개의 입력 특징 기회는 그대로다.',
f"동작 비교: 전체 {n:,}항목 그대로, 비교 근거 있음 {old['matched_support']:,} → {now['matched_support']:,}개 (+{facts['comparison_support_gain']:,}).",
f"비교 근거 비율 {facts['comparison_supported_percent_before']:.2f}% → {facts['comparison_supported_percent_after']:.2f}% (+{facts['comparison_supported_percentage_points']:.2f}%p). 이 수치는 정확도가 아니다.",
f"비교 근거 부족 {old['not_comparable']:,} → {now['not_comparable']:,}개. 부족 항목의 {facts['unavailable_relative_reduction_percent']:.2f}% 감소.",
'남은 5,890개가 모두 고쳐야 할 검출 오류인 것은 아니다. 이 수에는 순종의 미지원 비교 항목 4,912개도 포함된다. 이번 원본 확인 범위에서는 몸 옆으로 내린 손의 의도된 상반신 영역 제외가 확인됐으나, 4,912항목 전체의 원인을 전수 판독한 것은 아니다. 숫자를 줄이기 위해 비교 범위를 줄이거나 영역 밖 손을 넣지 않았다.',
'이전 39비교 구간과 12영상쌍·각 3참고를 유지했다. 새 관측의 영향이 있는 11구간만 새 기하 비교를 계산했고, 나머지 28구간은 저장된 결과를 그대로 재사용했다.',
'기존 시간 대응 경로는 그대로다. 비교 가능했던 모든 기존 연결의 차이값도 그대로 유지됐다. 새 정보가 늘면서 평균 차이는 커지거나 작아질 수 있으며, 작아졌다고 좋은 동작으로 판정하지 않는다.','',
'사례별 비교 근거 (같은 분모, 여러 참고·양쪽 시간축 포함)']
names={'P036':'순종','P048':'잘하다','P066':'조상','P072':'이집트'}
for alias,d in cases.items():
    lines.append(f"- {names[alias]}({alias}): {d['old']['matched_support']:,}/{d['old']['possible']:,} → {d['new']['matched_support']:,}/{d['new']['possible']:,}; 새 계산 {d['changed']}/{d['parents']}구간.")
lines += ['', '4. 무엇을 확인했나',
'기존 좌표·지원값·시간·상반신 범위·양손 배정 보존과 새 좌표/각도 계산을 별도 검산했다. 코드 자체 검사 31개, 계보 검사 8,229개, 새 입력 수치 검사 5,522개와 출처·특징 검사 7,182개, 비교 연결 검사 33,749개가 통과했다. 이 숫자는 시험 항목 수이며 정확도 표본 수가 아니다.',
'재검출한 20프레임 전체와 추가 18손을 원본 RGB 옆에서 확인했다. 이 작은 범위의 몸통·팔 화면에서는 큰 좌우 교환이나 반대 손 중복이 새로 보이지 않았다. 원래 정상 손과 교차의 저장값도 변하지 않았다.',
'검산은 별도 코드와 다른 Codex 작업 담당으로 수행했다. 독립 수어 전문가의 정답 검토는 아니다. 닫히거나 겹친 손의 모든 손가락 점이 정확하다고 확인한 것도 아니다.',
'검산 과정에서 native 관측과 ROI 관측을 혼동한 검산기 오류 및 정수 JSON 저장 오류가 있었다. 최초 오류를 보존하고 검산기만 수정해 통과했다. 관절 검출·완료 결과는 재실행하지 않았다.','',
'5. 남은 문제와 처리',
'P048 74프레임과 P063 72프레임은 영상에 손이 보이지만 전체/손주변 검출의 공통 지원을 얻지 못했다. P063 57프레임은 검출끼리 불일치했고, P066 71프레임은 겹침에서 어느 팔의 손인지 확정하기 어려웠다. 이 4회는 강제로 채우지 않았다.',
'P072 70프레임의 새 두 손 중 6관절은 서로 확인되지 않아 보류했다. 새 지원 372관절도 관절별 사람 정답과 비교한 정확도는 미검증이다.',
'조상의 기존 시간 정렬에는 서로 다른 손이 위에 있는 순간을 대응시킨 사례(P066 68 ↔ P063 58)가 남아 있다. 손이 더 보인다는 사실만으로 같은 동작 단계가 됐다고 볼 수 없다.',
'기존 학습된 손모양 점수와 위치 경로 점수는 원래 입력에서 나온 결과로 따로 보존한다. 새 관측을 학습한 새 점수인 것처럼 바꾸지 않았다.',
'수어 설명과의 연결은 기존의 부분 확인 수준을 유지한다. 모양 차이·위치 차이가 실제 의미를 해치는지, 허용되는 표현 차이인지는 이번 시험으로 확정하지 않았다.','',
'6. 속도와 적용 범위',
'새 선택적 관측 실행은 이 8영상의 대상 20프레임을 처리하는 데 내부 측정 약 5.13초였다. 전체 화면 검출 약 0.79초, 손 주변 검출 약 0.74초이고 원본 해독·초기화 등도 포함된다. 한 사용자 영상의 전체 피드백 대기시간 측정이 아니다.',
'별도 연구 입력·비교 연결까지 완료했다. 운영 웹, 기존 C·T3-F, 원본·과거 결과는 바꾸지 않았다. 사용자·멘토 촬영은 읽거나 학습·튜닝·선택에 쓰지 않았다. 추가 학습은 하지 않았다.',
'기존 http://127.0.0.1:8800/ 화면은 이전 결과다. 이번에는 교육 성능 향상이 입증되지 않았으므로 새 HTML은 만들지 않았다. 원본/새 관측 비교 이미지는 review/recovery_rgb/에 있다.','',
'후속 방향',
'관측을 더 확보한 부분은 완료했고, 남은 주된 문제는 같은 주요 동작 단계끼리 비교하는지다. 새 모델을 계속 추가하기보다, 현재 공개 원본의 앞뒤 양손 관계·방향·모양 근거로 대응 후보를 확인하고 정상 속도 차이·반복을 보존하는 조건을 먼저 확정한다.',
'관측이 없는 순간을 목표 설명대로 채워 맞추거나, 차이 수치를 낮추는 것을 성공으로 삼지 않는다. 단계 대응의 실제 근거와 평가 기준이 마련된 범위에서만 다음 정렬 보완을 실행한다. 일반적인 후속 검증/수정에는 같은 승인을 반복 요청하지 않는다.','',
'주요 근거: FINAL_REPORT_FACTS.json, RECOVERY_ADOPTION.json, BRIDGE_MANIFEST.json, review/*VERIFICATION*.json, review/RECOVERY_VISUAL_REVIEW.json.']
(U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'comparison_support_gain':facts['comparison_support_gain'],'cases':cases},ensure_ascii=False))
