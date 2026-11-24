from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (U/n).open('x',encoding='utf-8')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
c=read(U/'TRAINING_FROZEN.json');train=read(U/'TRAINING_COMPLETE.json');comp=read(U/'COMPARISON.json');ver=read(U/'SAVED_VERIFICATION.json')
assert ver['status']=='PASS'and ver['comparison_sha256']==sha(U/'COMPARISON.json');checks=0
for p,h in {**c['pins'],**c['protected_pins']}.items():assert sha(p)==h,p;checks+=1
for r in train['rows']:
    for name in ['model','prediction']:assert sha(r[name])==r[name+'_sha256'];checks+=1
assert len(train['rows'])==36
def vec(base,code,field):return '/'.join(str(next(r for r in comp['aggregate']if r['baseline']==base and r['seed']==s)['categories'][str(code)][field])for s in c['seeds'])
def family(base,f,field):return '/'.join(str(next(r for r in comp['by_family']if r['baseline']==base and r['seed']==s and r['family']==f)['categories']['1'][field])for s in c['seeds'])
passed=comp['status']=='DIAGNOSTIC_PASS'
lines=['C9 시간 순서를 보존하는 작은 판정부 비교','',
    '결론: '+('고정 진단의 움직임·유지 보존 조건을 통과했다. 웹·G2 전체 통과는 아직 아니다.'if passed else'관측 순서를 보존하는 후보를 고정 비교했지만, 움직임·유지 전체 보존 조건은 통과하지 못했다. 웹에는 적용하지 않았다.'),'',
    '왜 비교했나',
    '이전11개 실패가 모두 시간 평균 때문이라는 증거는 없었다. 다만 앞뒤 평균은 구간 내부의 이동 순서를 잃을 수 있어, 그 정보가 실제로 도움이 되는지 별도로 시험했다.','',
    '무엇을 바꿨나',
    '①평균 대조: 앞뒤 관측을 평균으로 압축. ②순서 후보: 앞4개·현재·뒤4개 실제 관측을 그대로 보존. 두 조건은 마스크·시간·현재값·정규화·신경망 구조와 초기값이 같다. 양손과 다른 신체관계를 함께 유지한다.',
    '같은 7197개 학습 파라미터의 시간 합성곱+작은 양방향GRU 판정부다. 기존100단어 모델을 교체한 것이 아니라 몸 움직임/유지를 검사하는 격리된 진단이다.',
    'C6와는 신경망 구조와 관측구간 요약 방식도 다르다. 따라서 순서 정보만의 효과는 새 평균 대조와 순서 후보의 짝 비교로 판단하며, 저장 C6점수는 별도의 보존 기준이다.',
    '누락을 정상 좌표로 채우지 않았다. 영상 경계 밖 행은 관측불가다. 팔을 따라가는 관계의 분자와 기준팔길이, 신체 기준좌표를 보존했다.',
    '다만 GRU는 가려진 순간에도 추정값을 낼 수 있다. 입력 마스크가 있다는 사실만으로 자동 판단보류가 보장되지는 않으며, 기존 관측부족 판정은 별도로 유지해야 한다.','',
    '자료와 비교',
    '공개 FIT24영상·기록ID02~07, 한 기록ID를 제외하는6분할×3초기값×2조건=36개 새 학습. 원래 정답·손실가중치·1200단계·문턱을 고정했다. 기존 C6/C7/C8 완료 모델은 재실행하지 않았다.',
    '점수는 같은 영상의 화면조건3종·정역 반복을 포함한다. 새 사용자나 독립 최종평가 정확도가 아니다.','',
    '결과: 초기값42/43/44 순, C6기준 → 같은구조 평균대조 → 실제순서 후보',
    '실제 움직임/822: '+vec('C6',1,'before')+' → '+vec('mean_control',1,'before')+' → '+vec('C6',1,'after'),
    '자세 유지/522: '+vec('C6',3,'before')+' → '+vec('mean_control',3,'before')+' → '+vec('C6',3,'after'),
    '다른 동작 구분/1308: '+vec('C6',2,'before')+' → '+vec('mean_control',2,'before')+' → '+vec('C6',2,'after'),
    '팔따라 움직임/180: '+family('C6',4,'before')+' → '+family('mean_control',4,'before')+' → '+family('C6',4,'after'),
    '양손 상대높이 교차단서/294: '+family('C6',5,'before')+' → '+family('mean_control',5,'before')+' → '+family('C6',5,'after'),'',
    '짧은 변화와 경계',
    '기존 시작/끝 경계 범위 안의 직접 움직임 정답을 따로 집계했다. 경계에 있다는 이유로 정답을 지우거나 유지로 바꾸지 않았다. 합성 짧은 변화가 입력에서 보존되는지도 사전 검사했다. 이것은 실제 빠른 수어의 별도 정답 평가와는 다르다.']
for r in comp['boundary_subset']:
    if r['baseline']=='mean_control':
        m=r['boundary_movement'];lines.append(f"초기값{r['seed']} 경계 움직임: {m['before']} → {m['after']} / {m['count']}")
lines+=['','검산과 제한',f"사전{read(U/'PRETRAIN_VERIFICATION.json')['checks']}검사, 저장결과{ver['checks']}검사와 추가해시{checks}검사를 통과했다. 같은 Codex의 별도 산술검산이며 한국수어 전문가의 독립 정답 검수는 아니다.",
    '손가락 모양·의미상 순서·일부만 한 동작의 완료·설명 구간 자동연결 전체를 새로 평가한 것은 아니다. 양손 높이 교차 점수를 손가락 좌우 정체성의 정확도로 확대하지 않는다.',
    '기존 C·T3-F·100단어 학습/운영웹·보호477파일 보존. DEV·최종8·사용자·멘토자료 미사용. CPU2threads로 완료. 현재 실행 중 학습 없음.','',
    '다음: '+('이 부품의 동결 결과를 기존 설명구간 연결 경로에 연결하는 별도 비교 계약을 세워야 한다. 아직 운영에 적용하지 않는다.'if passed else'완료한36개를 반복하거나 초기값·문턱을 골라 적용하지 않는다. 새 손실의 관측·정답·모델 일반화를 먼저 구분하고 다음 계약을 검토한다.')]
(U/'LEARNING_RESULT_KO.txt').write_text('\n'.join(lines)+'\n','utf8')
write('GRAPH_PATCH.json',dict(patch_only=True,root_modified=False,node_id='C9_OBSERVED_SEQUENCE_CONTEXT',status='DIAGNOSTIC_PASS_NOT_DEPLOYED'if passed else'COMPARED_REWORK_NOT_ADOPTED',G2_status='REWORK',G2_pass=False,new_models=36,prior_models_rerun=False,gates=comp['gates'],evidence=['TRAINING_FROZEN.json','PRETRAIN_VERIFICATION.json','TRAINING_COMPLETE.json','COMPARISON.json','SAVED_VERIFICATION.json','LEARNING_RESULT_KO.txt'],next=lines[-1],training_running=False))
write('CLOSEOUT_LEARNING.json',dict(status='COMPLETE_FIXED_C9',diagnostic_pass=passed,trained36=True,hash_checks=checks,protected_files=len(c['protected_pins']),all_hashes_unchanged=True,training_running=False,model_replacement=False,root_documents_modified=False,report_sha256=sha(U/'LEARNING_RESULT_KO.txt')))
print('C9fixedcomparisonclosed',comp['status'])
