from pathlib import Path
import sys,json
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=U
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    if '--running' in sys.argv:
        sync_flow.sync('추가 과정 정답·시간 신경망 네 조건 학습 시작',
            '추가172시점/파생정답1개수정의독립overlay와1011개변형매핑변경검산완료. 공개FIT252입력에서추가정답유무×복사된동작시계열GRU미세조정유무4조건80epoch를동일초기모델로학습중. 이미지/SignRep/보호관측부품과C/T3F/웹고정. 성능미확인,역할만반복하는비교와구분.',
            'train.py session37633 및TRAIN_PROGRESS 실제상태확인,중복실행금지. 완료후evaluate.py 새4모델×88=352비교한번,verify.py저장배열검산. primary new_adapt사전고정,다른3조건은원인분리대조. DEV11~13은노출개발자료,H14/15/사용자/멘토금지. 완료후결과/통과판정/다음원인분리기록. 전체goal미완료.',
            'EVENT_TEMPORAL_FACTORIAL_TRAINING',37633)
        return
    v=read(U/'VERIFIED.json');p=v['conditions']['new_adapt'];ev=read(U/'EVALUATION.json')
    lines=['공개 동작 과정 정답 보강 × 시간 신경망 미세조정 네 조건 비교',
        '', '목표: 준비·복귀와 주요 동작을 나누면서, 주요 동작의 내용과 순서를 더 정확히 판단한다.',
        '12개 공개 FIT 영상에서 172개 시점을 직접 확인하고, 기존 파생 정답 1개의 오류를 별도 정정 기록으로 연결했다.',
        '주요 동작 108 / 준비·복귀 60 / 관측된 대기 4. 단계가 애매한 주요 시점은 단계 번호 정답을 비워 두었다.',
        '원본·기존 결과는 바꾸지 않았다. 같은 Codex 영상 검토이며 독립 수어 전문가의 언어 정오 검증은 아니다.',
        '', '같은 공개 FIT252입력, 같은 초기모델/80epoch/seed42. 새 정답 유무와 동작시간 GRU+delta 미세조정 유무를 분리했다.',
        '기존 SignRep 입력특징과 이미지 추출기, 독립 손모양·신체·움직임 관측기는 고정했다.',
        '학습에 쓰지 않은 기록ID11~13의 기존 개발88조건을 새 모델마다 한번 비교했다. 반복 사용한 개발자료이며 새 최종 시험이 아니다.',
        '', '조건 | 주요 행동 자체 정답/615 | 주요로 남긴 정답/615 | 정상 순서/21 | 짧은 역순 탐지/3 | 정상 순서 오경고',
        '이전 Flow | 589 | 582 | 6 | 1 | 0']
    names={'old_frozen':'기존 정답·시간부 고정','new_frozen':'추가 정답·시간부 고정','old_adapt':'기존 정답·시간부 학습','new_adapt':'추가 정답·시간부 학습(사전지정 후보)'}
    for tag,z in v['conditions'].items():lines.append(f"{names[tag]} | {z['raw_correct']} | {z['used_correct']} | {z['normal_order']} | {z['short_reverse']} | {z['false_order']}")
    lines += ['', '이 수치는 단어 인식률이나 모든 손가락 정확도가 아니라, 검토한 주요 시점의 행동 종류와 순서 부품 시험이다.',
        '누락 프레임·원래 시간축·독립 관측 확률이 유지되는지 352개 저장 배열을 별도 코드로 검산했다.',
        f"판정부 계산 중앙값 {p['head_median_seconds']:.4f}초. 영상 전처리를 포함한 전체 대기시간은 이번에 재측정하지 않았다.",
        '', '통과 기준:']+[f"{k}: {'PASS' if b else 'FAIL'}" for k,b in v['gates'].items()]
    accepted=all(v['gates'].values())
    lines += ['', '판정: '+('부품 통과. 전체 피드백 연결과 전체 지연 검증은 남아 있다.' if accepted else '사전 통과 기준 미달. 기존 웹·운영 모델에 적용하지 않았다.'),
        'C·T3-F와 기존 웹 보존. 사용자·멘토 자료 학습/튜닝/후보 선택 없음. 새 HTML 없음.',
        '다음: 저장된 예측에서 실제 행동 판독 오류, 준비·복귀 제외 오류, 순서 판정 가정을 분리해 원인을 확인한다. 완료 모델을 다시 학습·추론하지 않는다.',
        '전체 목표 미완료: 새로운 촬영의 중요한 차이와 허용 차이를 가이드 문구/시간대에 신뢰성 있게 연결하는 데모의 검증이 남아 있다.']
    (U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    summary=f"공개FIT추가172시점·정정1개를연결한4조건80epoch학습/352개발비교/저장배열검산완료. primary new_adapt의주요판독 {p['raw_correct']}/615,주요활용 {p['used_correct']},정상순서 {p['normal_order']}/21,짧은역순 {p['short_reverse']}/3. " + ('부품통과·전체연결전.' if accepted else '통과기준미달·미채택.')+' C/T3F/기존웹/보호관측/시간축보존,HTML없음. 추가정답효과와시간부학습효과를분리했으며전체goal미완료.'
    sync_flow.sync('추가 과정 정답·시간 신경망 네 조건 비교 완료',summary,
        '완료4학습352비교재실행금지. VERIFIED/저장예측에서새정답효과·시계열학습효과·상호작용을비교하고,원행동판독/과정역할/순서판정규칙의오류를분리. 개선됐다면가이드시간대피드백전체연결/전체지연검증으로,미통과면근거있는최소후속작업을정의. 반복DEV를독립검증으로포장하거나문턱튜닝/강제정답순서금지. H14/15/사용자/멘토/C/T3F/기존웹보존. 현재학습/평가실행없음.',
        v['status'],None)
    print(summary)

if __name__=='__main__':main()
