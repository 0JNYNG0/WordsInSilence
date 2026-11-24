from pathlib import Path
import sys,json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    v=read(U/'REVIEW_VERIFIED.json');assert v['status']=='SOURCE_EVIDENCE_CONTRACT_REVIEW_COMPLETE'
    c=read(U/'EVIDENCE_CONTRACT.json')
    for p,h in c['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    text='''이번 학습 비교와 원인 확인

완료한 실제 작업
1. 새 직접정답만 배우는 조건과 기존 판단도 약하게 유지하는 조건을 각각80epoch 학습했다. 새모델만176개발비교를 했고 저장 배열과2115개 평가 정답의 동일성을 별도로 검산했다.
2. 기존 판단 유지 조건은 같은 학습 대조보다 정상 주요시점8개를 살렸다(572→580/615). 짧은 역순2/3, 정상순서20/21을 유지했다. 그러나 이전582 및 내용/역할 교차583보다 낮고 팔 이동 가족의 손실이 남아 미채택이다. 전체 정확도나 웹 개선으로 보고하지 않는다.
3. 고정 후보를 학습 원본36+웹캠시야36에 한 번 진단했다. 원본 역할정답1238/1238, 웹캠시야1066/1066이고 주요814/706을 제외한 사례0이었다. 기존 학습자료를 못 외운 문제가 아니라 개발자료의 사람/경계로 넘어갈 때의 문제다. 같은 자료의 학습 횟수나 KL계수를 계속 탐색할 근거는 약하다.
4. 별도 에이전트와 사건 정의/출처를 검토했다. 공개 FIT의4단어18구간, 연속248원본 프레임을 준비하고 전부 확인했다. 원본영상 해시·프레임·시각은 별도 재디코드로 검산했다. parent는 팔 이동9구간125프레임, 에이전트는 다른3표현9구간123프레임을 확인했다.

무엇을 확인했나
손을 닫은 모양을 유지하면서 팔 방향을 바꾸거나, 엄지를 세운 모양을 유지하면서 양손을 번갈아 움직이는 등 모양/관계와 이동이 함께 존재한다. 하나를 보았다고 다른 관측을 없애면 안 된다.
팔을 따라 이동하는 구간과 손목 근처의 모습은 같은 출처 동작 문구에 연결돼 있다. 다른 표현에서도 하나의 출처 문구에 여러 로컬 단계가 연결된다. 출처가 요구하지 않는 정지나 별도 필수 행동을 로컬 단계 수에서 만들어서는 안 된다.
다만 물리적 정보의 공존이 기존 사건 ID 자체가 틀렸다는 증거는 아니다. 정확히 한 프레임으로 나누는 소속 경계와 동시성은 확정하지 않았다. 개발 정답을 변경하거나 실패 점수를 성공으로 바꾸지 않았다.

실제로 준비된 결과물
EVIDENCE_CONTRACT.json: 손모양/관계/움직임 관측, 설명과의 사건 연결, 피드백 주장 조건을 구분하는 공통 자료 구조.8개 공유 출처 절을 추적했다. 단계 번호 차이만으로 수어 오류를 선언하지 않고 원본시간·짧은 반대근거·누락을 보존한다.
PARENT_TRAVERSAL_REVIEW.json 및 AGENT_CONTROL_REVIEW.json: 각 원본 시간대의 시각 관측과 불확실성을 기록했다. 아직 새 조밀한 학습정답으로 자동 변환하지 않았다.
MEDIA_VERIFIED.json/REVIEW_VERIFIED.json: 자료 연결 검산. 별도Codex 검토이며 독립 사람/수어전문가 정답이나 새 모델 성능을 증명하지 않는다.

다음 한 작업
기존에 직접 확인한 손모양·신체관계·움직임 정답들을 이 공통 구조로 연결하고, 각 출처 행동에 학습 가능한 근거와 빠진 근거를 표로 감사한다. 미주석/가림은 오답 또는 부재로 바꾸지 않는다. 동일 자료의 역할부만 다시 학습하지 않는다.
충분한 정답이 확인되면 같은 SignRep 기반 신경망에 관측별 출력과 설명 행동의 발생/연결 출력을 분리해 지도학습한다. 배타적인 단계번호를 맞히는 결과와 실제 설명 행동의 관측 결과는 별도 지표로 유지한다. 단어별 예외나 정답순서 강제는 쓰지 않는다.
이후 공개 정상/속도/짧은 역순/누락 비교, 최종 고정 후 미사용자료의 출처·중복 감사와 평가, 실제 전체영상 데모의 문구·원래시간·전체 처리시간 검증까지 해야 한다. 새 평가 점수로 규칙을 맞추지 않는다.

보존과 현재 한계
C·T3-F 및 기존웹/데모는 변경하지 않았다. 사용자·멘토 촬영과 잠긴 H14/15 자료를 학습·튜닝·선택에 쓰지 않았다. 새HTML은 만들지 않았다. 현재 실행 중인 학습/평가 프로세스는 없다. 전체 목표는 계속 진행 중이다.
이번 확인으로 중요한 병목을 좁혔지만 아직 피드백 전체가 개선됐다고 입증한 것은 아니다. 실제 데모와 새로운 공개 수행자료 검증을 건너뛰지 않는다.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
    plan=dict(next_single_action='Audit and compile existing directly reviewed predicate supervision and source-event support, without re-running role-only training.',
        required_input_fields=['clip/recorded-ID split','native frame/time','feature value and mask','visible versus temporal-only provenance','source clause pointer/hash','event binding and boundary uncertainty'],
        must_not=['infer absence from unlabeled','force a unique event boundary from sparse witnesses','turn local hold into mandatory pause','discard short contrary motion','update old DEV labels from candidate errors','use private or locked final footage'],
        next_training_preconditions=['Every trained predicate has direct or explicitly scoped evidence','source clause connection reviewed','positive/negative/unknown masks audited per person/word','fixed matched baseline/seed/update scope and test contract'],
        validation=['Keep old exclusive-family results separate','evaluate observed predicates and source-event connection','normal speed/rest/cross preservation','short removal/reversal and image-gap safeguards','locked new-person evidence after duplicate/use-history audit','actual runtime guide/native-time feedback and full latency'],
        whole_goal_complete=False)
    (U/'NEXT_PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'));import sync_flow;sync_flow.U=U
    sync_flow.sync('관측과 설명 행동의 근거 계약 검토 완료',
        'retention2학습176비교로 대조572→580 주요활용이나 이전582/583미달·미채택. FIT72진단은 역할1238/1238·1066/1066,주요제외0이라동일역할학습반복중단. 공개FIT4단어18구간248원본프레임을parent/별도agent시각검토·미디어검산. 모양/관계와이동공존 및같은출처절의여러로컬단계연결확인. 사건ID오류/정밀경계/필수멈춤은확정안함. EVIDENCE_CONTRACT준비,기존정답/모델/웹보존·전체goal active.',
        'NEXT_PLAN.json대로 기존직접검토 손모양/신체관계/움직임의값·마스크·native시간·출처절·사건연결을감사/편성. 미주석을부재로만들거나공유절만으로사건병합금지. 필요한감독근거확인후관측별출력과설명행동연결을분리한신경망고정비교;역할epoch/KL만재탐색금지. 완료retention176/FIT72/교차176/비주요440 재실행금지. 현재실행프로세스없음. H14·15/개인/멘토/C/T3F/기존웹보존. 실제새자료/전체데모완료전goal완료금지.',
        'SOURCE_EVENT_EVIDENCE_CONTRACT_REVIEWED',None)
    print('report and authoritative handoff synchronized; goal remains incomplete')
if __name__=='__main__':main()
