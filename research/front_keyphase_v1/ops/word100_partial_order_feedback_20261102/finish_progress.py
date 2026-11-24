from pathlib import Path
import sys,json,hashlib
Z=Path(__file__).resolve().parent;O=Z.parent;ROOT=Z.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (Z/'FINAL.json').exists()
    v=read(Z/'RUNTIME_RESULT_VERIFIED.json');r=read(Z/'public_runtime_probe/PARTIAL_ORDER_FEEDBACK.json')
    assert v['status']=='PASS_INDEPENDENT_SAVED_RUNTIME_ARITHMETIC'
    assert v['existing_source_feedback_fields_preserved']==47 and v['completion_unknown']
    report='''현재 목표는 여러 사람의 공개 한국수어 영상을 배운 신경망이 새 영상과 목표 표현을 받아, 손모양·팔과 손의 관계·움직임·순서를 설명과 실제 시간에 연결해 알려주는 작은 데모입니다.

진행 정도
- 자료와 학습: 4개 표현에서 공개 기록 ID01~09를 학습에, ID11~13을 개발 비교에 분리했습니다. 9개 기록 ID가 9명의 다른 사람임을 외형으로 증명한 것은 아닙니다.
- 신경망: SignRep을 포함한 특징 흐름에 지도학습과 일부 가중치 미세조정을 했습니다. 손모양·관계·일부 이동을 실제로 출력합니다.
- 시간 정렬: 사용자 입력을 학습한 기준 흐름 하나에 DTW로 맞추는 경로를 만들었습니다. 실행 때 참고 영상 9개를 각각 비교하는 방식이 아닙니다.
- 실제 데모: 설명 문구와 원본 시간에 일부 특징을 연결하는 연구용8808이 있습니다. 8804는 보존된 이전 데모입니다.
- 순서: 부분적인 앞뒤 관계를 관측하는 연결부를 이번에 추가 검증했습니다. 새 연결부는 아직 기존 웹 기본 기능으로 적용하지 않았습니다.
- 남은 핵심: 중요한 구간의 누락·반복·전체 순서, 정상 변이에 대한 잘못된 경고, 새 사람에 대한 신뢰성, 전체 처리시간입니다.

이번에 이어서 한 일
1. 두 순서 판정부 학습과 비교를 완료했습니다. 기존 관측을 유지하는 조건은 개발 관계103개 중82개를 맞는 방향으로 확정했고, 이를 분리한 새 후보는42개였습니다. 나머지는 보류이며 확정한 오판은 양쪽0개였습니다. 새 후보는 개선이 없어 적용하지 않았습니다.
2. 103개는 독립된 영상103개가 아닙니다. 15개 원래 앞뒤 관계에 속도/관측 누락/역순 등 변형을 포함한 반복 개발 평가입니다. 수어 단어 인식률도, 전체 피드백 정확도도 아닙니다.
3. 새 후보는 서로 다른 두 동작의 근거를 같은 시간에 겹쳐 잡았습니다. 같은 문제가 학습 자료에도 있었으므로 같은 학습을 무작정 더 돌리지 않았습니다.
4. 이미 관측 가능한 부분 순서를 보존하는 연결부를 만들었습니다. 저장 관계110개와 누락·겹침·짧은 역순 등 표시 검사를 통과했습니다. 영상 일부가 잘렸어도 남아 있는 A→B만 보고 전체가 완성됐다고 말하지 않습니다.
5. 새 입력과 학습 때의 전처리/시간 정렬 계약이 같은지 검사한 뒤, 기존 공개 '잘하다' 분석값에 새 순서 판독을1회 실행했습니다. 2.00~2.47초의 팔을 따라 이동하는 근거와2.50~2.80초의 손목 부근 근거를 앞뒤로 연결했습니다. 문장의 모든 세부 특징을 확인했다는 뜻은 아닙니다.
6. 별도 에이전트가 결과를 검산했습니다. 기존47개 결과 필드,119개 원본 시점/누락 정보,76개 고정 파일이 그대로 유지됐습니다. 이 검산은 한국수어 전문가의 독립적인 정오 판정과 다릅니다.

속도
앞선 실제 공개 영상 전체 분석은1회9.8009초였습니다. 이번 순서 판독 자체는0.02094초였지만 이미 추출한 입력을 사용했으므로 전체 기다리는 시간이0.02초라는 뜻이 아닙니다. 관절/RGB 추출을 포함한 지연 개선은 아직 남아 있습니다.

현재 판단
작은 범위에서 배운 특징을 새 영상에 적용하는 구조는 구현했습니다. 이제 중요한 차이를 안정적으로 구별하여 실제 교정 피드백을 제공하는 수준으로 검증·보완 중입니다. 전체 완성도를 신뢰할 만한 백분율로 환산할 근거는 아직 없습니다.
TCN의 C·T3-F 인식 모델은 그대로 유지했습니다. 이번 작업은 별도 피드백 신경망과 그 결과 연결이며, TCN 인식률 향상 작업으로 보고하지 않습니다.
사용자·멘토 촬영을 학습·튜닝·후보 선택에 사용하지 않았습니다. 새 외부 모델 다운로드, 추가 원본 촬영, 운영 교체는 없습니다.

다음 한 작업
검증한 부분 순서 연결부를 별도 연구용 실제 분석 경로와 화면에 연결합니다. 관측 순서·미확인 구간·원본 시간·기존 특징이 함께 보이게 하고, 공개 정상/역순/관측 누락/동작 일부 삭제에서 표시와 입력 일치를 확인합니다. 기존8804/8808은 보존합니다.
그 이후 누락·반복 판정에 필요한 근거를 보완하고, 전체 관측 처리시간을 줄입니다. 단순히 결과가 좋아 보이게 임계값을 바꾸거나 짧은 동작 프레임을 버리지 않습니다.
'''
    (Z/'RESULT_KO.txt').write_text(report,encoding='utf8')
    write(Z/'FINAL.json',dict(status='PARTIAL_ORDER_ADAPTER_VERIFIED_NOT_WEB_DEPLOYED',whole_goal_complete=False,
        new_training=0,new_tied_head_invocations=1,new_current_state_invocations=1,new_backbone_extraction=0,
        old_fields_preserved=47,frames=119,saved_display_pairs=110,source='public recorded material only',
        independent_saved_arithmetic=True,independent_human_expert_truth=False,accuracy_gain_claim=False,
        total_latency_gain_claim=False,web_default_changed=False,old_models_changed=False,
        report=str(Z/'RESULT_KO.txt'),verification=str(Z/'RUNTIME_RESULT_VERIFIED.json'),
        evidence={str(p):sha(p) for p in [Z/'INPUT_COMPATIBILITY_VERIFIED.json',Z/'DISPLAY_CONTRACT_VERIFIED.json',Z/'RUNTIME_CODE_REVIEW.json',Z/'RUNTIME_RESULT_VERIFIED.json']}))
    summary='Z 부분 순서 연결부 검증 완료'+': 저장110관계표시/입력130·RGB769·flow6·DTWpath exact/실제공개잘하다119프레임새head1회통과. 2.00~2.47초팔이동근거→2.50~2.80초손목부근근거. 별도검산47기존필드/시간·누락/76pin보존. head0.02094초는캐시입력부분시간이며전체9.8009초기록개선아님. Y direct후보82→42/103확정회귀로미채택,실패보존. 연구후향tied부분순서연결만/웹8804·8808변경없음. 전체누락·반복·교육정오·새사람검증/전체속도미완료,goalactive.'
    nxt='Z/FINAL·RESULT·RUNTIME_RESULT_VERIFIED 확인후 검증된부분순서adapter를새격리실제demo경로에연결. 8804/8808유지,새학습없이기존source/point/RGBmotion보존·원시간·unknown표시. 새입력/base job결합을일반capture계약으로검증하되개인/멘토촬영접근·선택금지/카메라자동호출금지. saved공개정상/역순/gap/cut에서브라우저표시·시간·기존결과보존검증,완료학습/평가와fd785d새head재실행금지. wholecompletion/누락없음/필수반복판정은이번부분관측으로승격금지. 예전자료의80epoch/threshold/seed반복금지. 전체goal미완료.'
    sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'));import sync_flow
    sync_flow.U=Z;sync_flow.sync('부분 동작 순서의 실제 입력 연결·보존 검산 완료',summary,nxt,'PARTIAL_ORDER_ADAPTER_VERIFIED_NOT_WEB_DEPLOYED',dict(training=False,evaluating=False,worker=None))
    state=read(O/'STATE.json');y='word100_direct_occurrence_order_20261102'
    state[y].update(status='COMPLETE_DIRECT_NOT_ADOPTED',active=None,report=str(O/y/'RESULT_KO.txt'))
    write(O/'STATE.json',state)
    old=read(O/y/'CURRENT_EXECUTION.json');old.update(status='COMPLETE_DIRECT_NOT_ADOPTED',active=None,next='후속 Z 부분 순서 연결부 검증 완료. 최신 root §9 참고.');write(O/y/'CURRENT_EXECUTION.json',old)
    assert read(O/'STATE.json')['current_stage']=='PARTIAL_ORDER_ADAPTER_VERIFIED_NOT_WEB_DEPLOYED'
    assert summary in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf8') and summary in (ROOT/'NEW_CHAT_HANDOFF.md').read_text('utf8')
    print('Progress and handoff synchronized; overall goal remains active.')

if __name__=='__main__':main()
