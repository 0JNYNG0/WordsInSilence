from pathlib import Path
import json,sys
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    v=read(U/'VERIFIED.json');assert read(U/'SOURCES_VERIFIED.json')['status']=='UNCHANGED_DEVELOPMENT_LABELS_VERIFIED'
    assert read(U/'TRAIN_PROGRESS.json')['status']=='COMPLETE';a=read(U/'CAUSE_AUDIT.json')
    lines=['주요 동작을 읽는 지식 유지 학습 비교','',
        '무엇을 바꾸었나',
        '손동작 내용을 읽는 새 actor는 고정했다. 주요 동작/준비·복귀를 구분하는 역할부만 추가 학습했다. 두 조건 모두 같은 추가 정답, 같은 초기값, 같은 흐름 입력과80회 학습이다.',
        '차이는 미주석 공개 학습 구간에서 이전 역할부의 부드러운 확률을 약하게 유지하도록 했는지뿐이다. 이전 판단을 새 정답으로 승인하지 않았다. 직접 정답/영상누락/편집 경계에는 이 유지 신호를 쓰지 않았다.',
        '', '같은 개발자료 결과: 615개 확인된 주요 시점, 21개 순서지원 정상조건, 짧은 역순3개',
        '조건 | 원동작 판독 | 실제 주요 비교에 남은 정답 시점 | 정상순서 | 다른 확인된 문제 없는 순서 | 짧은역순',
        '이전 Flow(저장 결과) |589|582|19|5|1',
        '새 내용+이전 역할 교차(저장 결과; 별도 실시간모델 아님) |590|583|19|5|1']
    for tag,ko in [('new_labels_only','추가정답만'),('new_labels_with_retention','추가정답+기존판단 유지')]:
        z=v['conditions'][tag];lines.append(f"{ko} |{z['raw_correct']}|{z['used_correct']}|{z['normal_order']}|{z['clean_order']}|{z['short_reverse']}")
    lines += ['', '판정과 의미',
        '기존판단 유지 조건은 같은 학습 대조보다 맞은 주요시점8개를 더 살렸다(572→580). 짧은 역순2/3, 정상의 거짓역순 경고0, 역순을 정상순서라고 통과시킨 사례0을 유지했다.',
        '하지만 이전의 주요비교582와 새내용/이전역할583보다 낮고, 팔을 따라 이동하는 가족의 활용이73→68로 줄었다. 통과기준 미달이다. 역할 학습 횟수나 유지 가중치를 개발점수에 맞춰 더 조정하지 않았다.',
        '10개 정답 주요시점이 역할부에서 제외된다. 단어/사람/원본시점으로 접으면5곳이며, 속도변형의 반복 시점을 독립 사례로 부풀리지 않는다. 행동 전체가 사라졌다는 뜻과도 구분한다.',
        '정상순서20/21은 해당 상태가 시간순으로 발견됐다는 수치이며 단어인식률/완벽한수어/모든피드백정확도가 아니다. 다른 이미 확인한 문제까지 없는 것은7조건이다. 반복표현의 필수횟수는 아직 정답 규칙으로 확정하지 않았다.',
        '', '검증과 제한',
        '2조건×80epoch 학습, 새모델2×88=176비교, 저장 배열 별도 검산 및 이전 평가의2115개 정답/시간 대응 동일성을 확인했다. actor 가중치/원판독/시간/누락은 보존했다.',
        'FIT 공개 기록ID01~09의36원본/252입력만 학습. 개발ID11~13은 반복 노출된 개발자료이므로 독립 최종평가가 아니다. 같은Codex 별도 검산은 독립 수어전문가 평가가 아니다.',
        '판정부만의 중앙시간은 약0.0075초이며 영상 읽기·관절·RGB·SignRep 추출을 포함한 사용자 대기시간이 아니다. 이번에 전체속도를 새로 측정하지 않았다.',
        'C·T3-F/기존웹 보존, 개인·멘토/H14·15 사용0, 새HTML 없음. 전체목표 미완료.',
        '', '다음',
        '단순 역할학습 반복을 멈추고, 주요 시작·끝에서의 감독 범위와 역할/내용 계약을 확인한다. 현재판정부가 공개학습의 원본/웹캠시야72조건에서도 어떤 근거로 주요를 제외하는지 고정후보 진단으로 분리하고, 별도 코드 검토 결과와 함께 다음 한 변경을 정한다. 기존 순서·짧은근거를 무시하는 완화나 단어별 예외는 쓰지 않는다.']
    hidden=a['conditions']['new_labels_with_retention']['correct_main_excluded']
    unique={(p['word'],p['person'],p['source_frame']) for p in hidden};assert len(unique)==5
    (U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'));import sync_flow;sync_flow.U=U
    sync_flow.sync('주요 구간 지식 유지 두 조건 비교 완료',
        '두조건80epoch/176개발비교/별도배열·정답2115동일성 검산완료. 유지신호로 대조572→580 주요활용,정상순서20/21·짧은역순2/3 유지. 이전582/교차583미달·팔이동가족73→68이라미채택. 원actor590/615/시간/누락/C/T3F/웹보존. 전체goal미완료.',
        '완료retention2학습176/교차176/이전352/비주요440 재실행금지. 다음 고정primary의FIT 원본/웹캠시야72조건 진단과 별도리뷰로 주요 시작끝 감독/내용·역할 계약의 원인을 확인한다. 역할epoch/KL가중치 반복탐색금지. 실제원인에 필요한 한 변경만 동결후 진행. H14·15/개인/멘토/운영 보존.',
        'ROLE_RETENTION_VERIFIED_NOT_ADOPTED',None)
    print(v['status'])
if __name__=='__main__':main()
