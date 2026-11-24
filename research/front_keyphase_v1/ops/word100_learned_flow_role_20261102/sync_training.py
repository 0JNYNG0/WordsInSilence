from sync_flow import *
if __name__=='__main__':
    assert read(U/'TRAIN_PROGRESS.json')['status']=='TRAINING'
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
    p.write_text(p.read_text(encoding='utf8').replace('学習基準 흐름','학습한 기준 흐름'),encoding='utf8')
    sync('학습한 기준 흐름 연결 판정부 학습 중',
        '공개FIT252/DEV88의기존학습prototype DTW340정렬준비완료. 모든query/기준시점·누락보존,정규화는FIT만. 정렬중앙약0.015초(전체처리시간아님). 기존 sparse과정정답만 사용하고 DTW구간힌트는정답으로승격하지않음. L초기 가중치/같은80epoch조건에서role층에만흐름위치·거리·가이드·압축보조6값을연결해1후보학습. actor동결·zero-init기존대조동일·흐름변경시관측불변검증. 새성능미확인.',
        'train_flow.py session27309 실제세션/TRAIN_PROGRESS확인후중복금지. 완료후 evaluate_flow.py 새88비교한번,verify_flow.py저장배열검산. V role_only 완료결과를같은조건대조로재사용(재학습/재추론없음). V2학습176/L/J/K/F/E/D/U완료재실행금지. 기준흐름힌트가실제짧은역순/누락을숨기면채택금지. 원판독589 exact·역할제외<9·정상순서>6·짧은역순>=2·대기>=96 등고정. 전체목표active/기존웹미적용.',
        'LEARNED_FLOW_ROLE_TRAINING',27309)
