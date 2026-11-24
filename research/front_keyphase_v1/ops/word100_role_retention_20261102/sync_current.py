from pathlib import Path
import sys
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=U
if __name__=='__main__':
    sync_flow.sync('主要 구간 지식 유지 두 조건 학습 시작',
        '교차결합176검산에서 새 역할부가 짧은역순1→2와 맞은주요활용583→578 손실의 공통원인임을 확인. 새actor590/615 고정, FIT252/9기록ID의 새 직접정답과 미주석22708시점의 약한 이전역할신호를 분리하여 두조건80epoch 학습. 신호는 정답이 아니며 직접정답·영상누락·편집경계 제외검산완료. 새성능미확인/웹미적용/전체goal미완료.',
        'train.py session19158 실제세션/TRAIN_PROGRESS 확인, 중복금지. 종료후 evaluate.py 새두모델×88=176 한번, verify.py 저장배열·배우지않는actor동일성 별도검산. 통과시 전체피드백 연결/전체처리시간 확인, 미달시 지표/원인과 재학습 필요성 재판정. 완료cross176/기존학습352/비주요440 재실행금지. 개인/멘토/H14·15/C/T3F/기존웹보존.',
        'ROLE_RETENTION_TRAINING',dict(session_id=19158,script=str(U/'train.py')))
