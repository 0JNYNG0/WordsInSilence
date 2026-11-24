from pathlib import Path
import sys,json
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'));import sync_flow
sync_flow.U=U
finished=[p.stem for p in U.glob('REVIEW_*.json')]
sync_flow.sync('주요 움직임의 직접 감독 자료 준비·공개45과정 검토',
 '직전goalturn은AA8809실제연결/373검산완료로progress. AB기존주석감사결과sparse850/81motion/630order는충분수행정답이아님확인. 새모델탐색대신기존FIT36(등록ID1~9)45physicalprocess의원프레임1354개/90page를W파생폴더준비완료(exec36760 terminal). process관계presence0/1/unknown와motioninterval을분리/겹침·통과·몸흔들림허용. 현재직접RGB검토완료파일:'+','.join(finished)+'. 0739 ID08목·어깨변이는canonicaltorso불확실과실제연속이동을분리보존. 새학습/추론0/원본·C/T3F·8804/8808/8809유지/전체goalactive.',
 'retention_direction_review의REVIEW_1567(18query/36page) 실제상태확인후중복검토금지. 네REVIEW가준비되면verify_process_labels.py로원본/페이지/시간/schema검산해PROCESS_TARGETS작성. 실제phaseabsence와시간역할/interval-vs-frame혼동금지. 이후고정FIT source-linked setup/progress/terminal지지로prefix/suffix/양끝보존middle-splice 대조를실제로검토하며 physicalcut=언어오류·terminalpresence=완료·가림=부재금지. 학습정답과입력같은센서조건/시간/원본hash를묶고 rawRGB편집후재추출하여잘린프레임context누출방지. 기존P native입력은mask/ROI가있고L seated입력은전체crop검출;무조건currentfullframe같다하지말고조건명시/일관입력선택. 기존readout·shape·motion불변,단일새감독head비교계약후학습. AA새publicjob/완료old추론·평가재실행금지. HTML추가없음.',
 'DIRECT_PROCESS_SUPERVISION_REVIEWING',dict(training=None,preparation='complete',review_agent='/root/retention_direction_review',demo_port=8809))
print('AB process review checkpoint synchronized')
