"""Keep canonical handoff aligned with this approved100-word run."""
import argparse
from datetime import datetime
import shutil
from common import U,BASE,ROOT,OUT,RUN,read,write

def main(stage):
    now=datetime.now().astimezone().isoformat(); f=read(U/'FROZEN_EXPERIMENT.json')
    paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',BASE/'ops/STATE.json']
    if not (U/'before').exists():
        (U/'before').mkdir()
        for p in paths:shutil.copy2(p,U/'before'/p.name)
    progress=read(OUT/'progress.json') if (OUT/'progress.json').exists() else {}
    run_progress=read(RUN/'progress.json') if (RUN/'progress.json').exists() else {}
    info=dict(status=stage,at_kst=now,unit='ops/'+U.name,authorization='2026-10-13 사용자100단어진행승인',
        manifest='ops/'+U.name+'/MANIFEST.json',contract='ops/'+U.name+'/FROZEN_EXPERIMENT.json',
        cache=str(OUT),run=str(RUN),classes=100,training_rows=899,development_rows=300,
        implementation='Codex local; existing production MediaPipe/MaskedSignTCN reused',
        constraints='학습01~09/검증11~13;14~18과10 배열읽기0;사용자/NIKL/KSL0;운영모델/Git변경0',
        preparation=progress,training=run_progress,
        checks={'first6':'INPUT_CHECK_FIRST6.json','throwaway4step_timing':'SMOKE.json'},
        caveat='개발자료로epoch선택후같은개발성능보고;미사용최종시험/새사용자수어정확성/기존93개대비향상주장아님')
    state=read(paths[2]);state['updated_date_kst']=now;state['current_stage']='WORD100_'+stage
    state['word100_training_20261013']=info
    state['next_single_action']='진행중100단어추출중복금지. 1199입력검산→동결whole/whole_context학습→300개개발점수별도검산→결과보고. 보류최종시험/웹기본교체없음.'
    write(paths[2],state);write(U/'STATUS.json',info)
    context=paths[0].read_text('utf-8-sig')
    def replace_top(text,num,tag,block):
        start=text.index('\n',text.index(f'## {num}. '))+1
        if tag in text[start:start+150]:
            end=text.index('\n### ',text.index(tag,start)+len(tag))
            return text[:start]+'\n'+block+'\n'+text[end:]
        return text[:start]+'\n'+block+'\n'+text[start:]
    title8='### 2026-10-13 승인된100단어 MediaPipe·TCN 학습/검증'
    block8=f'''{title8}
사용자“100개단어에대해서진행”승인. `ops/{U.name}` FROZEN_EXPERIMENT/STATUS. 상태{stage},갱신{now}.
전단계계획의해시순100단어그대로고정;학습899/검증300,개발표기충돌0. 새MaskBatchNorm TCN280/width32/100출력,기존구조재사용·랜덤초기화·AIHub-only.
whole기준/whole_context후보 각최대80epoch·patience12·dev CE최소checkpoint. AdamW .001/weight_decay .001/batch32/seed20261013 고정. 같은초기값·순서,차이는학습시간구간. 검증은whole만.
첫6관절입력별도재구성PASS,가중치버린학습4step·padding검산PASS. 추출캐시W별도·300개/2h묶음·1worker·재개지원. 원본ZIP/웹기본보존. 실제진행은W progress와run progress/PID부터확인.
독립적인최종시험은이번범위아님. 실험이진행됐다는것과성능향상은구분. 학습/검증종료후저장점수검산필수.
'''
    title9='### 2026-10-13 다음 — 승인100단어 학습·검증까지 이어받기'
    block9=f'''{title9}
`ops/{U.name}/STATUS.json`과W:`word100_inputs_20261013/progress.json`·`runs/word100_tcn_v1/progress.json`의실제프로세스확인후이어간다. 동일추출·학습중복금지.
미완료1199입력준비→verify_inputs.py→train.py(동결2조건)→verify_metrics.py→결과/문서갱신. MODEL 기본교체/14~18시험/NIKL/KSL/사용자학습없음.
이승인은직전300개입력준비만/학습금지범위를100단어학습·개발검증까지확대한최신지시다. 최초오류보존·소스/분할동결유지,점수보고계수탐색/단어교체금지.
'''
    context=replace_top(context,8,title8,block8);context=replace_top(context,9,title9,block9);paths[0].write_text(context,'utf-8')
    handoff=f'''# WordsInSilence 최신 인수인계
갱신 {now}. AGENTS/CONTEXT 최신§3·§8·§9/WORKFLOW 읽기.

## 현재 승인 목표 —100단어 학습·개발검증 완료
사용자 “오케이이해했어. 그럼100개단어에대해서진행하면될것같아.” 오늘결과·10/15발표목표.
`ops/{U.name}` FROZEN_EXPERIMENT/STATUS/MANIFEST. 상태 {stage}.
학습899(01~09)/검증300(11~13),해시순100코드고정·개발표기충돌0. 14~18/10미사용,사용자·NIKL·KSL사용금지. 범위축소로점수향상주장금지.
기존MediaPipe→75관측→280채널,기존MaskedSignTCN width32/4blocks 랜덤초기화100클래스. 소스/운영모델변경0.
학습whole vs전체/핵심±.30s교대 각1회,seed20261013/AdamW.001/wd.001/batch32/max80/patience12/dev CE최소epoch고정. 검증whole입력만. 개발선택결과이지미사용최종시험아님.

## 실행·재개
캐시 {OUT} / progress.json / records/*.json. 모델 {RUN} / progress.json.
실제현재PID·로그확인후중복실행금지. prepare.py는검산된기록해시확인후건너뜀,한번300개또는2h. 원본ZIP보존,임시선택F영상만개별삭제.
첫6/851프레임독립정규화·시각재구성PASS(INPUT_CHECK_FIRST6). smoke4step은학습6영상만/폐기가중치,개발읽기0,100출력·padding PASS. 처음측정약.045초/step,전체실행확정시간아님.
다음:나머지1199준비→verify_inputs.py 전체→train.py 동결2조건→verify_metrics.py→결과문서. 완료된records재추출금지;입력/학습소스pin변경시먼저중단사유확인.
학습중단시last.pt보존,임의재시작금지·동일context검증복구만. 첫실패FIRST_FAILURE 보존. 최종성적보고계수/seed/어휘다시선택금지.

## 직전 완료 —반복금지
word_label_time_contract_20261013:2494메타데이터후보22388학습행(2436×9+58×8),의미충돌506보류.8영상수형시각검토,원본라벨변경/합치기없음. 전체타임라인보존·core/context표·예산검산.
word_pilot12_inputs_20261013:12/1444 MediaPipe·RGB입력검산. 이번100범위와겹치는호환캐시1개재사용.
word_dataset_plan_20261013:33ZIP1.181TB/270000영상/F54000/3000코드×18ID×5시점. 전부디코딩완료아님.
후속전체확장시관절캐시재사용가능,클래스출력확장과기존+새자료학습필요. 운영TCN832와93클래스결과보존.
'''
    paths[1].write_text(handoff,'utf-8');print(stage)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');a=p.parse_args();main(a.stage)
