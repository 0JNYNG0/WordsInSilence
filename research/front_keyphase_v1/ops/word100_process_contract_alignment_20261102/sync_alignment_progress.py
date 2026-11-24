"""Root-only authoritative state synchronization; never rerun model work."""
from pathlib import Path
import json
import sys

U=Path(__file__).resolve().parent
O=U.parent
ROOT=U.parents[3]
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=U

title='과정 판정 실패 원인 확인·학습과 실행 조건 정렬 보완'
summary=(
 '전체 학습형 피드백 목표 active. 기존4표현 실제데모8809/C/T3F/8804/8808 보존. '
 'AB 완료 후보는 rawDEV68/75였으나 실제근거확인3/30·자동1/15, 부분오확인1/45로미채택; 완료추론재실행0. '
 '저장원인감사에서 global확신·point출발도착·local netmotion 과제불일치가복합적임확인. '
 'FIT directpoint41차원/37행(고유25), explicitmotion7구간과 DEVpoint12차원/12행(고유10),motion12구간이정답소비에서누락된것을 '
 'AC v2로lossless추가·별도26009검산PASS. 기존unknown·process정답불변. '
 '실제full-native48기존입력60질의에기존직접정답연결: FIT44yes1unknown/DEV15yes; sparse900/motion217, 별도10306검산PASS. '
 '루트원시간입력감사로bounded vs fullnative 같은프레임의mask/RGB 차이확인; 단독원인·검출정확도증명아님. '
 '새후보는동일47980param/seed42/final80epochs로v2·runtime문맥·직접nominal 과정감독을함께맞추는단일비교준비중. '
 'FIT frozen첫candidate45개직접검토·RGB실제지원시간범위·기존splice범위연결을정리중; global1을local에복사않음. '
 '기존netmotion와연결과정target분리,confidence/proposal기준낮추지않음. 새학습/평가아직0,새성능향상미확인. '
 '사용자/멘토/H14/15/ID16미사용,원본readonly/새download외부전송Git없음. 같은Codex검토·별도검산이며독립수어전문가정답아님.'
)
nxt=(
 'AC SCOPE/FOLLOWUP_PLAN/실제agent상태우선. predicate의process_data+traincode검토, egypt의fixed first45 nominal span준비및2523직접검토, '
 'retention의RGB_SUPPORT_MANIFEST/VERIFIED를중복없이이어받기. root1133와다른공개FIXED span을직접검토하고, '
 '기존explicit yes-chain positive와reviewed process0 splice support-contained negative만출처/시간검산하여nominal정답동결. '
 'RGB가nominal밖을보는보조context와직접판정대상을분리표시; 새exactspan의truth를overlap으로만들지않음. '
 'v2+runtime262FIT학습과같은기준eval357(297view+60auto)단일계약,oldgates·code/data/pins·별도검토후한번학습/평가. '
 'root evaluate_contract_candidate.py는새후보용아직미실행;trainfilename계약 ALIGNMENT_FROZEN/TRAIN_RESULT/EVALUATION. '
 '현재모델추출/학습없음. 이전AB80epoch/297raw/60auto,AApublicjob 재실행금지. '
 '큰파생W AC폴더,작은보고C. 전체목표완료아님; 정상보존/중요차이/정확시간/실제데모/속도남음.'
)

def main():
    checked=json.loads((U/'SUPERVISION_V2_SAVED_VERIFIED.json').read_text(encoding='utf8'))
    assert checked['status']=='PASS' and checked['total_checks']==26009
    active=dict(training=None,extraction=None,new_candidate_adopted=False,
                implementation_agent='predicate_contract_review',
                nominal_review_agent='egypt_source_binding_review',
                rgb_support_agent='retention_direction_review',demo_port_preserved=8809,
                previous_goal_turn='User clarification only; authoritative no-progress revalidated, then v2/runtime contract and implementation advanced')
    # sync_flow inserts a latest item once; refresh only the exact current title.
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf8')
    for heading,body in [('### 2026-11-02 '+title,summary),('### 2026-11-02 '+title+' 이어받기',nxt)]:
        needle=heading+'\n'
        if needle in text:
            start=text.index(needle)+len(needle)
            ends=[i for i in [text.find('\n### ',start),text.find('\n## ',start)] if i>=0]
            end=min(ends) if ends else len(text)
            text=text[:start]+'\n'+body+'\n\n'+text[end:]
    context.write_text(text,encoding='utf8')
    sync_flow.sync(title,summary,nxt,'PROCESS_CONTRACT_ALIGNMENT_PREPARATION',active)
    (U/'PROGRESS_KO.txt').write_text(
      '목표: 미리 배운 신경망이 새 영상에서 중요한 행동과 순서를 파악하고 설명 문구·시간대와 연결해 피드백한다.\n\n'
      '기존 4표현 실제 데모는 유지한다. 새 과정 판정부의 실패를 검사하니 정답 전달 누락, 학습·실행 영상 범위 차이, '
      '단순 움직임과 연결된 동작 과정을 같은 것으로 검사한 문제가 함께 있었다.\n\n'
      '완료: 직접 확인된 정답 누락을 보완하고 별도 검산했다. 실제 검사 때 사용하는 공개 영상 범위에도 출처·시간이 맞는 정답을 연결했다.\n'
      '진행: 작은 동작 구간 자체의 정답을 확인하고, 앞뒤 장면을 참고한 범위를 따로 기록한다. 이후 같은 모델 규모와 평가 기준으로 한 번 학습·비교한다.\n'
      '아직 새 후보의 성능 개선은 확인되지 않았다. 기존 모델과 웹을 바꾸지 않았고 사용자·멘토 촬영은 사용하지 않았다. 전체 목표는 진행 중이다.\n',encoding='utf8')
    print('Current context/state/handoff synchronized; no models executed')

if __name__=='__main__':main()
