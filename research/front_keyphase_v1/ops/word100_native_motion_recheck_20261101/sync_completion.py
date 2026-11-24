from pathlib import Path
import json,datetime
N=Path(__file__).resolve().parent;O=N.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    c=read(N/'COMPLETION.json');assert c['status']=='VERIFICATION_COMPLETE_TRAINING_NOT_READY'
    assert (N/'RESULT_KO.txt').exists() and (N/'review/FULL24_OUTCOME_VERIFICATION.json').exists()
    status='NATIVE_MOTION_VERIFICATION_COMPLETE_TRAINING_NOT_READY'
    now=datetime.datetime.now().astimezone().isoformat()
    state=read(O/'STATE.json');state['current_stage']=status;state['updated_date_kst']=now
    state['next_single_action']=c['next_single_action'];state[N.name]=c
    (O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    next_ko='추가학습전 별도 준비계약에서 기존FIT60영상·180창 전체의 정답 지원을 한 번에 확인한다. 원24FAIL·지원12는별도보존,새작은표본교체/기준완화X. 같은down/hold·13원프레임·unknown규칙과 표적별양/음성최소12창·6영상·4표현·3ID를유지하고,관측/ROI지원의가능최댓값부터확인해구조적미달이면추가판독도종료. 충분할때만전체판독·검산·고정학습. 이새준비계약/전수판독은아직실행안함.'
    text=(ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text(encoding='utf-8-sig')
    entry8='''### 2026-11-01 원본 재검토 완료 — 판독 개선, 학습용 반대 사례 부족

`word100_native_motion_recheck_20261101`: 재부팅후완료추출재사용,원9+남은15같은24창312frame native판독/검산완료. 왼손6애매함중5확정일치,N008양쪽unknown. 같은96항목: 이전77일치/13한쪽unknown/6양쪽unknown → native75일치/4한쪽unknown/17양쪽unknown,반대확정0. 더보수적인보류이며모델정확도개선아님. 원W24 gate중양손hold적격음성지원2개만FAIL,나머지준비검사PASS. 물리적hold음성N022는ROI밖,적격0. 별도FIT12창에선적격음성R4(3ID/2표현),L1(1ID/1표현)확인;원24분모/지원수혼합X. 새학습·모델추론0,HTML생성X. 기존C/T3F/운영32보호SHA유지,개인·멘토자료접근X. 원픽셀·저장산술별도Codex검산PASS(독립전문가정답아님). 근거COMPLETION.json/RESULT_KO.txt/full24_native/FULL24_AUDIT.json/review/FULL24_OUTCOME_VERIFICATION.json. 이번제한검증완료,조건부학습은준비기준미달로미실행.

'''
    entry9='### 2026-11-01 다음 단일 작업 — 학습 정답 지원을 전체 FIT에서 한 번에 점검하는 준비계약\n\n'+next_ko+' 기존C·T3F/개인자료/원본/이전결과보존. 완료검산·추출·점수재실행X.\n\n'
    for section,replacement in [('## 8.',entry8),('## 9.',entry9)]:
        start=text.index('### ',text.index(section));end=text.index('\n### ',start+4)+1
        text=text[:start]+replacement+text[end:]
    (ROOT/'WordsInSilence_PROJECT_CONTEXT.md').write_text(text,encoding='utf8')
    handoff=f'''# WordsInSilence 최신 인수인계

현재 {N.name}, {status}. 컴퓨터재부팅후완료된native9추출을해시재사용하고중단판독을재개해현재검증마감. active execution/agent 없음. CURRENT_EXECUTION/COMPLETION/RESULT_KO.txt 우선. 학습/추론/재추출0,HTML없음.

- 원9 왼손애매6중5확정일치,N008양쪽unknown. native117frame/pixel2252·진단1113검사PASS.
- 별도support_inventory: FIT180창×2손360기회에서동결12창,156frame. 적격hold없음 R4/3ID/2표현,L1/1ID/1표현. 한쪽unknown1/반대확정0보존. 선택·픽셀4057/결과589검사PASS. 모델성능아님.
- 같은원24전체: 원9raw정확재사용+미검토15native195frame,검산3553. A15는PART1 P01~09 +PART2 P10~15 원시행그대로병합(출처해시보존); B독립15. 원W의calculate/gates그대로,지원12분모혼합X. FULL24_AUDIT 최초exit0 및별도원시검산PASS. 확정일치75/96,반대확정0,한쪽unknown4,양쪽unknown17(이전77/0/13/6). 일치율향상/분류정확도로해석X.
- 원24 최종FAIL은 R_hold_eligible_both_classes/L_hold_eligible_both_classes 2개. 관측한정지없음1창N022는ROI밖이므로양손적격음성0. 코드오류가아니라준비자료지원부족. 원24FAIL/centreFAIL/native9/지원12모두불변. 아직전체288주석/6학습은진행안함.
- C·T3F/운영웹·DTW/입력·정상교차자료그대로,보호32SHA확인. 사용자·멘토자료접근/학습/튜닝/선택X. 단어중요도/교육정오/전문가정답아님.

다음 한 작업: {next_ko}

Python: research/front_keyphase_v1/ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8. 완료출력재실행X,원본W:/sources/과거동결읽기전용. root만§8§9/STATE/HANDOFF수정. 새자료/모델다운로드/카메라/외부전송/Git/운영교체X.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
    assert read(O/'STATE.json')['current_stage']==status
    assert '같은96항목' in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text(encoding='utf8')
    print('SYNC_PASS_CONTEXT_STATE_HANDOFF')
if __name__=='__main__':main()
