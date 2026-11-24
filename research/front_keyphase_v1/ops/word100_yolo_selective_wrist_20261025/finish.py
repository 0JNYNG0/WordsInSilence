from common import *
import datetime,re
def replacejson(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
f=read(U/'FROZEN.json');s=read(U/'SUMMARY.json');v=read(U/'VERIFY_SAVED.json');notes=read(U/'FINAL_NOTES.json')
assert v['status']=='PASS' and read(U/'CONTRACT_CHECK.json')['status']=='PASS'
assert not (U/'RUNNING.json').exists() and (U/'RESULT_KO.html').exists()
for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:assert sha(p)==sha(U/'before'/p.name),'Root context changed concurrently'
page=(U/'RESULT_KO.html').read_text('utf8');links=re.findall(r'(?:src|href)="([^"]+)"',page)
for link in links:
    if not link.startswith(('http:','https:','#')):assert (U/link).exists(),link
at=datetime.datetime.now().astimezone().isoformat();key=U.name
entry={'status':'COMPLETED_SELECTIVE_PUBLIC_VALIDATION_RESEARCH_ONLY','at_kst':at,'summary':s,'conclusion':notes['conclusion'],'next':notes['next'],
 'validation':v,'contract_checks':12,'visual_review':'all5held and all7new additions viewed; prediction-aware Codex, not blind GT',
 'existing_C_T3F_unchanged':True,'training':0,'TCN_inference':0,'user_mentor_recordings_access':0,'runtime_changes':0,'running':False,'report':str(U/'RESULT_KO.html')}
write(U/'FINAL_RESULT.json',entry)
state=read(O/'STATE.json');state['updated_date_kst']=at;state['current_stage']='YOLO_SELECTIVE_WRIST_COMPLETED_RESEARCH_ONLY';state['next_single_action']=notes['next'];state[key]=entry;replacejson(O/'STATE.json',state)
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업';p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text('utf-8-sig');assert key not in txt
status=f'''

### 2026-10-25 MP 재검출 이후 선택적 YOLO·손목/모양 분리 완료 — 운영 미적용
- 단위 `research/front_keyphase_v1/ops/{key}`. 기존39영상4781프레임+새YOLO비교확장07촬영자7영상856프레임, 합계46영상5637프레임. 사용자/멘토 접근0, TEST08/09 접근0, 새학습/TCN추론/운영변경0.
- {notes['conclusion']}
- M1(재검출)→S(누락시엄격손전체)→P(손목/모양분리): R24판독가능23개 12→13→13; V18판독가능16개 12→12→12. 추가지원R24 1개/기존지원손실0/새범위밖추가0. 기존주석표본내결과이며 전체손가락/수어정확도아님.
- 기존 M1 101680관절-프레임 및35교차프레임 정확보존. 새추가1손전체+6손목전용 프레임. 6손목은정밀정답없어정확도향상확정금지. 관측분모·영상프레임·시간축줄임없음.
- 보류5: R01/R14 확대중복·손모양불일치, R21/R22 손목도32.6/36.5px불일치, R02 확대미검출+M1이미잘못된위치존재로누락전용보조대상아님. 전부미해결유지.
- 호출139/4781프레임(97.1%절감)+새07에서1/856호출·추가0. 전체평균YOLO추가3.26ms, 호출프레임평균112.3ms/p95170.6ms. 과거M1+새YOLO부품합산70.0→73.3ms추정, 실제웹캠FPS/동일세션end-to-end아님.
- 공통규칙/기존0.08어깨폭허용치고정·학습라벨입력없음. 신규12계약검사/전체5637 native→ROI재구성/저장126행별도산술검산/보호32확인. 동일Codex별도검산, 독립전문가검토아님. 기존결과읽기전용, 운영보존. 다음계획·한계는RESULT_KO.html/FINAL_RESULT.json.
'''
nxt=f'\n\n### 2026-10-25 선택적 YOLO 검증 이후\n\n{notes["next"]}\n'
p.write_text(txt.replace(h8,h8+status,1).replace(h9,h9+nxt,1),encoding='utf8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 공개 영상에서 손 관측 근거를 정확히 개선. 기존C·T3-F/운영웹DTW 보존.
마지막 완료: {key}. MP재검출도놓친경우만YOLO, 손목위치/손가락모양근거분리 검증.46영상5637프레임(기존39+추가07촬영자7).
{notes['conclusion']}
R24판독23: M1 12/23→S/P13/23. V18판독16:12/16유지. 추가R24 1개,기존지원손실0,새범위밖추가0(주석표본내). 전체추가1손전체+6손목전용손-프레임;6개정밀정답없음.
기존M1 101680관절-프레임과35교차프레임보존. 프레임/시간/관측분모줄임없음.
5장면미해결: R01/R14중복확대+모양불일치, R21/R22손목도불일치, R02 M1잘못된검출은누락전용보조로고치지않음.
기존39에YOLO139/4781프레임,새07 1/856프레임호출·추가0. 새07은보존확인이지복원효과입증아님.
새YOLO비용전체평균3.26ms,호출프레임평균112.3ms/p95170.6ms. 이전M1측정+새YOLO합산70.0→73.3ms추정. 실웹캠속도미측정.
구현Codex/12공통계약검사/5637프레임nativeM1→ROI재구성/별도산술126행PASS/보호32불변. 예측본상태5보류+7추가시각검토, 독립정답아님.
모델가중치변경0/학습0/TCN추론0/사용자·멘토촬영접근0/운영교체0/TEST08·09접근0. 기존결과·원본W:/·sources읽기전용. Git·외부전송없음.
실행66594종료0, INDEX완료, RUNNING없음. 새모델채택없음.

다음 한 작업: {notes['next']}

근거: {U/'RESULT_KO.html'}
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
ex=read(U/'CURRENT_EXECUTION.json');ex.update(status='COMPLETED_EXIT_0',finished=at);replacejson(U/'CURRENT_EXECUTION.json',ex)
assert read(O/'STATE.json')[key]['status']==entry['status'];assert key in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf8')
write(U/'DELIVERY_CHECK.json',{'status':'PASS','report_links_checked':len(links),'protected':32,'documents_synced':True,'running':False})
print('SYNCED',key)
