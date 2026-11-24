from common import *
import html
def main():
    assert read(U/'VERIFIED.json')['status']=='PASS' and read(U/'CLASSIFIER_VERIFIED.json')['status']=='PASS'
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected'])
    d=read(O/'word100_recovery_dev300_20261026/PREFLIGHT.json');assert d['counts']['clips']==300
    ci=read(U/'CLASSIFIER_INPUTS.json');ss=read(U/'CLASSIFIER_RESULT.json');co=read(U/'COMPARE_RESULT.json')
    scales=spans=0
    for r in ci['rows']:
        a=r['meta']['native']
        for k in ('retry','retry_yolo'):
            b=r['meta'][k];scales+=a['body_scale']!=b['body_scale'] or a['hand_scales']!=b['hand_scales'];spans+=a['span']!=b['span']
    lines=[
      '검출 보완 결과를 실제 관측 입력·DTW·고정 분류 모델에 연결',
      '전체 개선 목표는 진행 중입니다. 운영 웹이나 기존 C·T3-F를 교체하지 않았습니다. 이번에는 저장된 실제 공개 영상 관측을 연구용 입력으로 연결하고, 수치가 안전하게 이어지는지 확인했습니다.',
      '처리 순서: 원본 전체 화면의 MediaPipe 관측 → 같은 원본 장면의 MediaPipe 재검출 → 그래도 손을 못 찾은 순간에 선택적 YOLO → 원래 상반신 범위로 제한 → TCN과 DTW에 각각 전달합니다. 한 프레임의 검출 확대 영역은 학습 화면 범위를 새로 자른다는 뜻이 아닙니다.',
      '대상: 이미 연구에 사용했던 공개 TRAIN 영상 46개, 20개 단어, 기록된 촬영자 ID 7개, 5,637프레임입니다. 새 사용자 일반화 시험이 아닙니다. 원래 손 관절점 101,680개와 기존 교차 장면 35프레임을 그대로 보존했습니다.',
      'MediaPipe 재검출은 원래 ROI 안의 관절 관측을 95,459개에서 101,680개로 늘렸습니다. 선택적 YOLO는 27개 관절 관측을 더했습니다. 21점 손 한 순간과 손목만 있는 여섯 순간입니다. 이는 정확한 손 27개를 새로 얻었다는 뜻이 아닙니다.',
      '손목만 찾으면 손 위치 비교에만 씁니다. 손바닥 중심이나 손가락 굽힘으로 바꾸지 않습니다. 손가락별 각도는 두 예측의 기하 관계가 비교 가능한지 별도로 기록합니다. 두 검출 결과의 일치는 실제 정답을 보증하지 않으며, 영상에 가려진 손가락의 정답 주석도 아직 없습니다.',
      'HaMeR의 추정 관절 546개는 별도 배열로 남겼습니다. 손 영역 추적과 손가락 정밀 좌표를 구분했습니다. 해결되지 않은 10프레임의 양손 신원 충돌은 좌우를 억지로 확정하지 않았고, 원래 좌표를 보존하면서 DTW의 해당 손 근거를 보류했습니다. 팔 관절 정보는 따로 유지합니다.',
      'DTW 확인: 위 묶음 가운데 기존 비교 경로가 있는 3단어·21영상의 다른 촬영자 조합 126쌍을 사용했습니다. 시간 정렬·참고 영상·비교할 시간 범위를 고정하고 쿼리 관측만 바꿨습니다. 여러 참고에 같은 프레임이 쓰이므로 아래 수는 중복을 포함한 비교 항목 수이며, 사람 수나 정확도 퍼센트가 아닙니다.',
      '손가락 각도 비교 가능 항목: 원래 75,648 → MediaPipe 재검출 78,489 → YOLO 보조 78,489. 손바닥 위치는 15,302 → 15,860 → 15,860. 손 검출기의 손목 x/y는 15,922 → 16,492 → 16,514. 기존에 비교 가능했던 항목의 손실은 세 그룹 모두 0개였습니다.',
      'YOLO가 이번 DTW 묶음에서 더한 것은 손목 x/y 비교 22항목입니다. 손가락 비교를 추가로 늘리지는 못했습니다. 팔꿈치·몸 검출기의 손목은 그대로 유지했습니다. 손이 보이지 않는다고 비교할 전체 범위가 줄지 않으며, 근거 없는 차이값은 0점 대신 미확인으로 남깁니다.',
      'TCN 확인: C와 T3-F 세 모델, 기존에 고정한 세 모델 평균 E3를 변경 없이 적용했습니다. 원래/재검출/YOLO 보조 모두 46개 중 46개를 맞혔습니다. 학습에 쓰인 자료에서 이미 모두 맞히던 상태라 인식률 개선을 입증한 결과는 아닙니다. 정상 인식의 회귀가 없었음을 이 범위에서 확인한 것입니다.',
      f'보완 전후 자연 전처리 92개 대조 중 {scales}개에서 크기 정규화 기준이 달라졌고 시작·끝 범위 변화는 {spans}개였습니다. 그래서 기존 크기와 범위를 고정한 두 조건도 추가로 확인했습니다. 이 조건도 46/46을 유지했습니다. 정규화가 달라지는 현상은 실제로 있으나, 이 묶음에서 오답을 만든 증거는 없었습니다.',
      '검증: 별도 산술 구현으로 각도 상태 507,330개와 DTW 채널 비교 2,070,000개를 검산했습니다. 저장된 분류 결과 1,150개를 재추론 없이 다시 집계했습니다. 동일 Codex의 별도 산술 검증이며, 독립 사람의 손가락 정답 검토는 아닙니다.',
      '다음 단계는 공개 개발 검증 300영상입니다. 37,295프레임의 원본 관측과 기존 상반신 입력이 일치하고 원본 ZIP 항목이 있는지 확인했습니다. 현재 손 전체가 없고 화면 안 팔·손목 근거가 있어 재검출을 시도할 순간은 2,563프레임입니다. 이 준비 점검에서는 새 검출이나 새 점수 계산을 아직 하지 않았습니다.',
      '이어서 이 300개에서 필요한 순간만 재검출하고, 검증한 입력으로 기존 고정 C·T3-F·E3 결과와 비교합니다. 좋아진 단어뿐 아니라 새 오답, 정상 손 보존, 비교 근거, 처리 시간을 함께 확인합니다. 이미 존재하지만 잘못된 손이나 팔 근거마저 없는 경우는 이 보완만으로 해결되지 않을 수 있습니다. 성공을 위해 좌우를 강제하거나 불확실한 손가락을 정답처럼 채우지 않습니다.',
      '사용자·멘토 촬영 접근 0, 추가199/200 평가 재사용 0, 새 학습 0, 모델 가중치 변경 0, 운영 웹·DTW 변경 0입니다. 검출 보완이 연구 입력에 연결되었으나, 검증된 개인별 교정 피드백이 완성되었다고 선언할 단계는 아닙니다.'
    ]
    (U/'RESULT_KO.txt').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
    names={'angle':'손가락 각도','palm':'손바닥 위치','elbow':'팔꿈치','pose_wrist':'몸 검출기의 손목','hand_wrist':'손 검출기의 손목'}
    table='<table><thead><tr><th>비교 근거</th><th>원래</th><th>MP 재검출</th><th>YOLO 보조</th></tr></thead><tbody>'
    for k,g in co['groups'].items():table+='<tr><td>'+names[k]+'</td>'+''.join(f'<td>{g["compared_"+str(i)]:,}</td>' for i in range(3))+'</tr>'
    table+='</tbody></table>'
    body='<h1>'+lines[0]+'</h1>'+''.join('<p>'+html.escape(x)+'</p>'+(table if i==7 else '') for i,x in enumerate(lines[1:],1))
    (U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>검출 보완 연결 검증</title><style>body{max-width:1050px;margin:40px auto;padding:24px;font:18px/1.8 system-ui;background:#f4f7fb;color:#16293f}p,table{background:white;padding:20px;border-radius:12px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:14px;border-bottom:1px solid #dbe2eb}h1{font-size:30px}</style><body>'+body+'</body></html>',encoding='utf8')
    contract='''연구 입력 계약\ninputs/*.npz: 조건0원래/1MP재검출/2선택적YOLO. xy/valid는 기존 관측 보존. point_origin0없음/1원래/2MP재검출/3YOLO.\nvalues/available/eligible: 두 손×23항목(15각도,손바닥xy,팔꿈치xy,몸손목xy,손손목xy). angle_state0미관측/1퇴화/2미검토/3예측불일치/4예측비교가능. 4는 정답 인증 아님.\nidentity_conflict: 좌우 보류, 원래 좌표 유지. inferred_xy/inferred_valid와 region_advisory는 measured 입력에 합치지 않음.\ncomparisons/*.npz: 기존 경로·고정 참고 시간분모, support와 difference를 함께 읽을 것. 근거부족은NaN,0차이로대체금지. 전체정답/교정방향판정없음.\nclassify.py는 연구용고정모델입력연결/5조건진단. 원래좌표를보존한채 기존280채널로전환. DTW품질마스크를새TCN채널로가르치지않았음. 동일입력해시별모델추론1회,가중치불변.\n운영교체금지. 사용자자료튜닝금지. Gloss/word_code는참고연결·결과라벨에만사용.\n'''
    (U/'INTEGRATION_CONTRACT_KO.txt').write_text(contract,encoding='utf8')
    summary='공개46 TRAIN영상/20단어/7ID의 실제 native→MP재검출→선택YOLO를 연구입력/DTW/고정TCN에연결. 원래101680점/35교차보존. 126다른사람쌍의고정경로에서각도75648→78489→78489,손목15922→16492→16514(비교항목수,정확도아님). HaMeR546추정점/10프레임20손신원충돌분리. 고정C/T3F3seed/E3 모든5조건46/46로회귀없지만TRAIN천장으로개선증명없음. 검산PASS. 운영/사용자자료불변;전체goalactive.'
    nextstep='word100_recovery_dev300_20261026/PREFLIGHT.json부터공개DEV300실제재검출조건동결·실행. 37295프레임ROI/시계/ZIP항목일치,2563미검출+화면안팔근거프레임(2716손)대상. 현재새검출0/신규DEV점수0. 기존상수/정상좌표보존하여 전체원본MP IMAGE재검출→그후놓친순간선택YOLO→같은ROI. 프로토콜·중복실행잠금확인,영상해시/입력독립검산·추가손시각검토후 고정C13/T3F3seed/E3비교. 동일기존입력은저장점수재사용. 크기·시작끝변화효과분리,DEV회귀/속도/근거부족함께보고. HaMeR/영역추정정밀점승격금지. 사용자/멘토/추가199/200접근·학습·튜닝금지,운영보존. 작은검증종료로goal완료처리금지.'
    at=now();write(U/'DONE.json',dict(at=at,status='VERIFIED_RESEARCH_BRIDGE_DEV_EXPANSION_NEXT',goal_complete=False,next=nextstep,report=str(U/'RESULT_KO.html')))
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf8-sig')
    for prefix,title,content in [('## 8.','### 2026-10-26 실제 검출 보완 연결 — 공개46 입력·DTW·분류 보존 검증',summary),('## 9.','### 2026-10-26 지속 개선 — 공개 개발 검증300 실제 재검출 확대',nextstep)]:
        assert title not in s;i=s.index('\n',s.index(prefix));s=s[:i+1]+'\n'+title+'\n'+content+'\n\n'+s[i+1:]
    context.write_text(s,encoding='utf8');statep=O/'STATE.json';state=read(statep)
    state.update(updated_at=at,updated_date_kst=at,current_stage='AUTONOMOUS_PUBLIC_DEV300_RECOVERY_PREPARED',next_single_action=nextstep)
    state[U.name]=dict(at=at,status='VERIFIED_RESEARCH_INPUT_COMPARISON',summary=summary,report=str(U/'RESULT_KO.html'),running_processes=[],goal_complete=False)
    state['word100_recovery_dev300_20261026']=dict(at=at,status=d['status'],preflight=str(O/'word100_recovery_dev300_20261026/PREFLIGHT.json'),counts=d['counts'],new_detector_inference=0,new_classifier_inference=0,running_processes=[],next=nextstep)
    statep.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n현재: '+summary+'\n\n다음: '+nextstep+'\n\n완료결과: '+str(U)+'\nVERIFIED/CLASSIFIER_VERIFIED PASS. 기존결과 재추론금지. DEV준비exec83016 종료0,진행프로세스없음. 새unit는PREFLIGHT만있고 아직FROZEN/실행기없음. 목표active,후속heartbeat없음. 구현·별도산술검산동일Codex,독립사람정답아님.\n',encoding='utf8')
    assert nextstep in (ROOT/'NEW_CHAT_HANDOFF.md').read_text('utf8') and summary in context.read_text('utf8')
    check(f['protected']);write(U/'SYNC_CHECK.json',dict(at=now(),status='PASS',goal_complete=False,report_validation='Static HTML checks only; not browser-render inspected'))
    print('SAVED; DEV300 preflight ready; larger goal active',flush=True)
if __name__=='__main__':main()
