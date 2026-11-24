"""Recount saved results, report limits and pin the next diagnostic; no inference."""
from common import *
from collections import Counter
from html import escape
def main():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected'])
    iv=read(U/'INPUTS_VERIFIED.json');cv=read(U/'CLASSIFIER_VERIFIED.json');assert cv['status']=='PASS';check(iv['pins'])
    ri=read(U/'CLASSIFIER_INPUTS.json');result=read(U/'CLASSIFIER_RESULT.json');visual=read(U/'VISUAL_REVIEW.json')
    assert sha(U/'verify_inputs_v2.py')==read(U/'VERIFY_REPLAY_RECOVERY.json')['recovery_verifier_sha']
    ctx=read(B/'training_CONTEXT.json');labels=ctx['labels'];timing=np.zeros(6);alltimes=[];trigger=[];totals=Counter();cross=0;scales=spans=0
    for row in f['rows']:
        p=load(U/'outputs'/(row['clip_id']+'.npz'));timing+=p['timing_ms'].sum(0);alltimes.extend(p['timing_ms'].sum(1).tolist());trigger.extend(p['timing_ms'][p['mp_called']].sum(1).tolist())
        totals['mp_trigger_frames']+=int(p['mp_called'].sum());totals['yolo_trigger_frames']+=int(p['yolo_called'].sum())
        rec=read(U/'records'/(row['clip_id']+'.json'));totals['mp_points']+=rec['mp_added'];totals['yolo_points']+=rec['yolo_added']
        # Screen-x reversal is a diagnostic group, not anatomical truth; no label-dependent rule.
        base=p['xy_px'][0];valid=p['usable'][0];ok=valid[:,33]&valid[:,54]
        crossed=ok&(base[:,33,0]<base[:,54,0]);cross+=int(crossed.sum())
        for k in (1,2):
            assert same(base[crossed][valid[crossed]],p['xy_px'][k,crossed][valid[crossed]])
            assert p['usable'][k,crossed][valid[crossed]].all()
    for r in ri['rows']:
        for k in ('retry','retry_yolo'):
            a=r['meta']['native'];b=r['meta'][k];spans+=a['span']!=b['span'];scales+=a['hand_scales']!=b['hand_scales'] or a['body_scale']!=b['body_scale']
    additions=Counter();
    for a in iv['queue']:additions[a['stage']+('_wrist_only' if a['position_only'] else '_multiple_points')]+=1
    counts=result['summary'];table=[]
    for name in ('C13','T3F20261013','T3F20261014','T3F20261015','E3'):
        rr=counts[name];table.append([name,rr['native']['correct'],rr['retry']['correct'],rr['retry_yolo']['correct'],rr['retry_fixed_scale_span']['correct'],rr['retry_yolo_fixed_scale_span']['correct']])
    mean_before=sum(counts['T3F'+str(s)]['native']['correct'] for s in (20261013,20261014,20261015))/9
    mean_after=sum(counts['T3F'+str(s)]['retry']['correct'] for s in (20261013,20261014,20261015))/9
    changes=[]
    for name in ('C13','T3F20261013','T3F20261014','T3F20261015'):
        scores=read(U/(name+'_SCORES.json'));ps=scores['predictions'];old={r['clip_id']:r for r in ps['native']}
        for r in ps['retry']:
            a=old[r['clip_id']]
            if a['prediction']!=r['prediction']:
                changes.append(dict(model=name,clip_id=r['clip_id'],target=labels[r['label_index']],before=labels[a['prediction']],after=labels[r['prediction']],correct_before=a['correct'],correct_after=r['correct']))
    detail=dict(at=now(),input_counts=iv['counts'],additions=dict(additions),table=table,changes=changes,
       t3f_mean_percent_before=mean_before,t3f_mean_percent_after=mean_after,t3f_mean_change_percentage_points=mean_after-mean_before,
       screen_x_crossing_frames_preserved=cross,scale_changed_clip_conditions=scales,span_changed_clip_conditions=spans,
       timing_sum_ms=dict(zip(('decode','mp_full','mp_crop','yolo_full','yolo_crop','geometry'),timing.tolist())),timing_extra_average_per_source_frame_ms=float(np.mean(alltimes)),
       triggered_p50_p95_ms=np.percentile(trigger,[50,95]).tolist(),timing_caveat='Extra preparation only: original detector cached, excludes original detection/DTW/end-to-end camera/UI and disk output. CPU, not realtime FPS guarantee.',
       visual_review=visual,numeric_replay_recovery=read(U/'VERIFY_REPLAY_RECOVERY.json'),candidate_adopted=False,production_changed=False,training=0,user_data_accessed=False)
    write(U/'DIAGNOSTIC.json',detail)
    txt=f'''공개 DEV300 실제 손 보완 비교 — 2026-10-26

결론
YOLO를 함께 쓰는 연구 경로를 실제 공개 영상300개에 연결했다. MediaPipe 전체 영상 관측을 유지하고, 화면 안 팔 근거가 있는데 손 전체를 놓친 순간만 재검출한다. 그래도 놓친 순간에는 YOLO를 보조로 사용하고 같은 상반신 범위 안의 점만 남긴다.
손 관측은 늘었지만 모든 모델의 인식이 안정적으로 개선된 것은 아니다. YOLO의 추가 관측만으로 정답 수가 더 늘지는 않았다. 손가락이 가려진 모양을 정확히 복원했다는 결론은 내리지 않는다. 운영 C·T3-F·웹·DTW는 변경하지 않았다.

자료와 처리
100개 단어 × 기록ID11/12/13 = 공개 개발 검증영상300개, 총37295프레임. 이미 모델 개발에 쓰인 자료이며 새 사용자 최종 정확도가 아니다. 다른ID를 외형으로 다른 사람이라고 단정하지 않는다. 사용자/멘토 촬영과 이전추가199/200자료는 접근·학습·튜닝·선택에 사용하지 않았다.
검출 대상은2563프레임(2716손 시간대). MP재검출2131손 장면/40822관절점 추가, 그 뒤 YOLO48손 장면/387점 추가. YOLO48중29는 손목1점만,19는 여러점. 같은 손의 여러 프레임을 포함하는 수이며48개 영상 또는48명의 사람이 아니다. 이 수는 검출 정확도가 아니다.
기존1285373 유효 관절·프레임 값과좌표를 보존하고, 원래분류입력6113800채널값을 정확히 재현했다. 원래 화면상 손목좌우 순서가 교차한 {cross}프레임도 기존좌표·마스크가 그대로다. 이는 기존관측 보존이며 모든 기존좌표의 정답 인증은 아니다.

단어 인식 결과 (모두300개 중 정답 수)
모델 | 기존 | MP재검출 | MP+YOLO | MP/기존크기·구간 고정 | MP+YOLO/기존크기·구간 고정
'''+ '\n'.join(' | '.join(map(str,row)) for row in table)+f'''

C13:294/300(98.00%)→295/300(98.33%),+0.33%p. 기존크기·구간을 유지하면294로같다.
T3-F세모델:295/299/297→296/298/298. 평균99.00%→{mean_after:.2f}%,+{mean_after-mean_before:.2f}%p. 범위98.33~99.67%→98.67~99.33%. 두 모델+1,한모델−1이며일관된개선이라고말할수없다.
고정E3(기존T3-F세모델확률동일평균):모든조건300/300. 이미기존입력에서천장이므로추가이득을증명하지못한다. 실사용100%라는뜻이아니다.
새학습0회. 원래입력은기존저장점수재사용, 바뀐고유입력674개를4개고정모델에각1회만계산. E3는저장점수집계.7500개논리예측의순위·개선/회귀·확률계산을별도검산PASS. 같은Codex의별도산술검산으로독립사람정답검증과다르다.

실제로바뀐예
'''+ '\n'.join(f"{x['model']}: {x['target']} / {x['before']}→{x['after']} ({x['clip_id']})" for x in changes)+f'''
수영은동일새입력에서한T3-F는정답으로개선되고다른T3-F는오답으로바뀌었다. 손크기고정에서도같은방향이어서크기변화만으로설명되지않는다. 보완된입력에각고정모델의결정이서로다르게반응한다. 이를수영전용예외로수정하지않는다.
전체두자연조건600클립조건중크기변화{scales},구간변화{spans}. 강진군수어통역센터의C개선은기존크기고정조건에서는없어정규화변화가함께작용한것으로보인다. 손을더찾음=정확도항상증가로해석하면안된다.

시각검토와정밀피드백의한계
분류점수를보기전에새관측2179손장면중103개를선정·검토했다(모든YOLO48,가까운양손24,ID당MP12표본,중복제거). 첫3영상29개검토도별도보존. 전체2179개시각전수검증은아니다. 손목의큰위치·팔연결은그럴듯했지만, YOLO의펴진손가락/주먹표현이뭉개진사례와겹침모양의불확실성이남았다. 대표검토번호40·59·76·80·81·100은정확한손가락교정근거로승격하지않는다. 숫자는정량정답주석이아니라검토위치이다.
두영상크기의예측이일치해도같은오류를공유할수있다. 손목1점을찾았으면위치근거로만사용하고손가락각도를만들지않는다. 가려진손가락은추정/근거부족으로남겨야하며, 보인점이많다고올바른수어수행으로판정하지않는다.

처리시간
CPU에서추가준비처리평균{np.mean(alltimes):.2f}ms/원본프레임. 실제보완호출프레임중앙값{np.percentile(trigger,50):.2f}ms,95백분위{np.percentile(trigger,95):.2f}ms. 기존검출은저장자료재사용. 이숫자에는기존검출·DTW·카메라UI전체시간이없으므로실시간FPS로환산하지않는다. 전체297추가준비실행430.51초,첫3표본15.77초와초기설정별도.

검증중복구
최초저장결과재생검산에서JSON null을실수NaN으로복원하지않아오류가났다. 원본검산기/오류근거를보존하고verify_inputs_v2.py에서숫자형복원만추가해전체검산을통과했다. 검출결과·좌표·판단상수·모델·점수는수정하거나다시뽑지않았다.

다음작업
전체목표는계속진행중이다. 이번재검출연결을무조건분류입력교체로채택하지않는다. 기존인식입력과보완관측을각각보존하는연구연결부를만들고, DTW에서는추가손목위치와불확실한손가락을구분해전달한다. 공개TRAIN의기존위치정답근거/정상손/교차자료로후보상태를먼저검증하고, 충분히관측된부분·추정·근거부족을구별한다. 공개수영회귀등은이후추가분류학습전에공통원인을진단하며단어별예외를만들지않는다. 같은DEV점수를반복보며임계값을맞추지않는다. 사용자녹화는학습/튜닝에사용하지않고운영교체는별도다.
'''
    (U/'RESULT_KO.txt').write_text(txt,encoding='utf8')
    body=escape(txt).replace('\n','<br>');images=''.join(f'<details><summary>검토 장면 {i*8}–{min(i*8+7,102)}</summary><img loading="lazy" src="review_visual/page_{i:02d}.jpg"></details>' for i in range(13))
    (U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>공개300 실제 손 보완 비교</title><style>body{font:17px/1.75 system-ui;background:#eef3f8;color:#172b43;margin:0}main{max-width:1080px;margin:32px auto;padding:42px;background:white;border-radius:18px}h1{font-size:30px}details{margin:20px 0;padding:12px;background:#f2f5fa}img{max-width:100%}.status{color:#914312;background:#fff2dd;padding:16px;border-radius:9px}</style><main><h1>공개300개 영상에서 실제 보완 효과 확인</h1><p class="status">손 관측 증가 확인 · 인식은 제한적 개선과 회귀 공존 · 정밀 손가락 피드백 미검증 · 운영 유지</p><div>'+body+'</div><h2>원본과 추가 관측을 나란히 보기</h2><p>주황:새로찾은손, 파랑:기존다른손. 왼쪽원본/오른쪽관절표시. 검토번호는성공점수가아닙니다.</p>'+images+'</main></html>',encoding='utf8')
    write(U/'DONE.json',dict(at=now(),status='DIAGNOSTIC_COMPLETED_MIXED_CLASSIFIER_EFFECT_SHAPE_UNCERTAINTY',preparation_session=56293,preparation_exit=0,verification_session=3453,verification_exit=0,classifier_session=15070,classifier_exit=0,goal_complete=False,automatic_adoption=False,
       evidence_pins={str(U/n):sha(U/n) for n in ['FROZEN.json','INPUTS_VERIFIED.json','VISUAL_REVIEW.json','CLASSIFIER_FROZEN.json','CLASSIFIER_INPUTS.json','CLASSIFIER_RESULT.json','CLASSIFIER_VERIFIED.json','DIAGNOSTIC.json','RESULT_KO.txt','RESULT_KO.html']}))
    print('REPORT WRITTEN',json.dumps({k:detail[k] for k in ['screen_x_crossing_frames_preserved','scale_changed_clip_conditions','span_changed_clip_conditions']}),flush=True)
if __name__=='__main__':main()
