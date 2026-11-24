from pathlib import Path
from datetime import datetime,timezone,timedelta
from html import escape
import json,hashlib,re
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];UNIT=U.name
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
NEXT=('사용자와 순간/연속 근거 및 모습/변화량 분리 결과를 공동 검토한다. 기존 순간별 결과 보존은 검증했으나 새 분리 차이 표시는 정상속도에도 크게 증가하여 자동 오류판정·운영 적용은 보류한다. '
      '다음 제안은 알려진 시간변환 대조에서 변화량 경고가 생긴 항목을 시간 샘플링·정렬 대응·개인차 범위 영향으로 나누는 진단이다. '
      '동일 동작의 변화량이 속도만으로 달라지는지 먼저 확인하고, 순간 비교를 유지하면서 연속 근거 부족은 별도 상태로 남긴다. '
      '경고를 줄이기 위한 문턱 튜닝이나 단어별 예외 없이 계획을 먼저 고정한다. 반복·별도 공개 단어/사람에 대한 일반화는 미검증. '
      'C·T3-F·운영 DTW/웹 유지, 사용자·멘토 자료 접근/학습/튜닝/선택 금지. 추가 학습은 결과 공동 확인 후.')

def main():
    a=read(U/'ANALYSIS.json');r=read(U/'RESULT.json');f=read(U/'FROZEN.json');v=read(U/'VERIFY.json')
    assert a['status']==v['status']=='PASS';assert v['total']==2808
    normal=[x for x in r['cross'] if x['kind'] in ['identity','fast075','slow15','pace','pause']]
    n=a['cohorts']['cross']['normal'];num=0;pointonlybad=0
    for row in normal:
        with np.load(U/'cross'/(row['name']+'.npz')) as z:
            pointonlybad+=int((z['point_bad']&~z['sequence_comparable']).sum())
    for g,data in n['groups'].items():
        for k in ('expected','point_comparable','point_bad','sequence_comparable','level_bad','change_bad','split_any_bad'):
            assert data[k]==sum(x['groups'][g][k] for x in normal);num+=1
    assert pointonlybad==36116
    for p,h in f['protected'].items():assert sha(Path(p))==h
    save(U/'AGGREGATE_CHECK.json',{'status':'PASS','at':now(),'metadata_sum_checks':num,'point_only_prior_differences_preserved':pointonlybad,
        'basis':'Saved metadata aggregation plus independent NPZ sum of point-only prior differences; new-array arithmetic verified separately in VERIFY.json.',
        'protected':len(f['protected']),'analysis_sha256':sha(U/'ANALYSIS.json')})
    text=[
    ('결론','기존 순간별 비교를 유지하고 연속 비교의 가능 여부를 따로 표시하는 보완을 완료했다. 위치·각도와 변화량도 각각 분리했다. 공개 자료 검산은 통과했다. 다만 분리한 차이 표시가 정상적인 속도 변형에도 크게 늘어나, 이를 바로 동작 오류나 교정 지시로 사용하는 것은 보류한다. 연구용 결과와 보고서에만 반영했고 운영 웹과 C·T3-F는 유지했다.'),
    ('이제 무엇을 구분하나','① 지금 이 순간의 모양·위치는 비교할 수 있는가 ② 앞뒤를 이어 움직임까지 비교할 수 있는가 ③ 실제 차이는 어느 항목에서 나타났는가를 각각 남긴다. 예를 들어 앞 장면의 손이 가려져도 현재 손의 위치가 보이면 순간 비교 결과는 그대로 남고, 앞뒤 움직임 비교만 근거 부족으로 표시된다. 연속 비교가 안 된다고 기존 차이를 지우거나 잘 맞음으로 바꾸지 않는다.'),
    ('기준을 어떻게 나눴나','기존 후보는 앞·중간·뒤 모습과 두 변화량 중 가장 큰 차이를 하나의 기준과 비교했다. 이번에는 세 모습의 차이와 앞→중간·중간→뒤의 변화량에 각각 기준을 둔다. 손가락 각도·손 위치·팔꿈치·손목은 단위를 섞지 않고 항목별로 유지한다. 큰 위치 차이를 허용하는 기준이 작은 움직임 차이까지 덮어주는 일을 분리해 볼 수 있다. 각 참고자의 연속 동작을 유지하며 가상의 평균 동작을 만들거나 가장 유리한 한 참고만 고르지 않는다.'),
    ('검증 범위','공개 TRAIN 자료의 수어·후원·멀다 3단어, 9명, 27영상으로 2,808개 대조와 243개 기준 범위를 검산했다. 한 비교의 기준은 대상과 해당 참고자를 제외한 7명 중 최소5명 관측으로 만든다. 같은 영상 대조는 다른8명이다. 기존 DTW 시간 연결과 이전에 검증한 관절 입력을 그대로 재사용했다. 새 검출·새 시간 정렬·TCN 추론·학습은 없다. 이전부터 분석하던 개발 묶음이며 별도 독립 최종 시험이나 전100단어 검증이 아니다.'),
    ('기존 순간별 정보 보존','정상 원본·속도 대조 1,080쌍의 순간 비교 가능 정보 1,148,965칸은 전부 그대로 유지했다. 그중 연속 비교까지 가능한 것은718,558칸, 순간 비교만 가능한 것은430,407칸이다. 이전 순간 비교에서 차이가 있었으나 연속 근거가 부족했던36,116칸의 차이 표시도 지우지 않았다. 손가락 각도 비교 가능 비율은 순간61.9%를 유지하고, 연속31.4%는 별도 표시한다. 이는 새 관측을 복구한 것이 아니라 기존 정보가 연속 검사에 가려지지 않게 보존한 결과다. 칸은 참고 시간×손×각도 또는 좌표 항목이다.'),
    ('짧은 구간을 지운 대조','3입력프레임씩 지운81변형에서, 다른8참고 중5개 이상에 새 차이가 나온 경우는 기존 순간30개, 이전 통합 연속31개, 모습만35개, 변화량만73개, 분리 항목 중 하나라도 반응70개였다. 변화량을 따로 보면 이전에 묻혔던 변화에 더 반응했다. 하지만 이것은 수어를 틀렸다고 정확히 판단한 비율이 아니다. 삭제된 구간의 의미상 필수 여부는 검증하지 않았다. 각 조건의 삭제 전 원본과 삭제 후 변형 모두 근거가 있는 곳에서 새 차이만 센다.'),
    ('정상 속도에서 남은 문제','정상 속도 변형864쌍에서 같은567,457칸을 비교하면, 새 차이 표시는 이전 통합 연속12,118칸에서 분리 항목40,685칸으로 늘었다. 모습만17,244칸, 변화량만39,252칸이다. 같은 원본을 시간만 바꾼 대조에서도 통합22칸→분리552칸으로 늘었다. 따라서 분리 방식은 작은 변화에 민감해졌지만 정상 속도에 대한 안전한 판정 기준이 되지는 못했다. 기존 순간 비교 결과에는 이 추가 경고를 합치지 않았다.'),
    ('73개와70개가 다른 이유','변화량만 볼 때73개인데 모습까지 포함한 항목 중 하나는70개인 것은 집계 오류가 아니다. 여기서는 원본에 없던 새 차이만 센다. 원본부터 모습 차이가 있었던 곳에 삭제 후 변화량 차이가 추가되면, 변화량에서는 새 차이지만 전체 항목 기준으로는 이미 차이가 있던 곳이다. 그래서 항목별 신규 변화 수를 단순히 더하거나 최댓값으로 해석하면 안 된다.'),
    ('손내림·관측 부족','원래 확인된 손내림·비교 필요 범위와 원촬영의 미관측 시간표를 그대로 유지했다. 손·팔 전체 또는 짧은 중간 관측을 제거한135개 대조에서도 필요한 비교 범위를 줄이지 않았다. 근거 부족은 정상이나0차이로 승격하지 않았다. 단 정상 손내림을 가로지르는 연속 창은 여전히 보류될 수 있다. 이번 보완은 그 때문에 비교 가능한 순간 정보까지 같이 보류되지 않도록 구분한 것이다.'),
    ('교차 보존','이전 검토한 후원 교차12프레임에서 손가락 각도360칸·손바닥 위치48칸은 원래 근거가 부족했으며 계속 보류했다. 팔꿈치48칸·손목48칸은 자기 영상 대조에서 비교 가능 상태와 차이0이 유지됐다. 원래 관측된 팔48개 점, 좌우 배정과 좌표 자체는 바꾸지 않았다. 이 결과를 교차할 때의 정확한 손 모양이나 모든 자연 교차의 정답성 검증으로 확대하지 않는다.'),
    ('판단과 다음 작업',NEXT),
    ('검증 기록','구현·실행·문서화는 Codex가 로컬에서 수행했다. 같은 Codex가 별도 검산 코드를 작성해 구간 중심·반경으로 차이를 다시 계산하고,5항목별 참고 기준과 저장 결과를 검산했다. 이전에 검증된 입력 재생 부분은 공유하므로 다른 에이전트의 완전 독립 검토나 수어 전문가 검토라고 주장하지 않는다. 2,808개 비교·243개 기준 검산,5개 합성 불변조건,집계·그림 확인을 통과했다. 보호 파일32개 해시를 확인했다. 사용자·멘토 촬영 접근/학습/튜닝/선택0. 추가 학습은 공동 확인 후 진행한다.')]
    (U/'RESULT_KO.txt').write_text('순간 비교 보존과 연속 항목 분리 검증\n\n'+'\n\n'.join(h+'\n'+p for h,p in text)+'\n',encoding='utf-8')
    cards=''.join('<section><h2>'+escape(h)+'</h2><p>'+escape(p)+'</p></section>' for h,p in text)
    table='''<div class="scroll"><table><tr><th>확인 항목</th><th>이전 통합 연속</th><th>모습만</th><th>변화량만</th><th>분리 항목 중 하나</th></tr><tr><td>짧은 삭제81개 중 5개 이상 참고 반응</td><td>31</td><td>35</td><td>73</td><td>70</td></tr><tr><td>정상 속도 대조의 새 차이 / 공통567,457칸</td><td>12,118</td><td>17,244</td><td>39,252</td><td>40,685</td></tr></table></div>'''
    pics=''.join('<figure><img src="visuals/'+p+'.png" alt="'+escape(t)+'"><figcaption>'+escape(t)+'</figcaption></figure>' for p,t in [('coverage','유지한 순간 정보와 추가 연속 정보를 별도로 표시한다.'),('short_response','원본·변형 모두 관측된 구간의 새 차이. 수어 오류 정답률이 아니다.'),('normal_speed','동일 근거에서 정상 속도 변형의 경고도 증가했다. 추가 표시의 자동 감점은 보류한다.')])
    links=' · '.join('<a href="'+p+'">'+t+'</a>' for p,t in [('FROZEN.json','고정 조건'),('VERIFY.json','별도 수치 검산'),('SELFTEST.json','합성 조건'),('AGGREGATE_CHECK.json','집계 확인'),('ANALYSIS.json','전체 집계'),('RESULT_KO.txt','텍스트 보고서')])
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>순간·연속 비교 분리 검증</title><style>*{box-sizing:border-box}body{margin:0;background:#f2f5fa;color:#182a40;font:17px/1.8 "Malgun Gothic",sans-serif}main{max-width:1120px;margin:auto;padding:36px 22px}header{background:#19324f;color:white;padding:32px;border-radius:16px}h1{font-size:32px;line-height:1.45}h2{font-size:23px;line-height:1.5;margin:0 0 12px}p{margin:0}section,figure{background:white;padding:26px;border:1px solid #dce5f0;border-radius:12px;margin:20px 0}img{width:100%;height:auto}figure{padding:12px}figcaption{padding:8px;color:#506178;font-size:14px}.scroll{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;width:100%;background:white;font-size:15px;min-width:730px}td,th{padding:14px;border:1px solid #dce5f0;text-align:left}th{background:#e7eef7}a{color:#16658d}footer{font-size:14px;padding:20px 0}@media(max-width:650px){main{padding:16px 12px}h1{font-size:25px}section,header{padding:20px}}</style><main><header><p>WORDS IN SILENCE · 공개 자료 DTW 연구 · 2026.10.22</p><h1>지금 보이는 정보는 유지하고,<br>앞뒤 움직임의 근거는 따로 구분</h1><p>정보 분리·보존 검증 완료. 새 차이 표시의 자동 오류판정은 보류.<br>C·T3-F·운영 웹 유지 / 사용자 촬영 사용·추가 학습 없음</p></header>'''+table+cards+pics+'<footer>'+links+'</footer></main></html>'
    (U/'RESULT_KO.html').write_text(html,encoding='utf-8')
    for target in re.findall(r'(?:src|href)="([^"]+)"',html):assert (U/target).is_file(),target
    save(U/'VISUAL_REVIEW.json',{'status':'PASS','at':now(),'plots':['coverage.png','short_response.png','normal_speed.png'],'method':'Parent view_image inspection: Korean labels and axes legible, no clipping. HTML links and structure checked; no full browser render claimed.','sections':len(text)})
    for p,h in f['protected'].items():assert sha(Path(p))==h
    c=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=O/'STATE.json';h=ROOT/'NEW_CHAT_HANDOFF.md'
    for p in (c,s,h):assert sha(p)==sha(U/'before'/p.name),f'Concurrent root edit: {p}'
    title='### 2026-10-22 순간 비교 보존·모습/변화량 분리 — 정보 보존 확인, 정상 속도 경고 증가로 운영 미적용'
    entry=title+'\n'+('\n'.join([
      '- 단위 `'+UNIT+'`. 공개 TRAIN3단어×9명27영상,2808비교/243개참고기준. 순간별 결과/expected/nativeunknown 정확보존. 연속근거는 별도 상태. C·T3-F·운영 DTW/웹 유지,학습/추론/새DP/검출/사용자·멘토접근0.',
      '- 이전동결descriptor/정렬/관측조건 재사용. 앞·중간·뒤3모습+두변화량을5항목별 거리·개인차최대범위로 분리. q/r제외7명(자기대조8명),최소5명 동일. 21각도/위치별 단위 유지. 단어별 예외/문턱튜닝 없음. whole-action 판정없음.',
      '- 정상1080쌍 순간비교1148965칸 그대로,연속718558칸,순간만430407칸; 그중 기존차이36116칸 보존. 전체고정분모1638000. 각도순간61.9%/연속31.4% 분리. 새 검출복구/인식률개선 아님.',
      '- 짧은삭제81변형의≥5참고 새차이 point30/combined31/levels35/changes73/split70;≥1은63/63/64/81/81. 변화량 민감도증가이나 의미상오류정답률아님. 원본부터차이있으면new로세지않아changes73>split70가능.',
      '- 정상속도864쌍 공통567457칸 새차이 point8213/combined12118/levels17244/changes39252/split40685. 같은원본known시간변형 공통82192칸 combined22→split552. 정상속도허용개선미입증,추가경고자동오류판정미채택.',
      '- 손내림역할/고정범위 및135누락대조 원시간표불변. 정상손내림전후 연속보류는남지만 순간정보별도유지. 교차12프레임 각도360/손위치48칸보류,팔꿈치48/손목48비교가능및자기대조차이0유지. 모든자연교차정답성/반복일반화/별도새사람검증아님.',
      '- freeze58051/run38549/verify47400/analysis47708 exit0. 같은Codex 별도중심반경계산검산(기존descriptor재생공유),2808/243 PASS·합성5조건·집계·3그림·보호32핀 확인. 독립에이전트/수어전문가 검토아님. 현재실행없음,완료실험재실행금지. 다음 공동검토 후 정상시간대조 변화량 경고 원인진단계획;추가학습은 공동확인후.'
    ]))+'\n\n'
    ct=c.read_text('utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n';assert ct.count(h8)==ct.count(h9)==1
    ct=ct.replace(h8,h8+'\n'+entry,1).replace(h9,h9+'\n### 2026-10-22 순간·연속 근거 분리 결과 공동 검토\n\n'+NEXT+'\n\n',1);c.write_text(ct,encoding='utf-8')
    st=read(s);st['updated_date_kst']=now();st['current_stage']='WORD100_POINT_SEQUENCE_SPLIT_VERIFIED_DIAGNOSTIC_ONLY';st['next_single_action']=st['resume_command']=NEXT
    st[UNIT]={'status':'POINT_PRESERVED_COMPONENTS_SEPARATED_NORMAL_SPEED_REGRESSION_NO_DEPLOYMENT','at':now(),'public_clips':27,'words':3,'people':9,'cases':2808,'calibrations':243,'point_cells_preserved':1148965,'sequence_cells':718558,'point_only_cells':430407,'point_only_prior_differences_preserved':36116,'short_five_refs':a['variants']['cross']['short']['counts'],'normal_speed_shared':a['cohorts']['cross']['speed']['shared_incremental'],'training':0,'runtime_change':0,'user_data_access':0,'report':str(U/'RESULT_KO.html'),'next':NEXT};save(s,st)
    hand='# WordsInSilence 최신 인수인계\n\n갱신 '+now()+'\n\n현재 목표: 순간 비교 정보를 보존하면서 연속 비교 근거와 모습/변화량의 차이를 구분하는 DTW 연구. TCN 학습 아님.\n마지막 완료: '+UNIT+' 구현·공개검증·별도수치검산·보고. 순간정보 보존은통과,새변화량경고는정상속도에도 증가해운영미적용.\n다음 한 작업: '+NEXT+'\n\n'+entry+'담당: Codex 로컬 구현/실행 및 별도 검산 코드. 이전 검증된 descriptor 재생을 공유하며 별도 에이전트/수어전문가 검토로 주장하지 않음.\n실행/잠금 없음. 완료실험 재실행 금지. 원본·이전동결결과·C/T3F/웹 읽기전용 유지. 사용자·멘토 자료 접근0.\n근거: '+str(U)+' / RESULT_KO.html/txt, FROZEN, SELFTEST, VERIFY, ANALYSIS, AGGREGATE_CHECK, VISUAL_REVIEW, FINAL_AUDIT, SYNC_CHECK.\n';h.write_text(hand,encoding='utf-8')
    for p in (c,h):assert UNIT in p.read_text('utf-8') and NEXT in p.read_text('utf-8')
    assert read(s)['current_stage']==st['current_stage']
    save(U/'SYNC_CHECK.json',{'status':'PASS','at':now(),'hashes':{str(p):sha(p) for p in (c,s,h)}})
    save(U/'FINAL_AUDIT.json',{'status':'PASS','at':now(),'training':0,'runtime_changes':0,'user_data_access':0,'new_dtw':0,'protected':32,'verification':'same-author separate arithmetic, shared frozen descriptor dependency','sessions':{'freeze':58051,'run':38549,'verify':47400,'analysis':47708},'exit_codes':0,'active':False,'adopted_for_automatic_error':False,'report_hashes':{p.name:sha(p) for p in (U/'RESULT_KO.html',U/'RESULT_KO.txt')}})
    print('REPORT_AND_SYNC_PASS',flush=True)

if __name__=='__main__':main()
