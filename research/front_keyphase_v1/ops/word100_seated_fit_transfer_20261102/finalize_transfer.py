from prepare_seated import *
from sync_progress import sync
from collections import Counter
DEST=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102')

def main():
    assert not (L/'FINAL.json').exists();v=read(L/'VERIFIED.json');a=v['conditions']['original_replay'];b=v['conditions']['seated_variation'];old=v['baseline_saved_outputs']
    before=read(DEST/'evaluations/original_replay.json');after=read(DEST/'evaluations/seated_variation.json')
    nn={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'};aa={(p['key'],p['frame']):p for p in before['points'] if p['kind'] in nn and p['y']<8}
    bb={(p['key'],p['frame']):p for p in after['points'] if p['kind'] in nn and p['y']<8};assert aa.keys()==bb.keys()
    changes=[]
    for key in aa:
        x,y=aa[key],bb[key]
        if x['pred']!=y['pred']:changes.append(dict(key=key[0],frame=key[1],word=x['word'],kind=x['kind'],truth=x['y'],old=x['pred'],new=y['pred'],fixed=x['pred']!=x['y'] and y['pred']==y['y'],regressed=x['pred']==x['y'] and y['pred']!=y['y']))
    counts=Counter(('fixed' if r['fixed'] else 'regressed' if r['regressed'] else 'changed_wrong') for r in changes)
    exclusive(L/'CHANGE_AUDIT.json',dict(at=clock(),counts=dict(counts),rows=changes,source='Saved matched condition outputs; diagnostic only, no threshold or candidate selection.'))
    # Store a Korean continuation in the canonical project documents.
    nextstep='완료L2학습176비교/J/K/F/E/D재실행금지. CAUSE_AUDIT에서원행동오류28→26이나준비복귀판정이맞은주요14시점을가림(대조3);14중12가해당행동가족의첫직접정답이며중복변형을접으면5원본시점이다. 다음단일작업은공개FIT의주요진입경계/준비복귀감독과관측·역할의동시성계약을감사하는것. 원관측보존,단일시점오류와행동전체누락구분,짧은역순/돌아오는동작을함께평가할범용조건을결과전동결후필요한지도학습보완을한다. 카메라변이만더늘리거나DEV기반문턱완화/정답순서강제/짧은반대근거삭제금지. 전체기능·전체처리시간·격리데모통과전운영교체없음. C/T3F/기존웹/H14·15/사용자/멘토보존. W별도파생경로사용/원본읽기전용.'
    summary=f"L공개36웹캠시야입력검증후원본재생/시야변이동일량2조건80epoch학습·176개발비교·별도검산완료. 주요시점{a['major']['correct']}→{b['major']['correct']}/{b['major']['known']},그중seated{a['seated']['correct']}→{b['seated']['correct']}/{b['seated']['known']},native{a['native']['correct']}→{b['native']['correct']}/{b['native']['known']}. 순서상태{a['order']}→{b['order']}/21,깨끗한순서{a['clean_order']}→{b['clean_order']},짧은역순{a['local_reverse']}→{b['local_reverse']}/3. 판정{v['status']}. 신규운영적용없음/전체goal active. C공간부족중단19보존후남은17은W별도파생폴더에서완료."
    times=[r['seconds'] for r in read(L/'INPUT_MANIFEST.json')['rows']]
    text=['공개 학습자료의 웹캠 화면 조건 보완 결과',clock(),'',
        '이번 작업은 수어 단어를 맞추는 TCN의 교체가 아니라, 피드백용 신경망이 주요 동작과 동작 순서를 읽는 부분의 학습 비교다.',
        '왜 했나: 공개 자료는 아래쪽 손이 계속 보이는 넓은 화면이 많고, 책상 앞 웹캠은 손을 내리면 화면 밖으로 사라진다. 같은 행동이 화면 구성 때문에 다르게 분석되는 문제를 줄이기 위한 시험이다.',
        '무엇을 했나: 공개 학습 ID01~09의4표현36영상에 이미 정한 몸 기준 화면 변이를 만들었다. 실제 사용자 화면에 제한선을 추가한 것이 아니다. 36개 모두 원래 프레임/시각을 보존했고, 주요정답706시점에서 추가잘림0을 확인했다. 9ID의선택화면을별도시각확인했다. 이것이 모든 프레임의 손가락 정확도나 전문가 정답을 보증하지는 않는다.',
        '학습방법: 같은 기존 신경망 가중치에서 시작하여, 기존216입력+추가36입력의252개를 두 조건에서 같은80epoch 학습했다. 추가36개만 원래화면/웹캠시야로 달랐다. 정답과학습량은같다. SignRep/RGB특징처리·기존관측은고정하고행동해석/과정판정부를미세조정했다. 잘려서 보이지 않는 대기 손모양은 추가 정답에서 제외했다.','',
        '같은 새 평가 조건으로 비교',
        '항목 | 원래 화면을 더 배움 | 웹캠에 가까운 화면을 더 배움',
        f"주요 동작을 맞춘 시점 | {a['major']['correct']}/{a['major']['known']} | {b['major']['correct']}/{b['major']['known']}",
        f"전체화면 새 사람 조건 | {a['native']['correct']}/{a['native']['known']} | {b['native']['correct']}/{b['native']['known']}",
        f"웹캠시야·속도 등 정상조건 | {a['seated']['correct']}/{a['seated']['known']} | {b['seated']['correct']}/{b['seated']['known']}",
        f"순서 확인 상태 | {a['order']}/21 | {b['order']}/21",
        f"순서 확인 중 주요오류/기존핵심관측보류도 없는 사례 | {a['clean_order']} | {b['clean_order']}",
        f"짧게 순서를 뒤집은 시험 탐지 | {a['local_reverse']}/3 | {b['local_reverse']}/3",
        f"역할 판단 때문에 잃은 원래 맞은 주요시점 | {len(a['correct_action_losses'])} | {len(b['correct_action_losses'])}",
        f"준비 여부를 적용하기 전 동작 자체의 정답 | {a['raw_major']['correct']}/615 | {b['raw_major']['correct']}/615",
        f"대기 정답 시점 | {a['normal_rest_correct']}/107 | {b['normal_rest_correct']}/107",
        f"별도미세조정전 저장결과의 같은입력집계 | {old['major']['correct']}/{old['major']['known']}; 순서{old['order']}/21",
        f"두 새 조건 사이 고친 시점{counts['fixed']}, 새로 틀린 시점{counts['regressed']}. 총점만으로 개선을 판정하지 않았다.",'',
        '원인을 더 나눠 확인한 결과',
        '손동작 자체를 읽은 정답은587→589로소폭늘었다. 그런데그중14시점을과정판정부가준비/복귀로잘못처리해최종575가됐다. 대조조건에서는이손실이3개였다. 즉이번전체저하는손이새로미검출돼서가아니라,맞게읽은동작을주요동작에서빼버리는해석이악화된영향이크다.',
        '14시점중12개는해당동작가족의첫직접정답이다. 속도변형등반복시험을원본시점으로모으면5곳이다. 따라서14개의서로독립된동작전체를놓쳤다는뜻은아니다. 그래도중요동작의시작을준비로취급하는문제는바로잡아야한다.',
        '조상양손동작의원판정은175/175로남았으나그중2시점을준비로처리했다. 정상교차보존을완전히통과했다고보고하지않았다. 순서확인6개중다른주요오류/보류도없는것은1개뿐이므로순서성능개선으로채택하지않았다.','',
        '남은 제한',
        '이 수치는 TCN 단어인식률이 아니라 정답을 붙여 둔 주요동작 시점의 분류다. 학습과 분리한공개ID11~13을 사용했지만 이미여러번분석한개발자료이므로독립최종평가가아니다.',
        '순서상태가 확인돼도 다른 중요한 시점이 틀리거나 근거가 부족할 수 있다. 따라서 모든수어가맞다는판정으로사용하지않는다. 조상의교대움직임은현재단일행동가족이므로반복세부순서/횟수까지검증된것이아니다.',
        '영상누락을정상으로확정하지않았고176출력의시간범위와원관측을검산했다. 중요한이동을지운시험에대한기존이동미확인근거도보존했다. 새입력의전체피드백웹연결은아직완료로부르지않는다.',
        '통과 여부: '+v['status']+'; 미통과항목: '+', '.join(k for k,value in v['gates'].items() if not value),
        f"현재입력추출은약{min(times):.1f}~{max(times):.1f}초. 새판정부만의중앙시간은{b['head_median_seconds']:.4f}초다. 이것을전체결과대기시간이라고부르면안된다. 전체처리속도개선은이번시험으로입증하지않았다.",
        '기존C·T3-F·웹기본은보존했다. 사용자/멘토촬영은접근·학습·튜닝·선택에쓰지않았다. 새로운모델/데이터다운로드없음.','',
        '중단·복구: C드라이브1GiB예비공간안전장치로19개추출후멈췄다. 최초오류/완료결과를보존하고남은17개파생결과와새모델만여유있는W:/WordsInSilence_derived에썼다. W:/sources와기존결과를이동/삭제하지않았다.',
        '구현과별도산술검산은같은Codex가수행했다. 독립수어전문가검토와같지않다.',
        '', '다음 작업: '+nextstep]
    (L/'RESULT_KO.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
    exclusive(L/'FINAL.json',dict(at=clock(),status=v['status'],result=str(L/'RESULT_KO.txt'),verified=str(L/'VERIFIED.json'),new_training=2,new_comparisons=176,production_changed=False,next=nextstep,whole_goal_complete=False))
    write(L/'CURRENT_EXECUTION.json',dict(at=clock(),status='COMPLETE_COMPARISON',active=None,train_session=22885,train_exit=0,whole_goal_complete=False,next=nextstep))
    sync('웹캠시야 학습2조건 비교·검산 완료',summary,nextstep,'SEATED_FIT_COMPARISON_COMPLETE',None)
    exclusive(L/'FINAL_VERIFICATION.json',dict(at=clock(),status='PASS',report_exists=(L/'RESULT_KO.txt').exists(),whole_goal_complete=False))
    print(summary)
if __name__=='__main__':main()
