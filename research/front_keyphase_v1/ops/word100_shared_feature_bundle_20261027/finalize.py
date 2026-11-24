"""Synchronize only this verified preparation unit; keep previous records."""
import json,re,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    result=read(U/'RESULT.json'); final=read(U/'review/FINAL_REVIEW.json');ui=read(U/'UI_VERIFY.json')
    assert final['status']=='PASS_WITH_SCOPE_LIMITS' and not final['errors']
    assert all(final['checks'].values()) and ui['status']=='PASS'
    for name,rel in [('WordsInSilence_PROJECT_CONTEXT.md','WordsInSilence_PROJECT_CONTEXT.md'),('NEW_CHAT_HANDOFF.md','NEW_CHAT_HANDOFF.md'),('STATE.json','research/front_keyphase_v1/ops/STATE.json')]:
        assert sha(U/'before'/name)==sha(ROOT/rel),'Concurrent root change: '+rel
    now=datetime.now(timezone(timedelta(hours=9))).isoformat();unit=U.name
    summary=('공통 특징의 다른 표현 전이 평가를 준비했다. 기존 멀다·믿다·감사·별4표현에 수어·화상·마음에들다·실명·순종·잘하다·조상·이집트8표현을 추가해 기록ID01~03 총36개 공개TRAIN영상을 고정했다. 새24개384RGB와48시트 생성, 이전12개192RGB재사용. 전체576RGB표본을 시트로 검토하되 정적주석은 사전고정5/8/11표본108시점에만 작성했다. 오른/왼검지단독폄·오른/왼네손가락폄·오른/왼손얼굴2D겹침·양손2D겹침7항목756칸을 root와별도Codex가 각각기록했다. 양쪽확정일치653칸, 보류103칸(둘다unknown56/한쪽unknown46/0대1충돌1). 순서변화72손항목은57확정초안/15보류, 표본사이움직임·정확횟수·필수단계로확장하지않았다. 공통특징최소구성7중5통과이며 왼검지fit양성0·왼손얼굴fit양성1표현만으로2항목부족. 결과보고평가자료이동/임계값조정없음. 이미지보기전fit16/다른사람8/다른표현같은ID8/다른표현다른ID4분리고정. 모두과거TCN TRAIN이므로 C/T3F신규평가아님. 별도코드로756+72칸·분모·분리·576RGB해시·47입력결과핀·32운영핀검산PASS. HTML36영상선택/180시점/7gallery/desktop·390px검증PASS. 같은AI관찰초안이지독립사람정답아님; 정답검증완료0/학습0/운영교체0/사용자·멘토접근0.')
    nxt=('다음 한 작업은 최소구성이 갖춰진5개 공통시각특징의 학습용·검증용 정답 신뢰도를 확보하고 작은관측모델시험으로넘길수있는지확정하는것이다. 현재653칸은두Codex일치초안이며독립정답으로승격금지. 기존공개특징정답과대응할수있는항목을먼저찾고, 없다면추가무한검출기실험대신불일치·경계·각항목양음성의작은독립검토묶음을통해필요한확인범위를줄인다. 모든수어의교육규칙수동검토와물리적시각라벨검토를구분. 현재12표현분리와3고정시점을성과에맞춰옮기지말고, 부족한왼검지/왼얼굴은별도후속추가자료계획으로분리. 조건충족시관측모델입력은RGB·관절·시간뿐,목표ID/Gloss/사전정답금지;사용자·멘토는학습튜닝선택금지. C/T3F임베딩은이미해당표현에노출되었으므로새표현일반화평가에사용하면그한계명시. 정확접촉/가려진손가락/허용위치/교육적정오는미확정유지. 단위준비완료이며학습실행없고활성실행없음. 승인된일반준비/검증은재승인없이가능하나정답없는학습을완료로보고하지않는다.')
    evidence=f'research/front_keyphase_v1/ops/{unit}/RESULT_KO.html'
    title='2026-10-27 공통 시각특징 12표현·36영상 주석 및 분리평가 준비'
    ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'; text=ctx.read_text(encoding='utf-8-sig'); assert title not in text
    for section,body in [(8,summary),(9,nxt)]:
        text,n=re.subn(rf'(^## {section}\. [^\n]*\n)',lambda m:m.group(1)+f'\n### {title}\n{now}\n{body}\n근거: {evidence}, FROZEN.json, RUBRIC.json, DRAFT_LABELS.json, LEARNING_MANIFEST.json, review/FINAL_REVIEW.json, UI_VERIFY.json.\n\n',text,count=1,flags=re.M)
        assert n==1
    ctx.write_text(text,encoding='utf-8')
    statepath=ROOT/'research/front_keyphase_v1/ops/STATE.json';s=read(statepath)
    s['updated_date_kst']=now;s['current_stage']='SHARED_FEATURE_12_EXPRESSION_BUNDLE_VERIFIED_INDEPENDENT_LABEL_TRUTH_PENDING';s['next_single_action']=nxt
    s[unit]={'status':result['status'],'at':now,'summary':summary,'next':nxt,'evidence':evidence,'counts':result['counts'],'role_counts':result['roles'],
             'implementation':'Codex root + local_feature_inventory; image drafts root + ksl_source_audit + linguistic_eval_review',
             'verification':'Separate Codex script arithmetic and decoded-RGB hash checks; same AI family, not independent human/expert truth.',
             'training_runs':0,'active_execution':None,'authority':'User requested next task after source-token report; existing autonomous preparation authorization retained.'}
    save(statepath,s)
    handoff=f'''# WordsInSilence 최신 인수인계

{now}

## 현재 목표와 마지막 완료
{summary}

## 다음 한 작업
{nxt}

## 실행·검증 구분
- 구현: root(동결분리·루브릭·36개 이미지초안·집계·뷰어), local_feature_inventory(새24영상 RGB추출), ksl_source_audit(기존12영상별도주석), linguistic_eval_review(새24영상별도주석·최종별도산술검산).
- 정적 108시점×7=756칸과 시퀀스36×2=72칸을 분리. unknown은 negative가 아님. 손모양과 2D관계는 서로별도.
- 접촉/정확관절각/표현정오/허용범위가 아니라 물리적으로 보이는 큰특징의 AI초안. 완전맹검·독립사람검증 아님.
- 최소구성통과5개: 오른검지단독폄,오른/왼네손가락폄,오른손얼굴투영,양손투영. 왼검지fit양성0,왼얼굴fit양성은조상1표현뿐. 현재분리유지.
- 시퀀스움직임은16표본안관찰이며준비/마무리변화포함가능. dense주석,연속단계경계,정확반복횟수로승격금지.
- 새24개384원본RGB+48시트 생성;8영상은기존캐시,16영상은동결ZIP멤버71MB복사. 다운로드없음. 이전12개192RGB재추출없음.
- 별도검산 최초 해시영역문제는 review/FIRST_VERIFICATION_HASH_DOMAIN_ERROR.json 보존. 파일컨테이너해시와디코딩RGB해시구분후576개검산PASS. 데이터/초기라벨교체없음.
- 구현된 ROOT_VISUAL_LABELS_RAW.json, review/IMAGE_LABELS_FIRST_PASS.json 등최초판단보존. 한쪽unknown46,명시충돌1은최종unknown으로유지.
- 학습0,새검출/새모델추론0,운영교체0,사용자/멘토접근0. C·T3F·운영웹·DTW32핀불변.
- 활성서버/학습/자동화새실행없음. 추출·검산·UI검사종료완료. Git/외부전송없음.

## 근거
- `{evidence}`: 특징별있음/없음/보류사진,36영상슬라이더,분리와부족항목.
- `{unit}/DRAFT_LABELS.json`: 원판단2종·보류사유·원본시간/프레임 연결.
- `{unit}/LEARNING_MANIFEST.json`: 향후특징모델입력·역할·미확정항목,현재학습승인파일아님.
- `{unit}/FROZEN.json`, `RUBRIC.json`: 이미지검토전구성과정의고정.
- `{unit}/review/FINAL_REVIEW.json`: 별도23검사·576RGB해시·정적/시퀀스검산.
- `{unit}/UI_VERIFY.json`: 36영상180시점·7gallery·desktop/mobilePASS.

이번완료는 다음준비단위의완료이고,새특징모델학습완료나TCN·DTW정확도향상이아니다.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
    save(U/'CLOSEOUT.json',{'status':'COMPLETE_PREPARATION_UNIT','at':now,'counts':result['counts'],'root_docs_synced':True,'review_status':final['status'],
                          'ui_status':ui['status'],'active_execution':None,'training_runs':0})
    assert read(statepath)[unit]['counts']==read(U/'CLOSEOUT.json')['counts']
    assert (ROOT/evidence).is_file()
    print('Synchronized context §8/§9, STATE and latest handoff')

if __name__=='__main__':main()
