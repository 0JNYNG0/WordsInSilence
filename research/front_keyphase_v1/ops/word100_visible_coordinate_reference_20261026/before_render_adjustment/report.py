import html,json
from collections import Counter
from PIL import Image,ImageDraw
import prepare as S
U=S.U
def e(x):return html.escape(str(x))
def make_images(f,result):
    images=[];rows=result['rows']
    for clip in f['rows']:
        cid=clip['clip_id'];sheet=Image.new('RGB',(1920,1440),'#f5f7fb');sd=ImageDraw.Draw(sheet)
        sd.text((16,10),clip['label']+' | 새 동작 시점 / 표시 범위와 세 좌표 비교',font=S.font(25),fill='#17253b')
        for idx,t in enumerate(clip['frames']):
            rr=[r for r in rows if r['clip_id']==cid and r['frame']==t and r['region']]
            assert rr
            source=Image.open(U/rr[0]['source']).convert('RGB');pts=[]
            for r in rr:
                b=r['region'];pts.extend([(b[0],b[1]),(b[2],b[3])])
                pts.extend(tuple(v) for v in r['points'].values() if v is not None)
            crop=(max(0,int(min(p[0] for p in pts))-65),max(0,int(min(p[1] for p in pts))-65),min(1920,int(max(p[0] for p in pts))+65),min(1080,int(max(p[1] for p in pts))+65))
            for col,k in enumerate(['regions','A','B','C']):
                im=source.copy();d=ImageDraw.Draw(im)
                for r in rr:
                    color='#00ebff' if r['side']=='left' else '#ffae38'
                    d.rectangle(tuple(r['region']),outline=color,width=2)
                    label=('L' if r['side']=='left' else 'R')+str(r['joint'])
                    d.text((r['region'][0],r['region'][1]-18),label,font=S.font(14),fill=color)
                    if k!='regions' and r['points'][k] is not None:
                        x,y=r['points'][k];c={'A':'#ff5070','B':'#42a6ff','C':'#b375ff'}[k]
                        d.line([(x-6,y),(x+6,y)],fill='black',width=4);d.line([(x,y-6),(x,y+6)],fill='black',width=4)
                        d.ellipse((x-3,y-3,x+3,y+3),fill=c)
                part=im.crop(crop);part.thumbnail((464,380));sheet.paste(part,(col*480+8+(464-part.width)//2,112+idx*440))
                title={'regions':'RGB에서 표시한 위치 범위','A':'A · 제공 좌표 그대로','B':'B · 일정 이동','C':'C · 매 순간 손목 맞춤'}[k]
                sd.text((col*480+10,75+idx*440),f'{t}프레임 | {title}',font=S.font(18),fill='#17253b')
            sd.text((15,505+idx*440),'L=왼손 / R=오른손 · 0=손목 4=엄지 끝 8=검지 끝 · 보이지 않아 보류한 점은 판정 표시 없음',font=S.font(17),fill='#526174')
        name='images/'+cid+'_comparison.jpg';sheet.save(U/name,quality=95);images.append(dict(clip_id=cid,label=clip['label'],file=name))
    return images
def run():
    f=S.read(U/'FROZEN.json');r=S.read(U/'RESULT.json');v=S.read(U/'VERIFIED.json');catalog=S.read(U/'REFERENCE_CATALOG.json')
    assert v['status']=='PASS_SOURCE_AND_RECTANGLE_ARITHMETIC'
    assert v['result_sha']==S.sha(U/'RESULT.json') and v['annotation_sha']==S.sha(U/'ANNOTATIONS.json')
    images=make_images(f,r)
    findings={
      '1290':'감사: 15개 표시 지점 모두 불확실한 위치 범위를 감안해도 B가 원본보다 가까워졌다. 그러나 B가 범위 안에 들어간 것은 11/15개다. 큰 전체 밀림은 줄었지만 남은 작은 오차와 개별 손가락 문제까지 해결됐다는 뜻은 아니다.',
      '2029':'무산: B의 범위 내 지점은 1/10개다. 보정 후 손목 일부는 가까워져도 표시한 손끝 6개는 모두 범위 밖이다. 일정한 이동만으로 이 자료를 정답으로 사용할 수 없다.',
      '0003':'수어: 6/12 → 8/12개로 범위 내 지점은 늘었지만, 표시 위치의 불확실성을 고려하면 큰 개선이라고 확정할 수 없다. 자동 보정의 성공 사례로 확대하지 않는다.',
      '0185':'후원: 3/10 → 7/10개로 늘었다. 다만 근소한 차이여서 엄격한 거리 범위 비교에서는 개선 여부를 보류했다. 내려놓은 왼손과 동작 중 가려진 손목을 구분해 기록했다.',
      '0920':'추출: 5/9 → 4/9개로 줄었다. 겹친 시점의 왼손목과 구분이 어려운 손끝을 보류했다. 팔의 손목에 강제로 붙여도 손 모양 근거가 새로 생기는 것은 아니다.',
      '1545':'수영: 원본은 표시한 14개 모두 범위 안이었으나 B는 10개, C는 7개였다. 원래 잘 맞던 자료까지 일괄 보정하면 안 된다는 대조 사례다. 80프레임의 가려진 왼손목은 성공/실패 집계에 억지로 넣지 않았다.'}
    plan=[
      '이번 RGB 위치 범위·원본·이동 보정 비교표를 고정 평가 자료로 보존한다. 28개 원본 유지 검토, 12개 이동 보정 검토, 30개 보이는 위치 불일치, 38개 가림·불확실 항목을 별도로 관리한다. 이 표를 학습 정답으로 바로 사용하지 않는다.',
      '어깨·팔꿈치의 이동 차이가 여러 시점에서 일관되는 경우만 선별할 공통 검증 규칙을 준비한다. 단어명이나 사용자의 촬영 결과는 입력하지 않는다. 실제 보이는 손의 자료와 맞는지 확인하며, 원래 양호한 자료는 그대로 두는 조건을 포함한다.',
      '작은 차이는 표본의 표시 오차보다 클 때만 개선이라고 판단한다. 새로운 시점·다른 자료에서 정상 손·교차 보존을 확인한 뒤 참고 답안 사용 범위를 확대한다. 모든 단어의 모든 점을 수작업으로 고치는 방식으로 확장하지 않는다.',
      '보정된 좌표의 품질과 사용 가능한 범위가 확인된 다음에 필요한 다른 촬영 ID 자료 확보와 학습 비교를 진행한다. 손목 확인을 전체 손가락·3D·수어 정확성 확인으로 확대하지 않는다.'
    ]
    S.write('NEXT_PLAN.json',dict(status='VISIBLE_REFERENCE_CATALOG_READY_CORRECTION_NOT_ADOPTED',steps=plan,next_single_action='Freeze a common body-anchor consistency and preserve-original gate; validate against held RGB regions/new public times without word-specific tuning. Do not create training truth from MP agreement.',additional_download=False,training=False,operating_change=False))
    text='''공개 원본 영상 기반 좌표 보정 재검증
공개 TRAIN REAL02의6표현·새 동작시점18장. 이전 시각검토와 다른 시점이지만 동일한 기존6영상이며 새사람 검증이 아님.
손목·엄지끝·검지끝108개 항목 중70개는 RGB에서 위치 범위를 표시하고38개는 가림·흐림·손가락구분불가로 보류.
표시는 같은Codex의 시각검토에 의한 예비2D범위이며, 독립사람 정답이나 정확한3D관절 중심이 아님. 후보좌표를 보이지않게 표시했지만 이전 영상결과는 알고있으므로 엄격한블라인드검증도 아님.
범위 안의 좌표수: A원본28/70, B일정이동41/70, C매순간손목맞춤25/70. 수어인식률·관절정확도백분율로해석금지.
손목13/31→20/31(B), 엄지/검지끝15/39→21/39(B). B새범위내18개/새범위밖5개. 70개 전체중불확실성범위까지감안해B가더가까운21개, 나머지49개는큰차이판정보류.
감사0/15→11/15; 무산0/10→1/10; 수어6/12→8/12; 후원3/10→7/10; 추출5/9→4/9; 수영14/14→10/14.
원본유지 검토28개, 보정 검토12개, 보이는 위치 불일치30개, 가림/불확실38개로 참고목록구성.12개는교정완료정답이아니라추가검토후활용할후보점이다.
원본양호수영에서범위밖4개발생; 정상교차/손내림/모든사람의손모양보존을보장하지않는다. 6영상모두전체관절정답으로승인하지않음. 엄격한전체표본손목통과조건도0/6(부분개선이없다는뜻아님).
검산:원본/고정조건/108항목/거리/집계/32보호파일검사통과. 검산코드의빈엄지집계None처리오류를최초코드/로그보존후수정;동결좌표·표시범위·결과·조건변경없음.
기존C·T3F·E3/웹/DTW 변경0. 새학습·검출·다운로드0. 사용자/멘토촬영접근0. 이번C는위치보정방법이름이며TCN C모델과별개.
다음:
'''+ '\n'.join(f'{i+1}. {x}' for i,x in enumerate(plan))
    with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as w:w.write(text)
    table=''.join('<tr><td>'+e(c['label'])+'</td><td>'+str(c['summary']['all']['marked'])+'</td>'+''.join('<td>'+str(c['summary']['all']['conditions'][k]['inside'])+'</td>' for k in 'ABC')+'<td>'+str(c['summary']['all']['withheld'])+'</td></tr>' for c in r['clips'])
    details=[]
    for x in images:
        key=x['clip_id'].split('_WORD')[1][:4]
        details.append(f'<article id="WORD{key}"><h3>{e(x["label"])}</h3><p>{e(findings[key])}</p><a href="{x["file"]}" target="_blank"><img src="{x["file"]}" alt="{e(x["label"])} 실제 RGB 위치 범위와 제공 좌표 비교"></a><p class="small"><a href="images/{x["clip_id"]}_regions.jpg" target="_blank">예측 좌표 없는 위치 표시만 보기</a></p></article>')
    allrows=[]
    for z in r['rows']:
        values=[]
        for k in 'ABC':
            met=z['metrics'][k];values.append('판단 보류' if not z['region'] else ('좌표 없음' if met is None else ('범위 안' if met['inside'] else f'밖으로 {met["min_distance"]:.1f}px 이상')))
        allrows.append('<tr><td>'+e(z['label'])+'</td><td>'+str(z['frame'])+'</td><td>'+('왼손' if z['side']=='left' else '오른손')+'</td><td>'+{0:'손목',4:'엄지 끝',8:'검지 끝'}[z['joint']]+'</td>'+''.join('<td>'+x+'</td>' for x in values)+'</tr>')
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 영상으로 확인한 좌표 보정</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#172a43;font:16px/1.8 "Malgun Gothic",sans-serif}main{max-width:1180px;margin:auto;padding:42px 26px}header{background:#183653;color:#fff;border-radius:20px;padding:34px}h1{font-size:32px;line-height:1.4}h2{font-size:24px;margin:0 0 18px}h3{font-size:22px;margin-top:0}.tag{color:#afdbec;font-size:13px;letter-spacing:.05em}.lead{font-size:20px}section,article{background:#fff;border:1px solid #d8e2ee;border-radius:18px;padding:27px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{background:#edf3f9;padding:20px;border-radius:12px}.n{display:block;font-size:32px;font-weight:700;color:#205276}.small{font-size:14px;color:#51647b}.note{background:#fff6e7;border-left:5px solid #d2912c;padding:16px 20px}.ok{background:#eaf6f3;border-left:5px solid #348e7d;padding:16px 20px}a{color:#1a5ab2;overflow-wrap:anywhere}header a{color:#b9e4ff}nav{display:flex;flex-wrap:wrap;gap:20px}img{width:100%;height:auto;border-radius:10px;border:1px solid #d8e2ee;display:block}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:12px;border-bottom:1px solid #dce4ed}th{background:#edf3f8}.scroll{overflow:auto}li{margin:12px 0}summary{font-weight:bold;cursor:pointer}@media(max-width:640px){main{padding:16px 12px}header,section,article{padding:20px}.grid{grid-template-columns:1fr}h1{font-size:25px}.lead{font-size:18px}th,td{padding:8px;font-size:13px}}
    </style><main><header><div class="tag">PUBLIC RGB CHECK · 2026.10.26</div><h1>실제 손을 기준으로 확인하니<br>어떤 좌표를 보정할 수 있었나?</h1><p class="lead">일정한 위치 보정의 효과는 실제 영상에서도 확인됐다.<br>다만 모든 자료와 모든 손가락을 한꺼번에 고칠 수는 없었다.</p><nav><a href="#result">확인 결과</a><a href="#visual">비교 화면</a><a href="#next">다음 작업</a></nav></header>
    <section><h2>이번에는 MediaPipe를 답으로 삼지 않았다</h2><p>공개 영상에서 손목·엄지 끝·검지 끝이 실제로 보이는 위치를 먼저 표시했다. 그다음 기존 세 방식의 좌표가 그 범위 안에 들어오는지 확인했다. 예측 좌표를 위치 표시 화면에 띄우지 않았고, 표시를 고정한 뒤 수치를 계산했다.</p><div class="grid"><div class="card"><span class="n">18장면</span>6표현의 새로운 동작 시점<br>기존 공개 TRAIN REAL02 영상</div><div class="card"><span class="n">70항목</span>실제 위치 범위를 표시할 수 있었음<br>손목31 · 엄지20 · 검지19</div><div class="card"><span class="n">38항목</span>가림·흐림·손가락 구분 불가<br>맞음/틀림을 억지로 판정하지 않음</div></div><p class="note">위치 범위는 내가 RGB를 보고 표시한 <b>예비 2D 기준</b>이다. 독립적인 사람의 정답 주석이나 정확한 관절 중심은 아니다. 이전 영상의 결과를 알고 있으므로 완전히 눈가림한 검증도 아니다. 이미 본 6영상의 다른 시점을 사용했으며 새 사람에 대한 검증은 아니다.</p></section>
    <section id="result"><h2>위치 범위 안에 들어온 좌표</h2><div class="grid"><div class="card"><span class="n">28 / 70</span>A · 제공 좌표 그대로</div><div class="card"><span class="n">41 / 70</span>B · 영상 전체에 일정 이동</div><div class="card"><span class="n">25 / 70</span>C · 매 순간 팔 손목에 맞춤</div></div><p>이 숫자는 <b>표시한 위치 범위 안에 들어온 항목 수</b>다. 수어 인식률이나 손가락 검출 정확도 백분율이 아니다. 이번 C는 비교 방법 이름이며 기존 TCN C 모델과 별개다.</p><div class="scroll"><table><thead><tr><th>표현</th><th>표시 가능</th><th>A 원본</th><th>B 일정 이동</th><th>C 손목 맞춤</th><th>보류</th></tr></thead><tbody>'''+table+'''</tbody></table></div><p class="ok"><b>감사:</b> 원본 0/15개에서 일정 이동 후 11/15개가 표시 범위 안에 들어왔다. 불확실한 범위 안에서 어느 점을 실제 위치로 보더라도 15개 모두 원본보다 가까워졌다. 전체 좌표가 밀린 문제를 줄였다는 근거가 더해졌다.</p><p class="note"><b>무산:</b> 보정 후에도 1/10개만 범위 안이었다. 특히 손끝 6개 모두 범위 밖이었다. <b>수영:</b> 원본은 14/14개였지만 보정 후 10/14개로 줄었다. 보정이 필요 없는 자료까지 옮기면 안 된다.</p><p>전체에서는 B로 새롭게 범위 안에 들어온 항목 18개, 반대로 밖으로 나간 항목 5개가 있었다. 표시 범위의 불확실성까지 고려한 엄격한 거리 비교에서는 21개가 분명히 가까워졌고, 나머지 49개는 개선·악화를 확정하지 않았다. 경계 안팎의 작은 변화와 큰 위치 개선을 구분한 결과다.</p></section>
    <section><h2>‘오답노트’에 바로 넣을 수 있나?</h2><p>검토할 목록은 만들었다. 하지만 지금의 예비 표시만으로 학습 정답으로 승인하지 않았다.</p><div class="scroll"><table><thead><tr><th>분류</th><th>수</th><th>활용</th></tr></thead><tbody><tr><td>원본 유지 검토</td><td>28</td><td>이미 보이는 위치 범위 안에 있는 원본을 보존</td></tr><tr><td>일정 이동 보정 검토</td><td>12</td><td>원본은 밖이고, 보정 후 안으로 들어오며 엄격한 거리 비교도 개선된 점</td></tr><tr><td>보이지만 좌표 확인 부족</td><td>30</td><td>범위 밖이거나 작은 변화여서 확정하지 못한 점</td></tr><tr><td>가림·불확실</td><td>38</td><td>보이지 않는 손가락을 정답처럼 만들어 넣지 않음</td></tr></tbody></table></div><p>12개 보정 검토 점은 감사 11개와 무산 1개다. 단어별 예외를 만든 것이 아니라 <b>모든 표시 항목에 같은 규칙</b>을 적용한 결과다. 이 표본에 맞춰 전체 단어의 자동 보정 규칙이 검증된 것은 아니다.</p><p>현재는 6영상 중 어느 것도 전체 손가락 정답으로 승인하지 않았다. 엄격하게 미리 정한 ‘양손의 여러 시점 손목이 모두 범위 안이며 충분히 개선’ 조건도 통과한 영상은 없었다. 부분 개선은 확인됐지만 전체 클립의 완전한 교정을 보장하지는 못했다.</p></section>
    <section id="visual"><h2>실제 화면에서 보기</h2><p>사각형은 원본 RGB에서 먼저 표시한 위치 범위다. 왼손은 청록색, 오른손은 주황색이며, 숫자 0은 손목·4는 엄지 끝·8은 검지 끝이다. 각 조건의 점이 사각형과 실제 피부 위 어디에 놓이는지 비교하면 된다.</p><p>첫 칸은 위치 범위만 표시한 원본, 다음은 A·B·C의 좌표다. 판단을 보류한 점은 판정 표시를 하지 않았다. 확대는 설명을 위한 화면이며 검출 범위를 바꾼 것이 아니다.</p></section>'''+''.join(details)+'''
    <section id="next"><h2>다음 단계</h2><ol>'''+''.join('<li>'+e(x)+'</li>' for x in plan)+'''</ol><p class="note">사용자의 손을 정답 위치로 강제로 끌어다 맞추는 방식은 아니다. 같은 원본 영상과 제공 좌표의 어긋남을 정리하는 작업이다. 사용자에게 피드백할 실제 위치·모양 차이는 남겨야 한다.</p></section>
    <section><h2>검증과 보존</h2><ul><li>108개 항목의 원본 연결·좌표·거리·집계를 별도 구현으로 검산했다. 38개 보류도 분모와 기록에 남겼다.</li><li>기존 좌표 배열을 수정하지 않았다. 원본·C·T3-F·E3·웹·DTW 등 32개 보호 파일 해시를 확인했다.</li><li>새 학습·새 검출·다운로드·운영 변경·사용자/멘토 촬영 접근은 모두 0이다.</li><li>검산 코드에서 비어 있는 엄지 집계를 처리하는 오류를 발견해 최초 코드와 오류를 보존한 뒤 수정했다. 위치 표시·결과·판정 기준은 변경하지 않았다.</li></ul><p class="small"><a href="ANNOTATIONS.json">고정 위치 표시</a> · <a href="REFERENCE_CATALOG.json">참고 목록</a> · <a href="VERIFIED.json">별도 검산</a> · <a href="FROZEN.json">고정 조건</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p><details><summary>108개 항목 전체 보기</summary><div class="scroll"><table><thead><tr><th>표현</th><th>프레임</th><th>손</th><th>지점</th><th>A</th><th>B</th><th>C</th></tr></thead><tbody>'''+''.join(allrows)+'''</tbody></table></div></details></section></main></html>'''
    with (U/'RESULT_KO.html').open('x',encoding='utf-8') as w:w.write(doc)
    names=['FROZEN.json','PREPARED.json','ANNOTATIONS.json','RESULT.json','VERIFIED.json','REFERENCE_CATALOG.json','NEXT_PLAN.json','RESULT_KO.html','RESULT_KO.txt','marks.py','prepare.py','annotate.py','evaluate.py','verify.py','report.py']
    S.write('REPORT_MANIFEST.json',dict(files={str(U/n):S.sha(U/n) for n in names},images={str(U/x['file']):S.sha(U/x['file']) for x in images}))
    print('REPORT_COMPLETE',len(images),flush=True)
if __name__=='__main__':run()
