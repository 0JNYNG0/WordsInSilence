"""Create an isolated reviewable demo; never rewrite prior demo files."""
import copy,numpy as np
from common import G,V,R,U,P,read,sha,write,exclusive
from learner_adapter import qualify

def new(name,text):
    with (G/name).open('x',encoding='utf8') as f:f.write(text)

def main():
    assert read(G/'DIAGNOSTIC_VERIFICATION.json')['short3_center_performance_preserved']
    server=(V/'run_demo.py').read_text(encoding='utf8').replace('from demo_adapter import','from learner_adapter import').replace('PORT=8806','PORT=8807')
    server=server.replace("U/'private_sessions'/jid/'analysis'","Path(job.get('analysis_directory',str(U/'private_sessions'/jid/'analysis')))")
    server=server.replace("('rgb_geometry','geometry_only')","('learned_motion',)")
    marker="   if route.startswith('/media/control/'):"
    assert marker in server
    server=server.replace(marker,"""   if route.startswith('/api/order/'):
    w=route.rsplit('/',1)[-1]
    if w not in WORDS:return self.json({'error':'Unknown word'},404)
    return self.json(read(U/'demo_order'/w/'RESULT.json'))
   if route.startswith('/media/order/'):
    w=route.rsplit('/',1)[-1]
    if w not in WORDS:return self.json({'error':'Unknown word'},404)
    return self.file(U/'order_videos'/f'{w}_reverse_main.mp4','video/mp4')
"""+marker)
    server=server.replace("if u.path!='/api/analyze':return self.json({},404)","if u.path not in ('/api/analyze','/api/analyze-public'):return self.json({},404)")
    marker="   try:length=int(self.headers.get('Content-Length','0'))"
    assert marker in server
    server=server.replace(marker,"""   if u.path=='/api/analyze-public':
    with LOCK:
     if not STATE['ready'] or (FUTURE is not None and not FUTURE.done()):return self.json({'error':'분석이 끝난 뒤 다시 시도해 주세요.'},409)
     jid=uuid.uuid4().hex;d=U/'public_runtime_runs'/jid;d.mkdir(parents=True,exist_ok=False)
     write(d/'INTENT.json',dict(source=str(PUBLIC[word]),word=word,condition=condition,training=False,public_only=True))
     JOBS[jid]=dict(id=jid,status='running',progress={'stage':'starting'},mime='video/mp4',analysis_directory=str(d/'analysis'),public_only=True)
     FUTURE=POOL.submit(perform,jid,PUBLIC[word],word,condition)
    return self.json({'id':jid},202)
"""+marker)
    new('run_demo.py',server)
    html=(V/'demo.html').read_text(encoding='utf8').replace('손모양과 신체 관계를 함께 확인해요','배운 손모양·관계·이동을 내 영상에서 확인해요')
    html=html.replace('엄지·손바닥 방향·접촉·짧은 동작 누락·반복 횟수는 아직 충분히 검증하지 못했습니다.',
        '손모양 일부·신체 관계 3종·이동 방향 5종을 확인합니다. 전체 동작 순서·엄지·손바닥 방향·정확한 접촉·필수 반복 횟수는 아직 충분히 검증하지 못했습니다.')
    html=html.replace('<button id="publicCut">주요 구간을 뺀 공개 시험</button>',
        '<button id="publicCut">주요 구간을 뺀 공개 시험</button><button id="publicOrder">순서를 바꾼 공개 시험</button><button id="publicFresh">공개 영상 새로 분석</button>')
    old='기존 손모양과 학습된 시간 흐름은 보존하고, 신체 관계를 읽는 작은 판정부를 추가했습니다. 아래 선택은 이 관계 판정부가 영상 특징도 함께 사용할지 비교합니다. C·T3-F와 이전 데모는 그대로입니다.'
    html=html.replace(old,'공개 기록 ID01~09의 영상으로 학습한 손모양·관계·시간 흐름과, 영상 특징을 직접 읽는 이동 판정부를 함께 사용합니다. 새 영상 한 편을 학습된 단어 흐름과 한 번 정렬합니다. C·T3-F와 이전 데모는 보존했습니다.')
    html=html.replace('<option value="rgb_geometry">신체 관계: 영상 + 관절</option><option value="geometry_only">신체 관계: 관절만 · 비교용</option>',
        '<option value="learned_motion">학습된 손모양·관계·이동 · 연구 후보</option>')
    new('demo.html',html)
    js=(V/'demo.js').read_text(encoding='utf8')
    js=js.replace('const statusNames={',"const statusNames={IMPORTANT_MOTION_UNCONFIRMED:'설명의 이동 확인 필요',")
    marker='if(s.source_phrases?.length)'
    assert marker in js
    js=js.replace(marker,"""for(let m of s.motions||[]){let row=el('div',undefined,'feature motion');let title=({DIRECTED_MOTION_OBSERVED:'이 방향의 이동이 보임',MOVEMENT_SEEN_STAGE_UNCONFIRMED:'이동은 보이나 설명 구간 연결은 불확실',MOTION_OBSERVATION_INSUFFICIENT:'관측 근거 부족',MOTION_CONTINUITY_UNCONFIRMED:'이동 과정 확인 필요',OPPOSITE_DIRECTION_CANDIDATE:'반대 방향으로 보이는 후보',EXPECTED_MOTION_UNCONFIRMED:'이동 확인 보류'})[m.display_status||m.status]||'추가 확인 필요';row.append(el('div',m.name),el('span',title));card.append(row,el('p',m.display_message||'이동 일부의 관측이며 전체 순서 판정은 아닙니다.','small'));}if(s.source_phrases?.length)""")
    js=js.replace("아직 확인하지 못하는 부분: 엄지·손바닥 방향·접촉 여부, 짧은 주요 동작 누락, 정확한 반복 횟수. 결과에 표시되지 않았다고 모두 맞았다는 뜻은 아닙니다.",
        "전체 순서·반복 횟수 판정은 검증 중입니다. ‘이동이 보임’은 방향 일부를 확인했다는 뜻이며 그 표현을 순서대로 끝까지 했다는 뜻은 아닙니다. 엄지·손바닥 방향·정확한 접촉도 아직 확인하지 못합니다.")
    js=js.replace('손모양 보존 + 신체 관계 후보','학습된 손모양·관계·이동 후보')
    js+='''
$('publicOrder').onclick=async()=>{try{let w=$('word').value;let r=await(await fetch('/api/order/'+w)).json();if(r.error)throw Error(r.error);clearStream();$('query').src='/media/order/'+w;$('queryLabel').textContent='주요 구간의 재생 순서를 뒤집은 공개 시험';render(r,'물리적 순서 변형 시험 · 모든 표현의 언어적 오답이라는 뜻은 아닙니다.');}catch(e){$('progress').textContent=e.message;}};
$('publicFresh').onclick=async()=>{if(busy||!ready)return;busy=true;$('publicFresh').disabled=true;let started=performance.now();try{let w=$('word').value;clearStream();$('query').src='/media/public/'+w;$('queryLabel').textContent='공개 영상 · 지금 새로 분석';result=null;let response=await fetch('/api/analyze-public?word='+w+'&condition=learned_motion',{method:'POST',headers:{'X-Session-Token':token}});let j=await response.json();if(!response.ok)throw Error(j.error||'분석 시작 실패');for(;;){await new Promise(r=>setTimeout(r,700));let job=await(await fetch('/api/job/'+j.id)).json();$('progress').textContent='공개 영상 새 분석 · '+((performance.now()-started)/1000).toFixed(1)+'초';if(job.status==='failed')throw Error(job.error);if(job.status==='complete'){render(job.result,'공개 영상 실제 새 분석 · 저장된 결과를 불러온 것이 아닙니다.');break;}}}catch(e){$('progress').textContent=e.message;}finally{busy=false;$('publicFresh').disabled=false;}};
'''
    new('demo.js',js)
    # No old inference rerun: use the exact saved numerical results and overlay.
    preserved=[]
    for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
        for kind,folder,oldfolder in [('reencoded_normal','demo_public','demo_public'),('action_cut','demo_controls','demo_controls')]:
            src=R/'feedback'/f'rgb_residual_42_{word}_{kind}.json';out=read(src)
            a=dict(np.load(U/'actual_controls'/f'{word}_{kind}/analysis.npz'))
            out=qualify(out,a);old=read(V/oldfolder/word/'RESULT.json');out['overlay']=old['overlay']
            out['latency']['total_analysis_seconds']=old['latency']['total_analysis_seconds']+out['motion_head_seconds']
            out['latency']['cached_example_not_new_end_to_end_measurement']=True
            exclusive(G/folder/word/'RESULT.json',out);preserved.append(dict(source=str(src),sha256=sha(src)))
        out=qualify(read(G/'order_feedback'/f'{word}.json'),dict(np.load(G/'order_inputs'/word/'analysis.npz')))
        exclusive(G/'demo_order'/word/'RESULT.json',out)
    for n in ['learner_adapter.py','run_demo.py']:compile((G/n).read_text(encoding='utf8'),str(G/n),'exec')
    exclusive(G/'DEMO_PREPARED.json',dict(port=8807,status='PREPARED_NOT_LAUNCHED',old_demo_preserved=True,old_outputs=preserved,
        files={str(G/n):sha(G/n) for n in ['learner_adapter.py','run_demo.py','demo.html','demo.js']},
        new_training=False,full_sequence_correctness_not_claimed=True))
    print('Isolated 8807 demo prepared with saved examples, gap/sequence limitations and live public check')

if __name__=='__main__':main()
