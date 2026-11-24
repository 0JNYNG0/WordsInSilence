from pathlib import Path
U=Path(__file__).resolve().parent;p=U/'index.html';s=p.read_text(encoding='utf8')
old='<input id="seek" type="range" min="0" max="100" value="0"><p id="time"></p>'
new='<input id="seek" type="range" min="0" max="100" step="0.001" value="0"><button id="prevFrame">원본 1프레임 이전</button> <button id="nextFrame">원본 1프레임 다음</button> <label><input id="dtwMode" type="checkbox" checked> 저장된 DTW 대응으로 참고 위치 맞추기</label><p id="time"></p>'
assert old in s;s=s.replace(old,new)
old='현재 슬라이더는 두 영상의 전체 진행 비율을 맞춰 보여줍니다. 정확히 같은 단계라는 주장이나 DTW 정답이 아닙니다. 이전 DTW 대응은 위 ‘이전 비교 결과’에서 별도로 볼 수 있습니다.'
new='개발 영상 4개는 저장된 DTW 대응을 볼 수 있습니다. 한 시점에 참고 여러 프레임이 연결되면 범위를 표시하고 중간 프레임을 보여줍니다. 이는 화면 표시용이며 검산은 연결된 모든 프레임을 사용했습니다. 참고용 영상끼리는 전체 진행 비율만 맞춥니다. DTW가 같은 핵심 동작을 정확히 찾았다는 보장은 아닙니다.'
assert old in s;s=s.replace(old,new)
old="let ri=Math.round(+el('seek').value/100*(r.frames-1));el('qvideo').currentTime=q.seconds[i];el('rvideo').currentTime=r.seconds[ri];el('time').textContent=`원본 f${i} · ${q.seconds[i].toFixed(3)}초 / 참고 f${ri} · ${r.seconds[ri].toFixed(3)}초`;"
new="let ri=Math.round(+el('seek').value/100*(r.frames-1)),mapping=q.projection?.reference_projections.find(x=>x.reference===r.alias),range='전체 진행 비율';if(el('dtwMode').checked&&mapping){let g=mapping.reference_frames_per_query[i];ri=g[Math.floor(g.length/2)];range=`DTW 참고 f${g[0]}–${g[g.length-1]}`;}el('qvideo').currentTime=q.seconds[i];el('rvideo').currentTime=r.seconds[ri];el('time').textContent=`원본 f${i} · ${q.seconds[i].toFixed(3)}초 / 참고 f${ri} · ${r.seconds[ri].toFixed(3)}초 (${range})`;"
assert old in s;s=s.replace(old,new)
old="for(let note of q.source_notes||[])"
new="for(let src of q.source_candidates||[]){let p=document.createElement('p'),a=document.createElement('a');p.textContent='연결을 검토한 설명: '+src.text+' ';a.href=src.url;a.target='_blank';a.textContent='공식 출처';p.append(a);el('sourceDetail').append(p);}for(let note of q.source_notes||[])"
assert old in s;s=s.replace(old,new)
old="el('sourceDetail').append(p);}}el('clip').onchange"
new="""el('sourceDetail').append(p);let f=document.createElement('p');f.textContent='영상에서 본 특징: '+(s.visible_features||[]).join(' / ');el('sourceDetail').append(f);if(s.source_uncovered_anchors?.length){let w=document.createElement('p');w.textContent='이 구간의 앞선 손동작은 연결한 설명으로 충분히 설명되지 않습니다. 삭제하거나 정답으로 채우지 않습니다.';el('sourceDetail').append(w);}for(let link of s.source_links||[]){let box=document.createElement('p');box.textContent='설명의 「'+link.text+'」와 부분 연결을 검토한 시점: ';for(let frame of link.anchor_frames){let b=document.createElement('button');b.textContent=`f${frame} (${q.seconds[frame].toFixed(2)}초)`;b.onclick=()=>{el('seek').value=frame/(q.frames-1)*100;seek()};box.append(b);}el('sourceDetail').append(box);}}}el('dtwMode').onchange=seek;function frameStep(d){el('seek').value=Math.max(0,Math.min(q.frames-1,Math.round(+el('seek').value/100*(q.frames-1))+d))/(q.frames-1)*100;seek();}el('prevFrame').onclick=()=>frameStep(-1);el('nextFrame').onclick=()=>frameStep(1);el('clip').onchange"""
assert old in s;s=s.replace(old,new)
s=s.replace('<details><summary>설명 연결과 관측 근거 자세히 보기</summary>','<details open><summary>설명 연결과 관측 근거 자세히 보기</summary>')
s=s.replace('현재는 이 연결의 시간 근거를 만드는 중입니다.','이번에는 공개 16개 영상에 단계 초안을 붙이고 설명과 연결되는 시점의 근거를 정리했습니다.')
s=s.replace('이번 작업은 동작의 순서를 보존하며','이번 작업에서는 공개 16개 영상의 동작 순서를 보존하며')
p.write_text(s,encoding='utf8')
