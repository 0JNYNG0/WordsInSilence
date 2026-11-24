import json
from pathlib import Path

U=Path(__file__).resolve().parent
def read(name): return json.loads((U/name).read_text(encoding='utf-8-sig'))
def main():
    index=read('visual_inputs/INDEX.json')
    for row in index['rows']:
        for key in ['full_sheet','detail_sheet']:
            row[key]=Path(row[key]).relative_to(U).as_posix()
        for f in row['frames']: f['file']=Path(f['file']).relative_to(U).as_posix()
    data=dict(index=index,links=read('SOURCE_FORM_LINKS.json')['links'],obs=read('TOKEN_OBSERVATIONS.json')['annotations'],result=read('RESULT.json'))
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>뜻·표현형과 실제 동작 연결 | WordsInSilence</title>
<style>
:root{--bg:#f2f5f8;--ink:#142638;--muted:#586879;--blue:#1e5c94;--green:#1a6a5a;--line:#d8e1e9;--amber:#95650d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 "Malgun Gothic",sans-serif}main{max-width:1280px;padding:44px 24px;margin:auto}h1{font-size:35px;line-height:1.3;margin:8px 0 20px}h2{font-size:26px;margin:0 0 12px}h3{font-size:19px;margin:0 0 12px}p{margin:10px 0}.eyebrow{font-weight:700;color:var(--blue);letter-spacing:.08em}.lead{max-width:1000px;font-size:19px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.metric,.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:22px}.metric b{display:block;font-size:29px;color:var(--blue)}.muted{color:var(--muted)}.warning{border-left:5px solid #cb982c;background:#fff8e8;padding:16px 20px;border-radius:6px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.flow div{background:#e7eef7;border-radius:10px;padding:15px}.flow b{display:block}.nav{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.nav a{background:var(--ink);color:white;padding:8px 20px;border-radius:30px;text-decoration:none}section{margin-top:32px;scroll-margin-top:15px}.wordhead{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.badge{font-size:14px;border-radius:20px;padding:5px 12px;background:#edf4f3;color:var(--green);font-weight:700}.bad{background:#fff0d6;color:var(--amber)}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.samples{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}.sample{background:#f7f9fc;border:1px solid var(--line);border-radius:12px;overflow:hidden;min-width:0}.sample header{padding:13px 16px;font-weight:700}.sample img{display:block;width:100%;height:auto;aspect-ratio:16/9;object-fit:contain;background:#142434}.sample .ctrl{padding:12px 16px}.sample input{width:100%}.sample output{font-variant-numeric:tabular-nums;font-size:14px}.observation{padding:12px;background:white;border-radius:8px;min-height:90px}.tag{display:block;font-size:14px;margin:4px 0}.faint{font-size:14px;color:var(--muted)}a{color:var(--blue);overflow-wrap:anywhere}.tablewrap{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:14px 0}th,td{padding:13px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{background:#f2f5fa}ul{padding-left:22px}li{margin:6px 0}details{margin:14px 0}summary{cursor:pointer;font-weight:700}.legend{font-size:14px;padding:10px 0}.verified{color:var(--green)}footer{margin-top:36px;color:var(--muted);font-size:14px}.artifacts{display:flex;gap:14px;flex-wrap:wrap}
@media(max-width:800px){main{padding:24px 14px}h1{font-size:28px}.metrics,.flow{grid-template-columns:1fr 1fr}.samples,.columns{grid-template-columns:1fr}.metric{padding:16px}.panel{padding:18px}.sample img{max-height:450px}th,td{padding:9px;font-size:14px}}
</style><main>
<div class="eyebrow">WORDS IN SILENCE · 공개 영상 근거 확인 · 2026.10.27</div>
<h1>이름에서 끝내지 않고,<br>실제 보이는 동작까지 연결했습니다.</h1>
<p class="lead">사전에 적힌 뜻과 설명이 <b>지금 영상의 표현형에 맞는지</b> 먼저 확인했습니다. 그다음 어느 시점에 어떤 손 모양·신체 부위 관계가 보이는지 기록했습니다.</p>
<div class="metrics"><div class="metric"><b>4개 표현</b>기존 후보 중 고정 선택</div><div class="metric"><b>12개 영상</b>표현마다 기록 ID 01·02·03</div><div class="metric"><b>192장</b>실제 RGB 표본 확인</div><div class="metric"><b>38개 묶음</b>141개 고유 표본에 관찰 초안</div></div>
<div class="warning"><b>현재 성과는 학습 정답을 잘못 붙이지 않도록 준비한 것입니다.</b> 인식률 향상 실험이나 수어의 옳고 그름 검증은 아닙니다. 192장은 전체 영상의 모든 프레임이 아니며, 표본 사이의 동작은 자동으로 정답 처리하지 않았습니다.</div>
<div class="flow"><div><b>① 뜻·표현형 확인</b>같은 이름의 다른 뜻을 구분</div><div><b>② 실제 장면 확인</b>어느 손·어느 시점·무슨 특징인지</div><div><b>③ 주석 검증·평가 분리</b>확실한 부분만 학습 정답으로</div><div><b>④ 특징 학습·효과 비교</b>조건 충족 후 별도 후보로</div></div>
<p class="muted">이번에 ①·②를 작은 공개 묶음에서 진행했습니다. 큰 형태의 연결 3개, 전체 표현형의 최종 확정 0개입니다. ③의 독립 주석 검증과 평가 묶음 확정이 남아 있어 추가 학습은 0회입니다.</p>
<nav class="nav"><a href="#WORD0949">멀다</a><a href="#WORD1139">믿다</a><a href="#WORD1290">감사</a><a href="#WORD2551">별</a><a href="#next">다음 작업</a></nav>
<div id="words"></div>
<section id="next" class="panel"><h2>다음에는 무엇을 배우게 하나?</h2>
<div class="tablewrap"><table><thead><tr><th>구분</th><th>준비·검증할 내용</th></tr></thead><tbody>
<tr><td>영상에서 보이는 특징</td><td>검지를 폈는지, 손을 펼쳤는지, 반대 손·얼굴 근처인지, 어떤 순서로 바뀌는지. 우선 공통 특징을 배우는 후보를 준비합니다.</td></tr>
<tr><td>학습 자료 확장</td><td>같은 특징이 나오는 서로 다른 표현과 대조 사례를 추가합니다. 4개 표현만 학습하면 특징 대신 단어를 외울 수 있습니다.</td></tr>
<tr><td>평가 분리</td><td>기록 사람과 표현을 나눈 평가를 고정합니다. 단어 ID·Gloss·목표 정답을 관측 보완 모델의 입력으로 주지 않습니다.</td></tr>
<tr><td>피드백 정오 기준</td><td>“검지가 보인다”와 “이 표현에서는 검지가 필수다”는 다른 주장입니다. 필수 특징·허용 범위·실제 오류 근거는 따로 검증합니다.</td></tr>
</tbody></table></div>
<p>불확실한 손가락·접촉은 <b>모름</b>으로 남깁니다. 이를 0도, 틀림 또는 잘 맞음으로 바꾸지 않습니다. C·T3-F와 운영 웹·DTW는 유지했고, 사용자·멘토 촬영에는 접근하지 않았습니다.</p>
</section>
<section class="panel"><h2>검증과 한계</h2><ul><li>원본 영상·실제 프레임·시간·해시를 연결하고, 원본과 운영 모델 보존을 검산합니다.</li><li>별도 에이전트는 사전표를 읽기 전에 12개 영상의 표본을 먼저 관찰했습니다. 같은 AI 계열의 교차 검토이며 한국수어 전문가 정답은 아닙니다.</li><li>좌우는 원본 정면 영상에서 팔과 손의 연결을 보고 표시한 관찰 초안입니다. 관절 검출기의 좌우가 정답이라는 전제는 쓰지 않았습니다.</li><li>표정은 다루지 않았고, 정상적인 교차·가림을 오류로 판정하지 않았습니다. 이번에는 기존 좌표·시간 정렬을 바꾸지 않았습니다.</li></ul>
<div class="artifacts"><a href="visual_inputs/VIEWER.html">192장 원본 보기</a><a href="TOKEN_OBSERVATIONS.json">시간·특징 주석</a><a href="SOURCE_FORM_LINKS.json">사전 연결 판단</a><a href="review/SOURCE_TOKEN_DECISIONS.json">별도 시각 검토</a><a href="ANNOTATION_READINESS.json">학습 준비 상태</a><a href="review/FINAL_REVIEW.json">최종 검산</a></div></section>
<footer>공개 AI Hub 원본의 로컬 연구 검토 자료입니다. 사전 설명의 연결과 언어적 정오 판단은 구분합니다.</footer></main>
<script>const DATA=__DATA__;
const names={index_extended_coarse:'검지가 펴진 큰 모양',face_side_eye_cheek_neighborhood:'눈·광대 주변의 얼굴 옆',away_from_face_in_front:'얼굴에서 떨어져 앞쪽',approach:'두 손 접근',open_upright_coarse:'펼쳐 세운 큰 손 모양',right_index_near_left_hand:'오른 검지와 왼손이 가까움',torso_front:'몸통 앞',open_fingers_together_coarse:'손가락을 모아 펼친 모양',open_coarse:'펼친 손',right_above_left:'오른손이 왼손 위',upper_hand_approach_and_separate:'위 손의 접근과 떨어짐',two_open_hands_coarse:'두 펼친 손',grouped_or_curled_coarse:'모으거나 굽힌 큰 모양',upper_face_forehead_front:'얼굴 윗부분·이마 앞',open_spread_coarse:'손가락을 벌려 펼침',lowering_toward_rest:'손을 아래로 내림'};
const side={R:'오른손',L:'왼손',both:'두 손',active_hands:'동작하던 손'};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function sourceLink(id){const [s,n]=id.split(':');return s==='NIKL_legacy'?`https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=${n}&top_category=CTE`:`https://ksldict.korean.go.kr/kslserv/sign/signWord.do?signWordId=${n}`;}
const host=document.getElementById('words');
for(const w of DATA.links){const section=document.createElement('section');section.className='panel';section.id=w.word_id;
 section.innerHTML=`<div class="wordhead"><h2>${esc(w.label)}</h2><span class="badge ${w.state==='FORM_LINK_UNRESOLVED'?'bad':''}">${esc(w.status_ko)}</span></div><p><b>${esc(w.sense)}</b></p><p>${esc(w.summary)}</p><div class="columns"><div><h3>이번에 연결하는 것</h3><p>${esc(w.use)}</p></div><div><h3>아직 확정하지 않는 것</h3><p>${w.unknown.map(esc).join(' · ')}</p></div></div><div class="samples"></div><details><summary>사전 근거와 잘못 연결하지 않은 항목</summary><p>${w.source_ids.map(id=>`<a target="_blank" href="${sourceLink(id)}">${esc(id)}</a>`).join(' · ')}</p><p>${w.rejected_ids.length?'이 표본에 강제 적용하지 않음: '+w.rejected_ids.map(esc).join(', '):'세부 표현형과 허용 범위는 최종 확정하지 않았습니다.'}</p>${w.word_id==='WORD0949'?'<p>한 손 형태의 다른 공식 설명을 추가 6건의 한정 조회로 찾았으나 확보하지 못했습니다. 다른 형태가 없다는 뜻도, 이 영상이 틀렸다는 뜻도 아닙니다. <a href="source_addendum/RESULT_KO.txt">조사 결과</a></p>':''}</details>`;
 host.append(section);const grid=section.querySelector('.samples');
 for(const alias of w.aliases){const row=DATA.index.rows.find(r=>r.alias===alias);const card=document.createElement('div');card.className='sample';card.dataset.alias=alias;
 card.innerHTML=`<header>기록 ID ${String(row.recorded_person_id).padStart(2,'0')} · ${alias}</header><a class="image" target="_blank"><img alt="${esc(w.label)} 공개 영상의 실제 RGB 표본"></a><div class="ctrl"><output></output><input aria-label="${alias} 표본 시점" type="range" min="1" max="16" step="1" value="7"><div class="observation"></div><p class="faint">슬라이더는 실제로 확인한 16장만 이동합니다. 사진을 누르면 원본 크기로 봅니다.</p><a href="${row.detail_sheet}" target="_blank">16장 한눈에 보기</a></div>`;grid.append(card);
 const render=()=>{const ord=Number(card.querySelector('input').value);const f=row.frames[ord-1];card.querySelector('img').src=f.file;card.querySelector('.image').href=f.file;card.querySelector('output').textContent=`${(f.pts_us/1e6).toFixed(3)}초 · 프레임 ${f.frame_id} · 표본 ${ord}/16`;
 const a=DATA.obs.filter(a=>a.alias===alias&&a.evidence.some(e=>e.ordinal===ord));const facts=[...new Set(a.flatMap(x=>x.observed_features.map(f=>`${side[f.hand]} · ${names[f.value]||f.value}`)))];card.querySelector('.observation').innerHTML=facts.length?facts.map(s=>`<span class="tag">${esc(s)}</span>`).join(''):'<span class="muted">확인한 표본이지만, 이번 공통 특징 주석은 붙이지 않은 장면입니다.</span>';};card.querySelector('input').addEventListener('input',render);render();
 }
}
</script></html>'''
    page=page.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    (U/'RESULT_KO.html').write_text(page,encoding='utf-8')
    text='''뜻·표현형과 실제 동작 연결 — 공개12영상 검토

이번에는 단어 이름을 사전에 연결한 뒤 끝내지 않고, 멀다·믿다·감사·별 각3기록사람의 실제 RGB를 확인했다. 12영상에서16장씩192표본, 그중141개 고유표본에38개 관찰묶음을 저장했다. 시간범위는 표본의 처음과 끝일 뿐, 사이 모든 프레임의 정답이 아니다.

멀다: 사전8171의 두손 시작과 달리 세영상 모두 한손 검지가 얼굴옆→앞으로 향했다. 추가6공식조회에서도 일치하는 설명을 찾지 못했다. 명칭만으로 뜻/표현형 확정불가; 기존사전규칙 강제적용보류. 영상이 틀렸다는 판단아님.
믿다: 편검지와 반대펼친손의 접근이라는 큰특징은6700과잠정연결. 정확접촉·숨은손가락 미확정.
감사: 검사·감찰의동음이의12468제외. 고마움2372의 두펼친손·위아래접근은큰구조상맞음. 아래손은손바닥위처럼보이는장면이있어 사전손등방향을정답으로복사하지않음. 면방향·정확횟수·접촉보류.
별: 얼굴윗부분에서모음/굽힘과펼침이관찰됨.6163부분연결. 정확가로이동·별1/2변이·높이허용미확정. V11은종료장면이표본범위밖이므로종료를새로만들지않음.

학습:0회.사용자승인부족이아님.현재는같은AI교차검토한눈에보이는특징초안이며독립정답이아니다.4표현만으로특징학습성과를주장하면단어암기와구분하기어렵다.다음은공통특징별다른표현/대조사례확장과주석검증,사람및표현분리평가동결이다.관측모델에는목표단어/Gloss정답을주지않는다.표현의필수조건/허용범위/실제오류정답은별도다.

C·T3-F·운영웹/DTW 변경없음. 사용자/멘토자료접근없음. 새로운TCN인식률·DTW정확도향상결과가아니다.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    print('Report written')
if __name__=='__main__':main()
