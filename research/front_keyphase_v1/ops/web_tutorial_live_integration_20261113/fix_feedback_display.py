from pathlib import Path
p=Path(__file__).parent/'site/feedback.js'
s=p.read_text('utf-8')
s=s.replace("if(!result||!ref||!$('feedback-dialog').open)return;", "if(!result||!ref||!$('feedback-dialog').open||referenceOnly)return;")
s=s.replace("function selectCard(i){active=i;const c=result.cards[i];", "function selectCard(i){active=i;const c=result.cards[i];desired=null;setReferenceOnly(!c.query_seconds);")
s=s.replace("$('compare-play').onclick=()=>{active=-1;", "$('compare-play').onclick=()=>{active=-1;setReferenceOnly(false);")
s=s.replace("$('feedback-seek').oninput=()=>{", "$('feedback-seek').oninput=()=>{setReferenceOnly(false);")
s=s.replace("el.className=c.display_status;", "el.className=c.display_status==='difference'&&c.difference_spans?.length?'uncertain':c.display_status;")
s=s.replace("$('feedback-bands').append(el)}}", "$('feedback-bands').append(el);if(c.display_status==='difference')for(const [a,b]of c.difference_spans||[]){const d=document.createElement('span');d.className='difference';d.style.left=data.timestamp_us[a]/1e6/dur*100+'%';d.style.width=Math.max(.5,(data.timestamp_us[b]-data.timestamp_us[a])/1e6/dur*100)+'%';$('feedback-bands').append(d)}}}")
s+='''
function setReferenceOnly(value){
 referenceOnly=value;
 let note=document.getElementById('query-unmatched');
 if(!note){note=document.createElement('div');note.id='query-unmatched';note.textContent='이 행동에 해당하는 구간을 확인하지 못했어요. 전체 재생으로 직접 살펴보세요.';$('query-video').parentElement.append(note);}
 note.hidden=!value;$('query-video').hidden=value;$('query-skeleton').hidden=value;
}
'''
p.write_text(s,encoding='utf-8')
c=p.with_name('live.css')
c.write_text(c.read_text('utf-8')+'\n#query-unmatched{position:absolute;inset:0;display:grid;place-items:center;padding:30px;text-align:center;font-size:14px;line-height:1.8;color:#d1d9e5;background:#182131}#query-unmatched[hidden]{display:none}\n',encoding='utf-8')
