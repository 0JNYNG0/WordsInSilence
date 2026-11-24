from pathlib import Path
V=Path(__file__).resolve().parent
s=(V.parent/'word100_learned_guide_feedback_20261102/check_render_only.py').read_text(encoding='utf8').replace('http://127.0.0.1:8805','http://127.0.0.1:8806')
s=s.replace('DEMO_SCREEN.png','DEMO_CONTROL_SCREEN.png').replace('DEMO_RESULT.png','DEMO_CONTROL_RESULT.png')
needle="  assert js('document.querySelector(\".feature\").textContent.includes(\"배운 기준:\")')"
extra='''  for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
   js("$('word').value="+json.dumps(word)+";setWord();$('publicCut').click()")
   for _ in range(100):
    if js('result!==null&&$("query").readyState>=2'):break
    time.sleep(.1)
   assert js('result.word_id')==word
   assert js('result.overlay.points.length===result.frames')
   assert js('$("query").currentSrc.includes("/media/control/")')
   checks.append(word+' edited public comparison visible without any new inference')
  assert js('result.stages[0].status')=='IMPORTANT_RELATION_UNCONFIRMED'
  assert js('$("result").textContent.includes("양손의 긴 손가락 끝이 한곳으로 모임")')
  assert js('$("result").textContent.includes("이 관계를 확인하지 못함")')
  assert not js('$("result").textContent.includes("IMAGE_ESTIMATE_NEEDS_CONFIRMATION")')
'''
assert needle in s;s=s.replace(needle,extra+needle)
with (V/'check_render_only.py').open('x',encoding='utf8') as f:f.write(s)
