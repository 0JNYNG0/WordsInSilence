from pathlib import Path
U=Path(__file__).resolve().parent
p=U/'site/app.js';s=p.read_text(encoding='utf-8')
s=s.replace('const dt=Math.min((now-last)/1000,.05);', 'const dt=(now-last)/1000;')
s=s.replace("$('#zoom').textContent=zoom>1.8?'−':'＋'", "$('#zoom').textContent=zoom>1.8?'−':'＋'")
# Reset animation and zoom controls on every new visit.
s=s.replace("setScene('npc-arriving');", "$('#gesture').textContent='동작 시험 보기';$('#gesture-note').textContent='기존 캐릭터 · 등장 연출';$('#zoom').textContent='＋';setScene('npc-arriving');")
s=s.replace("$('#gesture').textContent=gesture?'동작 멈추기':'동작 시험 보기';", "$('#gesture').textContent=gesture?'동작 멈추기':'동작 시험 보기';")
# Page visibility resumes without replaying a slowed version; timeline remains wall-clock based.
p.write_text(s,encoding='utf-8')
p=U/'site/renderer.js';s=p.read_text(encoding='utf-8')
s=s.replace('c.rgb+=stripe*(1.-reveal)*.18;', 'c.rgb+=stripe*(1.-reveal)*.18;c.a*=smoothstep(0.,.2,reveal);')
p.write_text(s,encoding='utf-8')
