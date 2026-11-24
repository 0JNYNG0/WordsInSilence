from pathlib import Path
D=Path(__file__).resolve().parent
s=(D/'capture_flight_sequence.py').read_text(encoding='utf-8-sig')
s=s.replace("source=MOCK)","source=MOCK+CLOCK)")
clock=r'''
CLOCK=r"""
const actualNow=performance.now.bind(performance),actualRAF=window.requestAnimationFrame.bind(window);
window.__visualClock={manual:false,now:0,q:[]};
Object.defineProperty(performance,'now',{value:()=>__visualClock.manual?__visualClock.now:actualNow()});
window.requestAnimationFrame=cb=>{if(__visualClock.manual){__visualClock.q.push(cb);return -__visualClock.q.length;}return actualRAF(t=>{if(__visualClock.manual)__visualClock.q.push(cb);else cb(t);});};
window.__holdVisual=()=>{__visualClock.now=actualNow();__visualClock.manual=true;};
window.__advanceVisual=async(ms)=>{for(let n=0;n<ms;n+=20){__visualClock.now+=20;const q=__visualClock.q.splice(0);for(const cb of q)cb(__visualClock.now);await Promise.resolve();}};
"""
'''
s=s.replace('def main():',clock+'\ndef main():')
a=s.index('  js("document.querySelector(\'[data-scene=practice]');b=s.index("  check('no physical camera POST'",a)
s=s[:a]+'''  time.sleep(.6);js('__holdVisual()');time.sleep(.08)
  js("document.querySelector('[data-scene=practice]').click()")
  for i in range(25):
   if i:js('await __advanceVisual(200)')
   shot(f'fade_seq_{i:02d}.png')
  check('no JS errors',not errors)
'''+s[b:]
s=s.replace('qa/FLIGHT_CAPTURE.json','qa/FADE_CAPTURE.json').replace('FLIGHT SEQUENCE CAPTURED','FADE SEQUENCE CAPTURED')
(D/'capture_fade.py').write_text(s,encoding='utf-8')
