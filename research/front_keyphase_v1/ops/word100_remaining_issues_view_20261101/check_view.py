import hashlib,html,json,re,subprocess,sys
from pathlib import Path
U=Path(__file__).resolve().parent
V=Path('C:/Users/local-user/.codex/visualizations/2026/09/18/01a0b4a2-0a79-7b80-b823-d6469869041f')
renderer=Path('C:/Users/local-user/.codex/plugins/cache/openai-bundled/visualize/1.0.66/skills/visualize/scripts/render.py')
fragment=V/'remaining-sign-evidence.html';qa=U/'qa_fragment.html'
qa.write_text(fragment.read_text('utf-8')+'\n'+(U/'qa_suffix.html').read_text('utf-8'),encoding='utf-8')
target=U/'qa_preview.html'
subprocess.run([sys.executable,'-B',str(renderer),str(qa),str(target),'--force'],check=True,capture_output=True)
text=target.read_text('utf-8');text=text.replace('</body>',(U/'qa_parent.html').read_text('utf-8')+'\n</body>')
target.write_text(text,encoding='utf-8')
results=[]
for width in (800,390):
    screen=U/f'qa_{width}.png'
    sized=U/f'qa_preview_{width}.html'
    sizing='<style>iframe{width:'+str(width-32)+'px!important;max-width:none!important;margin-left:0!important}#wis-qa-result{display:none}</style>'
    sized.write_text(text.replace('</head>',sizing+'</head>'),encoding='utf-8')
    args=['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless','--no-first-run','--disable-background-networking','--disable-component-update','--disable-sync','--disable-gpu','--host-resolver-rules=MAP * ~NOTFOUND',
          '--user-data-dir='+str(U/f'qa_profile_{width}'),f'--window-size={width},1100','--virtual-time-budget=5000','--dump-dom','--screenshot='+str(screen),sized.as_uri()]
    p=subprocess.run(args,capture_output=True,timeout=35)
    (U/f'qa_{width}_browser.log').write_bytes(p.stderr)
    (U/f'qa_{width}_dom.html').write_bytes(p.stdout)
    raw=p.stdout.decode('utf-8','replace');match=re.search(r'<pre id="wis-qa-result">(.*?)</pre>',raw,re.S)
    result=json.loads(html.unescape(match.group(1))) if match else {'status':'NO_QA_RESULT','exit':p.returncode}
    result['requested_width']=width;result['screenshot_exists']=screen.exists();results.append(result)
    print(json.dumps({'width':width,'status':result['status'],'checks':len(result.get('checks',[]))}))
(U/'BROWSER_QA.json').write_text(json.dumps({'fragment_sha256':hashlib.sha256(fragment.read_bytes()).hexdigest(),'results':results},ensure_ascii=False,indent=2),encoding='utf-8')
assert all(r['status']=='PASS' and r['screenshot_exists'] for r in results)
