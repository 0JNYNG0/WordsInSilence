from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_semantic_episode_pilot_20261031'
def main():
 auth=json.loads((U/'AUTHORIZATION.json').read_text(encoding='utf8'));text=(OLD/'RESULT_KO.html').read_text(encoding='utf8')
 for r in auth['rows']:text=text.replace(Path(r['source']).as_uri(),'/media/'+r['alias'])
 assert 'file:///' not in text
 css='''<style id="readability-fix">:root{color-scheme:only light!important}html,body{background:#edf2f6!important;color:#17283b!important}h1,h2,h3,h4,p,li,div,span,table,td,th,summary,details,label{color:#17283b!important}a{color:#145e96!important}button,select,option,input{color:#17283b!important;background-color:#fff!important}svg text{fill:#17283b!important}video{color-scheme:dark!important}::selection{background:#b6dcff!important;color:#10243c!important}</style>'''
 text=text.replace('</style>','</style>'+css,1)
 text=text.replace('<main>','<main><p><a href="/">← 영상·학습 원리와 이번 단계 구분으로 돌아가기</a></p>',1)
 # User-visible loading/error rather than silent empty video boxes.
 text=text.replace("loadClip();window.reportReady=true;","for(const v of document.querySelectorAll('video')){let n=document.createElement('p');n.style.color='#17283b';n.textContent='영상을 불러오는 중입니다.';v.after(n);v.addEventListener('loadeddata',()=>n.textContent='영상 준비됨 · 재생 버튼으로 확인할 수 있습니다.');v.addEventListener('error',()=>n.textContent='영상 연결 실패: 현재 로컬 보고서 주소에서 다시 열어 주세요.');}loadClip();window.reportReady=true;")
 (U/'previous_fixed.html').write_text(text,encoding='utf8')
 (U/'PRESENTATION_REPAIR.json').write_text(json.dumps(dict(original=str(OLD/'RESULT_KO.html'),original_sha256=hashlib.sha256((OLD/'RESULT_KO.html').read_bytes()).hexdigest(),copy_sha256=hashlib.sha256((U/'previous_fixed.html').read_bytes()).hexdigest(),fixes=['replace file URLs with same-origin public allowlisted HTTP media routes','byte-range streaming for seek','explicit foreground/background for text/forms/SVG, only-light scheme','visible loading/error state'],scientific_outputs_changed=False,root_cause_status='File-media and inherited-theme failure modes addressed; original app surface could not be inspected because both CUA runtimes failed to initialize. Verify repaired local HTTP route directly.'),ensure_ascii=False,indent=2),encoding='utf8')
 print('PRESENTATION_COPY_READY')
if __name__=='__main__':main()
