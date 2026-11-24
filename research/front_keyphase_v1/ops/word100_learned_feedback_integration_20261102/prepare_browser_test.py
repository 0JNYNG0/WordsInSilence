from common import G,V,exclusive,sha

def main():
    code=(V/'check_render_only.py').read_text(encoding='utf8').replace("BASE='http://127.0.0.1:8806'","BASE='http://127.0.0.1:8807'")
    code=code.replace('DEMO_RENDER_VERIFICATION','DEMO_UI_VERIFICATION').replace('DEMO_RENDER_FIRST_FAILURE','DEMO_UI_FIRST_FAILURE')
    marker="  assert js('result.stages[0].status')=='IMPORTANT_RELATION_UNCONFIRMED'"
    assert marker in code
    start=code.index(marker);end=code.index('  js("scrollTo(0,0)")',start)
    replacement='''  for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
   js("$('word').value="+json.dumps(word)+";setWord();$('publicOrder').click()")
   for _ in range(100):
    if js('result!==null&&$("query").readyState>=2'):break
    time.sleep(.1)
   assert js('result.word_id')==word
   assert js('$("query").currentSrc.includes("/media/order/")')
   assert js('document.querySelectorAll(".motion").length===result.motion_observations.length')
   assert js('$("result").textContent.includes("전체 순서")')
   checks.append(word+' actual main-order edit, phase/motion scope shown')
  # Fresh analysis is deliberately public-only and stored outside private sessions.
  js("$('word').value='WORD1133';setWord();$('publicFresh').click()")
  for _ in range(120):
   if js('!busy&&result!==null'):break
   time.sleep(.5)
  assert js('!busy&&result!==null'),js('$("progress").textContent')
  assert js('result.training===false&&result.tuning===false&&result.individual_reference_comparisons===0')
  assert js('result.model_version==="learned-guide-rgb-motion-20261102-fixed42"')
  assert js('result.whole_sign_correctness===null')
  assert js('stream===null')
  assert js('document.querySelectorAll(".motion").length===result.motion_observations.length')
  assert js('$("result").textContent.includes("이 방향의 이동이 보임")')
  checks.append('fresh public video analyzed through new integrated pipeline; camera not opened; no private upload/read')
'''
    code=code[:start]+replacement+code[end:]
    code=code.replace('saved_example_seconds=','fresh_public_analysis_seconds=')
    code=code.replace("print(json.dumps(result,ensure_ascii=False),flush=True)","print(json.dumps(result,ensure_ascii=False),flush=True)")
    compile(code,str(G/'check_demo_browser.py'),'exec')
    (G/'check_demo_browser.py').write_text(code,encoding='utf8')
    exclusive(G/'BROWSER_TEST_INTENT.json',dict(tool_failure='In-app browser kernel assets unavailable (os error3). Owned local app headless regression fallback.',
        source='Fixed public ID12 video via /api/analyze-public only',camera=False,user_media=False,private_sessions=False,
        new_inference_count=1,test_code_sha256=sha(G/'check_demo_browser.py')))

if __name__=='__main__':main()
