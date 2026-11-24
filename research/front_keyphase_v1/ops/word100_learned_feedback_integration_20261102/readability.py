"""Present supported positive observations first; retain detailed evidence."""
from common import G,read,sha,exclusive

def main():
    p=G/'demo.js';s=p.read_text(encoding='utf8');before=sha(p)
    marker="card.append(el('p',s.description),play,el('p',s.message));for(let f of s.features)"
    assert marker in s
    replacement="""let title=s.motions?.length?s.motions.map(m=>m.name).join(' / '):(s.source_phrases?.length?s.source_phrases.join(' '):s.description);card.append(el('p',title),play,el('p',s.message));let confirmed=s.features.filter(f=>f.expected>=.5&&f.status==='WITHIN_PUBLIC_TRAINING_RANGE').map(f=>f.name);let uncertain=s.features.filter(f=>!['WITHIN_PUBLIC_TRAINING_RANGE','DIFFERENCE_CANDIDATE'].includes(f.status)).map(f=>f.name);if(confirmed.length)card.append(el('p','확인한 손모양·배치: '+confirmed.join(', '),'small'));if(uncertain.length)card.append(el('p','추정하거나 보류한 항목: '+uncertain.join(', '),'small'));let featureDetails=el('details');featureDetails.append(el('summary','손모양 세부 근거와 원본 설명 보기'),el('p',s.description,'small'));for(let f of s.features)"""
    s=s.replace(marker,replacement)
    marker="card.append(row)}for(let t of s.relations||[])"
    assert marker in s;s=s.replace(marker,"featureDetails.append(row)}card.append(featureDetails);for(let t of s.relations||[])")
    p.write_text(s,encoding='utf8')
    exclusive(G/'PRESENTATION_CHANGE.json',dict(before_js_sha256=before,after_js_sha256=sha(p),model_output_changed=False,
        reason='Short supported-positive summary and uncertainty first; detailed negative attributes/long observation description collapsible. All numerical predictions unchanged.'))
    # Retest rendering only: the earlier real public analysis remains verified.
    t=(G/'check_demo_browser.py').read_text(encoding='utf8');start=t.index('  # Fresh analysis is deliberately public-only');end=t.index('  js("scrollTo(0,0)")',start)
    t=t[:start]+'''  js("$('word').value='WORD1133';setWord();$('public').click()")
  for _ in range(100):
   if js('result!==null&&$("query").readyState>=2'):break
   time.sleep(.1)
  assert js('$("result").textContent.includes("확인한 손모양·배치")')
  assert js('$("result").textContent.includes("손모양 세부 근거와 원본 설명 보기")')
  assert js('document.querySelectorAll(".motion").length===result.motion_observations.length')
  checks.append('clear positive summary, separate uncertain attributes, preserved detailed evidence; saved rendering only')
'''+t[end:]
    t=t.replace('DEMO_UI_VERIFICATION','DEMO_RENDER_VERIFICATION').replace('DEMO_UI_FIRST_FAILURE','DEMO_RENDER_FIRST_FAILURE').replace('fresh_public_analysis_seconds=','saved_example_not_fresh_seconds=')
    (G/'check_render_only.py').write_text(t,encoding='utf8')
    print('Simplified visible feedback; detailed evidence retained; render-only verification prepared')

if __name__=='__main__':main()
