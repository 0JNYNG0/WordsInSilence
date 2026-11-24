from pathlib import Path
D=Path(__file__).resolve().parent
s=(D/'capture_plume.py').read_text(encoding='utf-8-sig')
s=s.replace("call('Page.navigate',dict(url=BASE))","call('Emulation.setDeviceMetricsOverride',dict(width=1600,height=1000,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url=BASE+'/shape-study.html'))")
a=s.index('  until("document.body?');b=s.index("  check('no JS errors'",a)
s=s[:a]+'''  until("document.body?.dataset.ready==='true'")
  time.sleep(.2);shot('MOA_SMOKE_SHAPE_OPTIONS.png')
'''+s[b:]
s=s.replace('qa/PLUME_CAPTURE.json','qa/SHAPE_CAPTURE.json').replace('PLUME SEQUENCE CAPTURED','SHAPE OPTIONS CAPTURED')
(D/'capture_shapes.py').write_text(s,encoding='utf-8')
