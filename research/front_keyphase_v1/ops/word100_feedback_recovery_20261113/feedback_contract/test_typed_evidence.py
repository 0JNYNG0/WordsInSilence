from typed_evidence import explain_window
from pathlib import Path
import json

def main():
    p=dict(clip_id='public_fixture',source_sha256='a'*64,attribute='window_contains_torso_relative_wrist_translation',native_interval_us=[100000,366667],hand='R',state='M',observed=True,validated=True)
    c=dict(clip_id=p['clip_id'],source_sha256=p['source_sha256'],native_interval_us=p['native_interval_us'],hand='R',clause_id='guide:1',state='not_this_clause',validated=True)
    r=explain_window(p,c);assert r['physical']=='wrist_translation_present' and r['clause']=='not_this_clause'
    r=explain_window(dict(p,state='H'),dict(c,state='this_clause'),{'attribute':'wrist_translation','clause_id':'guide:1'})
    assert r['requirement_supported'] is False and r['physical']=='wrist_location_held'
    r=explain_window(dict(p,state='U'),c);assert r['physical']=='unknown'
    r=explain_window(dict(p,observed=False),dict(c,state='this_clause'));assert r['physical']=='unknown'
    r=explain_window(p,dict(c,hand='L'));assert r['clause']=='unknown'
    r=explain_window(p,dict(c,state='this_clause'),{'attribute':'wrist_translation','clause_id':'guide:1'})
    assert r['requirement_supported'] and r['completion']=='not_established' and not r['can_issue_correction']
    r=explain_window(dict(p,attribute='wrist_rotation'),c);assert r['physical']=='unknown'
    r=explain_window(p,dict(c,clip_id='other_public_clip'));assert r['clause']=='unknown'
    r=explain_window(p,dict(c,source_sha256='b'*64));assert r['clause']=='unknown'
    r=explain_window(dict(p,hand='X'),c);assert r['physical']=='unknown'
    r=explain_window(dict(p,source_sha256=None),c);assert r['physical']=='unknown'
    r=explain_window(p,dict(c,native_interval_us=[200000,466667]));assert r['clause']=='unknown'
    r=explain_window(p,dict(c,state='this_clause'),{'attribute':'wrist_translation','clause_id':'guide:2'});assert r['requirement_supported'] is False
    r=explain_window(dict(p,validated='false'),c);assert r['physical']=='unknown'
    r=explain_window(dict(p,observed='unknown'),c);assert r['physical']=='unknown'
    r=explain_window(p,dict(c,validated='false'));assert r['clause']=='unknown'
    r=explain_window(p,dict(c,state='this_clause'),{'attribute':'wrist_rotation','clause_id':'guide:1'});assert r['requirement_supported'] is False
    r=explain_window(p,dict(c,state='this_clause'),{'attribute':'wrist_translation'});assert r['requirement_supported'] is False
    try:explain_window(p,c,'bad');raise AssertionError('required type accepted')
    except ValueError:pass
    out=dict(status='PASS_TYPED_CONTRACT_SYNTHETIC_ONLY',checks=19,real_model_accuracy_test=False,production_changed=False,note='Source SHA/clip/native interval/hand/clause identity and strict boolean support required. A local motion does not prove order, extent or completion.')
    path=Path(__file__).with_name('TYPED_CONTRACT_TESTS.json')
    before=Path(__file__).with_name('TYPED_CONTRACT_TESTS_INITIAL_7.json')
    if path.exists() and not before.exists():before.write_bytes(path.read_bytes())
    interim=Path(__file__).with_name('TYPED_CONTRACT_TESTS_INITIAL_12.json')
    if path.exists() and not interim.exists():interim.write_bytes(path.read_bytes())
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8')
    print(out)
if __name__=='__main__':main()
