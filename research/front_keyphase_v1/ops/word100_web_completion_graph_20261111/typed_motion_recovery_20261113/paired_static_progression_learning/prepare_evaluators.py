from static_core import *

def main():
    changes=[]
    for name in ['evaluate_full.py','evaluate_local.py']:
        src=P/name;dst=S/name;assert not dst.exists()
        before=src.read_text('utf-8')
        after=before.replace('from presence_core import *','from static_core import *\nP=S',1).replace("P/'presence_core.py'","S/'static_core.py'")
        assert after!=before
        dst.write_text(after,'utf-8')
        changes.append(dict(source=str(src),source_sha256=sha(src),output=str(dst),output_sha256=sha(dst),changes=['load static_core with same ProgressionHead and source label function','set experiment output/model root to S','pin static_core instead of P presence_core'],same_scoring_and_denominators=True))
    write(S/'EVALUATOR_REUSE.json',dict(at=stamp(),rows=changes,original_sources_unchanged=True))

if __name__=='__main__':main()
