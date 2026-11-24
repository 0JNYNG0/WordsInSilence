"""Targeted filename-only search; inspect payload only for the chosen public clip."""
from pathlib import Path
import subprocess,json,hashlib

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
roots=[Path('C:/Users/local-user/Downloads'),Path('W:/WordsInSilence/data/AIHub_103'),ROOT/'research/front_keyphase_v1/ops']
patterns=['*keypoint*.zip','*keypoint*.zip.*','*keypoint*.tar*','*NIA_SL_WORD1567_REAL02*','*camparam*']
out=[]
for root in roots:
    command=['rg','--files',str(root)]
    for pattern in patterns:command+=['-g',pattern]
    result=subprocess.run(command,capture_output=True,encoding='utf-8',errors='replace')
    matches=[x for x in result.stdout.splitlines() if x]
    out.append(dict(root=str(root),command=command,exit_code=result.returncode,errors=result.stderr.strip(),matches=matches))
report=dict(status='TARGETED_EXISTING_SOURCE_SEARCH',roots=out,
            historical_archive='C:/Users/local-user/Downloads/수어 영상/1.Training/[라벨]02_real_word_keypoint.zip',
            historical_archive_exists=Path('C:/Users/local-user/Downloads/수어 영상/1.Training/[라벨]02_real_word_keypoint.zip').exists(),
            unrelated_tar='W:/ksl_json_sets.tar.gz',unrelated_tar_excluded_reason='Existing corpus_annotations_20261014 audit explicitly found no landmarks; not AIHub103 keypoint archive',
            filename_only=True,private_payload_access=False,external_transfer=False)
(U/'SOURCE_SEARCH.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({r['root']:dict(exit_code=r['exit_code'],matches=len(r['matches']),errors=r['errors'])for r in out},ensure_ascii=False))
