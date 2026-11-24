"""Public metadata only. Never downloads a corpus, reads credentials or logs tokens."""
from pathlib import Path
import json, hashlib, shutil

D = Path(__file__).resolve().parent
U = D.parent

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    live = read(D/'OFFICIAL_CATALOG_CURRENT.json')
    old = read('W:/WordsInSilence/data/AIHub_103/manifests/AIHUB103_PUBLIC_CATALOG.json')
    cols = ['fileSn', 'streFileNm', 'fileSize', 'fileStreCours']
    rows = [{k:r[k] for k in cols} for r in live['response']['list']
            if 'real_sen_video' in str(r.get('streFileNm')) or 'real_sen_morpheme' in str(r.get('streFileNm'))]
    byid = {int(r['fileSn']): r for r in rows}
    oldbyid = {int(r['fileSn']):r for r in old['files']}
    assert all(int(r['fileSize']) == int(oldbyid[int(r['fileSn'])]['fileSize']) for r in rows)
    selected = read(U/'GREETING_SELECTION_CONFIRMED.json')['selected']
    p = Path(selected['representative'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == selected['sha256']
    result = dict(checked_at=live['checked_at'], source=live['url'], metadata_matches_cached=True,
        selected_expression=selected, files=rows,
        proven_original_container=dict(**byid[39625], member='01/NIA_SL_SEN0354_REAL01_F.mp4',
            member_bytes=p.stat().st_size, evidence='evidence/greeting_candidate_source_20260918/acquisition.json',
            download_needed_for_existing_representative=False),
        recommended_first_metadata=[byid[39584],byid[494853]],
        first_training_labels_bytes=int(byid[39584]['fileSize']),
        both_label_archives_bytes=sum(int(byid[i]['fileSize']) for i in [39584,494853]),
        all_real_sentence_video_archives=33,
        all_real_sentence_video_bytes=sum(int(r['fileSize']) for r in rows if 'video' in r['streFileNm']),
        capabilities=dict(archive_selection='Official filekey option supported; 103 is datasetkey.',
            select_mp4_inside_remote_zip='NOT VERIFIED; documented selection is filekey/ZIP, not archive member.',
            person_to_other_zip_mapping='NOT YET KNOWN. ZIP prefix cannot be assumed to equal performer ID.',
            full_corpus_required=False),
        plan=['Acquire only small SEN morpheme labels after scope confirmation; inspect SEN0354 names and performer IDs.',
              'Resolve exact source ZIP mapping for missing performers; skip existing REAL01 representative.',
              'If no approved member-level endpoint exists, select needed ZIPs one at a time; retain only required front clips locally.',
              'Verify actual RGB expression and annotation before any training; preserve signer-disjoint evaluation.'],
        space=dict(current={k:shutil.disk_usage(k).free for k in ['C:/','W:/']},
            official_shell_recommendation='2–3 times download size for automatic merge and extraction',
            known_01_archive_transient_bytes=[int(byid[39625]['fileSize'])*2,int(byid[39625]['fileSize'])*3],
            multi_performer_total='Cannot determine before mapping; 29.14GB is one known ZIP, not the total requirement.'),
        official_docs='https://aihub.or.kr/devsport/apishell/list.do?currMenu=403',
        downloaded_corpus_bytes=0, auth_material_accessed=False,
        implementation='Codex; independently rechecked byte totals against live official metadata, no external reviewer')
    (D/'GREETING_ACQUISITION_SCOPE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['metadata_matches_cached','first_training_labels_bytes','both_label_archives_bytes','downloaded_corpus_bytes']},ensure_ascii=False))

if __name__ == '__main__': main()
