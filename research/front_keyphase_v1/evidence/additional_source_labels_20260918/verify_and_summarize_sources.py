"""Independent standard-library ZIP directory check and saved-byte validation."""
from pathlib import Path
from collections import Counter
import hashlib
import io
import json
import re
import shutil
import zipfile
import zlib

import av

OUT = Path(__file__).resolve().parent


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


index = json.loads((OUT / "ksl77_zip_index.json").read_text(encoding="utf-8"))
acq = json.loads((OUT / "ksl77_acquisition.json").read_text(encoding="utf-8"))
mapping = json.loads((OUT / "ksl77_label_mapping.json").read_text(encoding="utf-8"))
directory = (OUT / "ksl77_zip_directory.bin").read_bytes()
tail = (OUT / "ksl77_zip_tail.bin").read_bytes()
start = index["central_directory_offset"]
size = index["archive_bytes"]
suffix = directory + tail[start + len(directory) - (size - len(tail)):]
assert start + len(suffix) == size


class ArchiveSuffix(io.RawIOBase):
    """Read-only virtual archive exposing just independently downloaded metadata."""
    def __init__(self):
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else size) + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, length=-1):
        assert self.position >= start, "Attempted access outside downloaded directory"
        end = size if length < 0 else min(size, self.position + length)
        data = suffix[self.position - start:end - start]
        self.position = end
        return data


with zipfile.ZipFile(ArchiveSuffix()) as archive:
    independent = archive.infolist()
assert len(independent) == len(index["members"]) == 1296
for info, row in zip(independent, index["members"]):
    assert (info.filename, info.file_size, info.compress_size, info.CRC, info.header_offset, info.compress_type, info.flag_bits) == (
        row["name"], row["uncompressed_size"], row["compressed_size"], int(row["crc32"], 16), row["local_header_offset"], row["compression_method"], row["flags"])
video_members = [m for m in index["members"] if not m["name"].endswith("/")]
groups = Counter(m["name"].split("/")[0] for m in video_members)
prefixes = sorted({Path(m["name"]).name.split("_")[0] for m in video_members})
rows = []
for row in acq["files"]:
    path = Path(row["path"])
    data = path.read_bytes()
    assert len(data) == row["bytes"]
    assert hashlib.sha256(data).hexdigest() == row["sha256"]
    assert f"{zlib.crc32(data) & 0xffffffff:08x}" == row["crc32"]
    with av.open(str(path)) as container:
        metadata_keys = sorted(set(container.metadata).union(*(set(s.metadata) for s in container.streams)))
        identity_keys = [key for key in metadata_keys if re.search(r"signer|subject|participant|actor|person", key, re.I)]
        metadata = {"container_keys": sorted(container.metadata), "all_metadata_keys": metadata_keys,
                    "candidate_identity_tag_keys": identity_keys,
                    "creation_time": container.metadata.get("creation_time"),
                    "frames_decoded_by_this_verifier": 0}
    rows.append({"member": row["member"], "bytes_crc_sha_match": True, "metadata": metadata})
selected = {r["member"] for r in acq["files"]}
expected = {m["name"] for m in video_members if m["name"].split("/")[0] in ("16", "48")}
assert selected == expected and len(selected) == 36
assert len({r["sha256"] for r in acq["files"]}) == 36
classes = []
for folder, target in (("16", "MEOKDA"), ("48", "GONGBU")):
    class_prefixes = sorted(Path(r["member"]).name[:2] for r in acq["files"] if r["member"].startswith(folder + "/"))
    classes.append({"source_class_id": int(folder), "source_label": mapping["class_labels"][str(int(folder))],
                    "target_candidate": target, "files_available_and_acquired": len(class_prefixes),
                    "filename_prefixes": class_prefixes,
                    "absent_prefixes_compared_to_archive": sorted(set(prefixes) - set(class_prefixes)),
                    "people_count": "UNVERIFIED_FILENAME_PREFIXES_ARE_NOT_CONFIRMED_IDENTITIES"})
verification = {"status": "PASS_SOURCE_BYTES_AND_DIRECTORY_ONLY", "zip_directory_independent_reader": "Python stdlib zipfile",
                "directory_member_rows_exact": len(independent), "stored_files_verified": len(rows),
                "all_available_two_class_members_acquired": True, "unique_sha256_count": 36,
                "raw_bytes": sum(r["bytes"] for r in acq["files"]), "media_frames_decoded": 0,
                "files_with_candidate_identity_tag_keys": sum(bool(r["metadata"]["candidate_identity_tag_keys"]) for r in rows),
                "rows": rows}
save("independent_source_verification.json", verification)
coverage = {"status": "TWO_TARGET_CANDIDATES_NOT_TRAINING_APPROVAL", "nonempty_author_label_count": 77,
            "archive_video_count": len(video_members), "archive_class_directory_count": len(groups),
            "label_ids_without_directory": [int(k) for k, v in mapping["class_labels"].items() if v.strip() and f"{int(k):02}" not in groups],
            "archive_filename_prefixes": prefixes, "classes": classes,
            "selected_distinct_prefix_count": len({Path(x).name[:2] for x in selected}),
            "verified_signer_count": None, "approved_new_training_rows": 0,
            "remaining_targets_without_direct_label_candidate_in_this_archive": ["GAM", "MAEKJU", "MUL", "PPANG", "SARANG", "UNDONG"],
            "INSA": "hi label ID1 exists; not acquired or merged; equivalence not established",
            "prior_nine_target_training_coverage_gaps_resolved": 0,
            "source_grouping_policy": "Separate KSL77 namespace; all split UNASSIGNED. Prefix00 review-exposed, exclude it from future blind holdout. Other prefixes media-uninspected. Filename grouping is provisional until documentation verifies identity semantics.",
            "cross_source_signer_overlap": "UNKNOWN; separate filename namespaces do not prove disjoint people"}
save("ksl77_coverage_and_grouping.json", coverage)
triage = {"status": "BROAD_FORM_COMPATIBILITY_OBSERVED_NOT_LEXICAL_APPROVAL", "review_type": "assistant static-frame source triage",
          "reviewed_source_prefixes": ["00"], "reviewed_new_videos": 2, "uninspected_new_videos": 34,
          "observations": {"MEOKDA": "Both displayed sequences raise a right hand toward the mouth; exact finger configuration and movement repetition are not resolved by these samples.",
                           "GONGBU": "Both displayed sequences raise two upright hands beside the head; position, range and timing differ in the displayed frames."},
          "limitations": "Ten static frames per clip do not establish complete motion, correctness or lexical equivalence. No face-based identity matching was performed.",
          "source_class_mapping": "Author English label + matching class directory + broad form compatibility support candidates only.",
          "semantic_merges_approved": 0, "training_approved": False, "classifier_inference_calls": 0}
save("ksl77_visual_triage.json", triage)
asset_dir = Path("C:/Users/local-user/AppData/Local/Temp/browser-use/assets/99b64743-bdaa-4b65-99da-df7b9ab83e98")
assets = []
for filename, localname, url in (
    ("43fadee2782ca89b.png", "aihub636_public_json_example.png", "https://aihub.or.kr/web-nas/aihub21/files/editor/2022/06/73fec026939b4354a55f219976b011e2.png"),
    ("5cdaaf1803b74348.png", "aihub636_public_korean_text_example.png", "https://aihub.or.kr/web-nas/aihub21/files/editor/2022/06/72884e365d464255bac7cf2c4db80a22.png")):
    dest = OUT / localname
    shutil.copyfile(asset_dir / filename, dest)
    assets.append({"source_url": url, "local_file": localname, "bytes": dest.stat().st_size,
                   "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
aihub = {"status": "SCHEMA_AND_FILE_CATALOG_OBSERVED_TARGET_COVERAGE_UNKNOWN",
         "dataset_url": "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=636", "dataset_name": "재난 안전 정보 전달을 위한 수어영상 데이터",
         "public_example_assets": assets, "public_json_example_observed_gloss": "날씨1",
         "actual_signer_value_observed": False, "target_word_occurrences_verified": 0,
         "zero_occurrences_does_not_mean_absent": True,
         "training_label_archive_ui": [
             {"key": "61894", "description": "keypoint XML", "displayed_size": "51.95 GB"},
             {"key": "61895", "description": "morphology/nonmanual JSON", "displayed_size": "89.91 GB"},
             {"key": "61896", "name": "수어스크립트_TL.zip", "displayed_size": "372.37 MB"}],
         "sample_public_url": "https://aihub.or.kr/file/down.do?fileSn=15287&smpleFileSn=15287&dataSetSn=636",
         "anonymous_sample_request": "HTTP200 text/html login-required alert; not a ZIP; sample_fetch.json",
         "authenticated_browser_sample_download": "No local sample artifact verified; downloadMedia returned no path. Not retried indefinitely.",
         "browser_downloaded_video_zip": False, "training_approved": False,
         "reference_paper": "https://aclanthology.org/2022.sltat-1.9.pdf",
         "author_guide": "https://eq4all-data.github.io (under construction when observed)",
         "related_benchmark_repository": "https://github.com/SSL-Sign-Language/Korean-Disaster-Safety-Information-Sign-Language-Translation-Benchmark-Dataset",
         "next_access_route": "If revisiting636, use official logged-in access to the smaller training script ZIP; inspect actual gloss/signer fields before media selection. Korean text substring alone is not a sign label."}
save("aihub636_review.json", aihub)
print(json.dumps({k: v for k, v in verification.items() if k != "rows"}, ensure_ascii=False))
print(json.dumps(coverage, ensure_ascii=False))
