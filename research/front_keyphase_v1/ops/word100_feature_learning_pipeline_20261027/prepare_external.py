"""Safely inventory author-distributed manual handshape crops; never infer signer IDs."""
from pathlib import Path
from collections import Counter
import hashlib
import io
import json
import tarfile
from PIL import Image

UNIT = Path(__file__).resolve().parent
DATA = UNIT / "data"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    destination = DATA / "DEEPHAND_INVENTORY.json"
    if destination.exists():
        raise FileExistsError(destination)
    archive = DATA / "ph2014-dev-set-handshape-annotations.tar.gz"
    receipt = json.loads((DATA / "DEEPHAND_ACQUISITION.json").read_text(encoding="utf-8"))
    assert sha(archive.read_bytes()) == receipt["sha256"]
    root = "ph2014-dev-set-handshape-annotations/"
    (DATA / "deephand_images").mkdir(exist_ok=True)
    rows = []
    with tarfile.open(archive, "r:gz") as tar:
        labels = tar.extractfile(root + "3359-ph2014-MS-handshape-annotations.txt").read()
        for name in ["README", "LICENSE.txt"]:
            (DATA / ("DEEPHAND_" + name)).write_bytes(tar.extractfile(root + name).read())
        (DATA / "DEEPHAND_ORIGINAL_LABELS.txt").write_bytes(labels)
        # Decompress each image once in archive order. Random gzip seeks are quadratic.
        image_bytes = {}
        for m in tar:
            if m.isfile() and m.name.lower().endswith(".png"):
                assert not m.issym() and not m.islnk()
                image_bytes[m.name] = tar.extractfile(m).read()
        for line in labels.decode("utf-8").splitlines():
            if not line.strip():
                continue
            member, target = line.rsplit(" ", 1)
            m = tar.getmember(root + member)
            assert m.isfile() and not m.issym() and not m.islnk()
            raw = image_bytes[root + member]
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            pixels = sha(im.tobytes())
            key = sha(member.encode("utf-8"))
            filename = DATA / "deephand_images" / (key + ".png")
            if filename.exists():
                assert filename.read_bytes() == raw, "Partial extraction mismatch"
            else:
                filename.write_bytes(raw)
            clip = member.split("/")[2]
            rows.append({"id": key, "archive_member": root + member,
                         "image": str(filename), "image_sha256": sha(raw),
                         "pixel_sha256": pixels, "width": im.width, "height": im.height,
                         "label": target, "clip": clip,
                         "broadcast": clip.split("_default-")[0],
                         "original_frame_index": int(member.split("_fn")[-1].split("-")[0]),
                         "recorded_signer_id": None,
                         "label_origin": "author-distributed manual handshape class",
                         "ksl_semantic_label": False})
    assert len(rows) == 3359 and len({r["id"] for r in rows}) == 3359
    duplicate_pixels = {p: [r["id"] for r in rows if r["pixel_sha256"] == p]
                        for p, n in Counter(r["pixel_sha256"] for r in rows).items() if n > 1}
    result = {"status": "MANUAL_LABEL_INVENTORY_NOT_YET_SPLIT",
              "dataset": "RWTH PHOENIX2014 MS Handshapes / Deep Hand",
              "purpose": "external physical handshape development; not official benchmark or KSL correctness",
              "archive_sha256": receipt["sha256"], "rows": rows,
              "counts": {"images": len(rows), "classes": dict(Counter(r["label"] for r in rows)),
                         "clips": len({r["clip"] for r in rows}),
                         "broadcasts": len({r["broadcast"] for r in rows}),
                         "sizes": dict(Counter(f'{r["width"]}x{r["height"]}' for r in rows)),
                         "exact_pixel_duplicate_groups": len(duplicate_pixels)},
              "duplicate_pixels": duplicate_pixels,
              "license": "CC BY-NC-SA 3.0; preserve attribution, research isolation",
              "reference": "Koller, Ney, Bowden. Deep Hand. CVPR 2016.",
              "publication_url": "https://www-i6.informatik.rwth-aachen.de/publications/download/1000/Koller-CVPR-2016.pdf",
              "notes": ["Safe hashed filenames preserve literal-asterisk archive member mapping.",
                        "Directory /1/ is not treated as a signer ID.",
                        "No class was merged into project binary features without a reviewed taxonomy.",
                        "Sparse frame indices are not assumed to form complete motion sequences."]}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
