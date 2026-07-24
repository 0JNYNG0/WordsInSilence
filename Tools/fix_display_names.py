"""
TrainingSamples 폴더의 JSON 파일 안 DisplayName 을
파일명 기준 레이블로 일괄 수정한다.

예: 감사합니다_1784890488633.json  →  DisplayName = "감사합니다"
"""
import os
import json
import re

FOLDER = os.path.join(
    os.path.expanduser("~"),
    "AppData", "LocalLow", "DefaultCompany", "WordsInSilence", "TrainingSamples"
)

def main():
    if not os.path.isdir(FOLDER):
        print(f"[오류] 폴더 없음: {FOLDER}")
        return

    files = sorted(f for f in os.listdir(FOLDER) if f.endswith(".json"))
    print(f"[정보] {len(files)}개 파일 발견: {FOLDER}\n")

    fixed = 0
    for fname in files:
        # 파일명에서 레이블 추출
        # 감사합니다_1784890488633.json → 감사합니다
        label = re.sub(r"_\d+\.json$", "", fname)
        path = os.path.join(FOLDER, fname)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        old = data.get("DisplayName", "")
        if old == label:
            print(f"  [변경 없음] {fname}  ({label})")
            continue

        data["DisplayName"] = label
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  [수정] {fname}  '{old}' → '{label}'")
        fixed += 1

    print(f"\n완료: {fixed}개 수정, {len(files)-fixed}개 이미 정상")

if __name__ == "__main__":
    main()
