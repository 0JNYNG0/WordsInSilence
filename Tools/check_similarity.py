"""
레퍼런스 JSON들 간의 유사도 행렬을 출력한다.
점수가 높을수록 두 동작이 알고리즘 상 비슷하게 보임.
"""
import json, math, os

FOLDER = os.path.join(os.path.dirname(__file__), "videos")
RESAMPLE = 20

def load(path):
    return json.load(open(path, encoding="utf-8"))

def vec2_dist(a, b):
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2)

def hand_shape_score(pts_a, pts_b):
    if not pts_a or not pts_b or len(pts_a) < 21 or len(pts_b) < 21:
        return None
    ax, ay = pts_a[0]["x"], pts_a[0]["y"]
    bx, by = pts_b[0]["x"], pts_b[0]["y"]
    a9x, a9y = pts_a[9]["x"], pts_a[9]["y"]
    b9x, b9y = pts_b[9]["x"], pts_b[9]["y"]
    sa = math.sqrt((ax-a9x)**2 + (ay-a9y)**2)
    sb = math.sqrt((bx-b9x)**2 + (by-b9y)**2)
    if sa < 1e-4 or sb < 1e-4: return None
    total_n, total_f = 0, 0
    for i in range(1, 21):
        an = ((pts_a[i]["x"]-ax)/sa, (pts_a[i]["y"]-ay)/sa)
        bn = ((pts_b[i]["x"]-bx)/sb, (pts_b[i]["y"]-by)/sb)
        bf = (bn[0], -bn[1])
        dn = math.sqrt((an[0]-bn[0])**2 + (an[1]-bn[1])**2)
        df = math.sqrt((an[0]-bf[0])**2 + (an[1]-bf[1])**2)
        total_n += dn; total_f += df
    d = min(total_n, total_f) / 20
    return max(0, 1 - d * 0.5)

def wrist_score(ur, rr, ul, rl):
    s, w = 0, 0
    def safe(v): return v and not math.isnan(v.get("x", float("nan")))
    if safe(ur) and safe(rr):
        dn = vec2_dist(ur, rr)
        df = vec2_dist(ur, {"x": rr["x"], "y": -rr["y"]})
        s += max(0, 1 - min(dn,df)/2); w += 1
    if safe(ul) and safe(rl):
        dn = vec2_dist(ul, rl)
        df = vec2_dist(ul, {"x": rl["x"], "y": -rl["y"]})
        s += max(0, 1 - min(dn,df)/2); w += 1
    return s/w if w else None

def score_frame(uf, rf):
    score, weight = 0, 0
    ur = uf.get("HasRightHand") and uf.get("RightHandPoints")
    rr = rf.get("HasRightHand") and rf.get("RightHandPoints")
    ul = uf.get("HasLeftHand")  and uf.get("LeftHandPoints")
    rl = rf.get("HasLeftHand")  and rf.get("LeftHandPoints")
    if ur and rr:
        s = hand_shape_score(uf["RightHandPoints"], rf["RightHandPoints"])
        if s is not None: score += s*0.5; weight += 0.5
    elif ul and rl:
        s = hand_shape_score(uf["LeftHandPoints"], rf["LeftHandPoints"])
        if s is not None: score += s*0.5; weight += 0.5
    if uf.get("HasValidShoulders") and rf.get("HasValidShoulders"):
        ws = wrist_score(
            uf.get("RightWristBodyRelative"), rf.get("RightWristBodyRelative"),
            uf.get("LeftWristBodyRelative"),  rf.get("LeftWristBodyRelative"))
        if ws is not None: score += ws*0.5; weight += 0.5
    return score/weight if weight else -1

def resample(frames, n):
    if not frames: return []
    if len(frames) == 1: return [frames[0]] * n
    result = []
    for i in range(n):
        t = i / (n-1)
        idx = round(t * (len(frames)-1))
        result.append(frames[idx])
    return result

def trim(frames):
    s = next((i for i,f in enumerate(frames) if f.get("HasLeftHand") or f.get("HasRightHand")), 0)
    e = next((i for i,f in enumerate(reversed(frames)) if f.get("HasLeftHand") or f.get("HasRightHand")), 0)
    e = len(frames) - 1 - e
    return frames[s:e+1] if s <= e else frames

def compare(a_frames, b_frames):
    af = resample(trim(a_frames), RESAMPLE)
    bf = resample(trim(b_frames), RESAMPLE)
    scores = [score_frame(u, r) for u, r in zip(af, bf)]
    valid = [s for s in scores if s >= 0]
    return sum(valid)/len(valid) if valid else 0

# 파일 로드
files = sorted(f for f in os.listdir(FOLDER) if f.endswith(".json"))
data  = {f: load(os.path.join(FOLDER, f)) for f in files}
names = [d["MotionId"] for d in data.values()]

# 헤더
print(f"{'':8}", end="")
for n in names: print(f"{n:8}", end="")
print()

# 행렬
for fa, da in data.items():
    print(f"{da['MotionId']:8}", end="")
    for fb, db in data.items():
        score = compare(da["Frames"], db["Frames"])
        print(f"{round(score*100):8}", end="")
    print()
