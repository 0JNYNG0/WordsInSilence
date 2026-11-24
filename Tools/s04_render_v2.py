"""
Tools/s04_render_v2.py — S01 인사 3시점 렌더링 (Blender Python)

blend_animate_multiview_v2.py 가 저장한 .blend 파일을 열어
front / left45 / right45 세 카메라 시점에서 PNG 시퀀스 렌더링.
렌더 완료 후 s04_encode_mp4.py (시스템 Python + cv2) 로 MP4 인코딩.

Blender CLI 실행:
  blender.exe <blend_file> --background --python Tools/s04_render_v2.py
"""

import bpy
import math
import os
import subprocess

# ─── 경로 ────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(_THIS_DIR, "s04_out_v2")
RENDER_DIR = os.path.join(OUT_DIR, "renders")

# 시스템 Python + 인코더 스크립트 (cv2 사용)
SYSTEM_PYTHON = r"C:\Users\local-user\AppData\Local\Programs\Python\Python313\python.exe"
ENCODE_SCRIPT = os.path.join(_THIS_DIR, "s04_encode_mp4.py")

OUTPUT_FPS = 30

# ─── 카메라 위치 정의 ─────────────────────────────────────────────────────────
# Blender 좌표계: Z=up. 캐릭터는 origin 근처.

CAMERAS = {
    "front": {
        "location": (0.0, -3.5, 1.2),
        "rotation_euler": (math.radians(80), 0.0, 0.0),
    },
    "left45": {
        "location": (-2.5, -2.5, 1.2),
        "rotation_euler": (math.radians(80), 0.0, math.radians(-45)),
    },
    "right45": {
        "location": (2.5, -2.5, 1.2),
        "rotation_euler": (math.radians(80), 0.0, math.radians(45)),
    },
}

RENDER_RESOLUTION = (640, 480)
RENDER_SAMPLES    = 32

# ─── 유틸 ────────────────────────────────────────────────────────────────────

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def get_or_create_camera(name, loc, rot_euler):
    if name in bpy.data.objects:
        cam_obj = bpy.data.objects[name]
    else:
        cam_data = bpy.data.cameras.new(name)
        cam_data.lens = 35
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location       = loc
    cam_obj.rotation_euler = rot_euler
    return cam_obj

# ─── 렌더 설정 ────────────────────────────────────────────────────────────────

def setup_render(scene, resolution, samples, engine="BLENDER_EEVEE_NEXT"):
    scene.render.engine                 = engine
    scene.render.resolution_x           = resolution[0]
    scene.render.resolution_y           = resolution[1]
    scene.render.resolution_percentage  = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps                    = OUTPUT_FPS
    if engine == "CYCLES":
        scene.cycles.samples = samples
    try:
        scene.eevee.taa_render_samples = samples
    except AttributeError:
        pass

# ─── PNG 시퀀스 → MP4 (시스템 Python subprocess) ──────────────────────────────

def encode_mp4(frame_dir, out_mp4, fps=OUTPUT_FPS):
    if not os.path.exists(SYSTEM_PYTHON):
        print(f"  [경고] 시스템 Python 없음: {SYSTEM_PYTHON}  — MP4 인코딩 건너뜀")
        return 1
    if not os.path.exists(ENCODE_SCRIPT):
        print(f"  [경고] 인코더 스크립트 없음: {ENCODE_SCRIPT}  — 건너뜀")
        return 1
    cmd = [SYSTEM_PYTHON, ENCODE_SCRIPT, frame_dir, out_mp4, str(fps)]
    print(f"  [encode] {os.path.basename(out_mp4)} ...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.stdout: print(result.stdout.rstrip())
    if result.returncode != 0 and result.stderr:
        print(f"  [경고] 인코더 오류:\n{result.stderr[-300:]}")
    return result.returncode

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end   = scene.frame_end
    print(f"[렌더] 프레임 범위: {frame_start}-{frame_end}  ({frame_end - frame_start + 1}프레임)")

    setup_render(scene, RENDER_RESOLUTION, RENDER_SAMPLES)

    frame_dirs = {}

    # ── 1단계: 모든 뷰 PNG 시퀀스 렌더링 ────────────────────────────────────
    for view_name, cam_cfg in CAMERAS.items():
        print(f"\n[시점] {view_name}")
        cam_obj = get_or_create_camera(
            f"RenderCam_{view_name}",
            cam_cfg["location"],
            cam_cfg["rotation_euler"],
        )
        scene.camera = cam_obj

        frame_dir = ensure_dir(os.path.join(RENDER_DIR, view_name))
        scene.render.filepath = os.path.join(frame_dir, "")
        bpy.ops.render.render(animation=True)
        print(f"  PNG 완료: {frame_dir}")
        frame_dirs[view_name] = frame_dir

    # ── 2단계: 시스템 Python 으로 MP4 인코딩 ────────────────────────────────
    print("\n[인코딩] PNG -> MP4")
    for view_name, frame_dir in frame_dirs.items():
        out_mp4 = os.path.join(OUT_DIR, f"preview_{view_name}.mp4")
        encode_mp4(frame_dir, out_mp4)

    print("\n=== 렌더 완료 ===")
    for view_name in CAMERAS:
        mp4 = os.path.join(OUT_DIR, f"preview_{view_name}.mp4")
        if os.path.exists(mp4):
            print(f"  {os.path.relpath(mp4)} ({os.path.getsize(mp4)//1024}KB)")
        else:
            print(f"  {view_name}: MP4 없음 (PNG 시퀀스만)")


main()
