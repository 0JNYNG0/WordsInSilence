"""
Tools/diag_hand_basis3.py  (rev2)
────────────────────────────────────────────────────────────────
Hand bone basis 진단 (3차 — 구조 오류 수정 버전)

변경 사항 (rev2):
  1. IK influence = 0 유지로 종료, 원복 명령만 출력
  2. 진단 전용 Empty/constraint (DIAG3_ 접두사), 기존 프로젝트 불침
  3. A/B 리포트 target: effective_pn 기반 실제 적용 basis 역산
  4. C/D 리포트 target: X = Y×Z (오른손 좌표계), determinant 출력
  5. TEST_BOTH_VALID_INDEX 설정, TimestampMs 포함 프레임 정보 출력

실행:
    exec(open(r"C:\\MyUnityProjects\\WordsInSilence\\Tools\\diag_hand_basis3.py").read())
────────────────────────────────────────────────────────────────
"""

import bpy, json, math, mathutils

# ================================================================
# 설정
# ================================================================
JSON_PATH             = r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\Recordings\감사합니다.json"
BODY_SCALE            = 1.0
DEPTH_SCALE           = 0.0
FORWARD_OFFSET        = 0.15
HAND_DEPTH_GAIN       = 1.0
MIRROR_X              = True
SWAP_HAND_LABELS      = True
TEST_BOTH_VALID_INDEX = 0   # both_valid 목록 내 선택 인덱스
ARM_MATCH_TOL_DEG     = 1.0

P_L_SHOULDER=11; P_R_SHOULDER=12; P_L_WRIST=15; P_R_WRIST=16
H_WRIST=0; H_INDEX_MCP=5; H_MID_MCP=9; H_PINK_MCP=17

BONE_L_ARM='mixamorig:LeftArm'; BONE_L_FOREARM='mixamorig:LeftForeArm'
BONE_L_HAND='mixamorig:LeftHand'
BONE_R_ARM='mixamorig:RightArm'; BONE_R_FOREARM='mixamorig:RightForeArm'
BONE_R_HAND='mixamorig:RightHand'
ARM_BONES=[BONE_L_ARM, BONE_L_FOREARM, BONE_R_ARM, BONE_R_FOREARM]

DIAG_EMPTY_L  = "DIAG3_IK_L_Wrist"
DIAG_EMPTY_R  = "DIAG3_IK_R_Wrist"
DIAG_CONSTR_L = "DIAG3_IK_LEFT"
DIAG_CONSTR_R = "DIAG3_IK_RIGHT"

REST_BONES_L={'hand':BONE_L_HAND,'index1':'mixamorig:LeftHandIndex1',
              'middle1':'mixamorig:LeftHandMiddle1','pinky1':'mixamorig:LeftHandPinky1'}
REST_BONES_R={'hand':BONE_R_HAND,'index1':'mixamorig:RightHandIndex1',
              'middle1':'mixamorig:RightHandMiddle1','pinky1':'mixamorig:RightHandPinky1'}


# ================================================================
# 유틸
# ================================================================
def get_pt(pts, idx):
    if not pts or idx>=len(pts): return None
    p=pts[idx]; return (p.get('x',0),p.get('y',0),p.get('z',0)) if isinstance(p,dict) else None

def to_blender(x,y,z,cx,cy,sw,origin,char_sw):
    sc=char_sw*BODY_SCALE; xs=1 if MIRROR_X else -1
    return origin+mathutils.Vector((xs*(x-cx)/sw*sc, z*sc*DEPTH_SCALE-FORWARD_OFFSET, -(y-cy)/sw*sc))

def convert_hand_pt(x,y,z,cx,cy,sw,origin,char_sw):
    sc=char_sw*BODY_SCALE; xs=1 if MIRROR_X else -1
    return origin+mathutils.Vector((xs*(x-cx)/sw*sc, z/sw*sc*HAND_DEPTH_GAIN-FORWARD_OFFSET, -(y-cy)/sw*sc))

def angle_deg(a,b):
    try: return math.degrees(a.angle(b))
    except: return 999.0

def det3(c0,c1,c2):
    return c0.dot(c1.cross(c2))

def mat3_angle_deg(m1,m2):
    return max(angle_deg(m1.col[i].normalized(), m2.col[i].normalized()) for i in range(3))


# ================================================================
# IK / constraint 관리 — 프로젝트 데이터 보존
# ================================================================
def make_diag_empty(name, col):
    """진단 전용 Empty. 기존 프로젝트 Empty 불침."""
    if name in bpy.data.objects:
        o=bpy.data.objects[name]; o.animation_data_clear(); o.hide_viewport=True
        if name not in [x.name for x in col.objects]: col.objects.link(o)
        return o
    o=bpy.data.objects.new(name,None); o.empty_display_size=0.04; o.hide_viewport=True
    col.objects.link(o); return o

def save_constr_state(arm_obj, bone_names):
    """지정 뼈의 기존 constraint 상태 저장 (진단용 제외)."""
    saved={}
    for bn in bone_names:
        pb=arm_obj.pose.bones.get(bn)
        if pb:
            saved[bn]=[(c.name,c.type,getattr(c,'target',None),c.influence)
                       for c in pb.constraints if not c.name.startswith('DIAG3_')]
    return saved

def verify_constr_state(arm_obj, saved):
    """기존 constraint가 변경되지 않았는지 검증."""
    ok=True
    for bn,orig_list in saved.items():
        pb=arm_obj.pose.bones.get(bn)
        if not pb: continue
        cur=[(c.name,c.type,getattr(c,'target',None),c.influence)
             for c in pb.constraints if not c.name.startswith('DIAG3_')]
        for item in orig_list:
            if item not in cur:
                print(f"  [경고] {bn}: 기존 constraint 변경 감지 — {item}"); ok=False
    if ok: print("  [OK] 기존 constraint 무결")
    return ok

def add_diag_ik(arm_obj, bone_name, target, chain_count, cname):
    """진단용 IK constraint 추가 (기존 constraint 보존)."""
    b=arm_obj.pose.bones.get(bone_name)
    if not b: return None
    for c in list(b.constraints):          # 이전 진단 잔여물만 제거
        if c.name==cname: b.constraints.remove(c)
    c=b.constraints.new('IK'); c.name=cname
    c.target=target; c.chain_count=chain_count; c.use_rotation=False; c.influence=1.0
    return c

def set_diag_ik_influence(arm_obj, inf):
    """진단용 IK constraint만 influence 변경."""
    for bn,cn in [(BONE_L_FOREARM,DIAG_CONSTR_L),(BONE_R_FOREARM,DIAG_CONSTR_R)]:
        pb=arm_obj.pose.bones.get(bn)
        if pb:
            for c in pb.constraints:
                if c.name==cn: c.influence=inf

def suspend_existing_ik(arm_obj):
    """기존(비진단) IK constraint를 임시로 0으로. 반환: {(bn,name):orig_influence}."""
    snap={}
    for bn in [BONE_L_FOREARM,BONE_R_FOREARM]:
        pb=arm_obj.pose.bones.get(bn)
        if pb:
            for c in pb.constraints:
                if c.type=='IK' and not c.name.startswith('DIAG3_'):
                    snap[(bn,c.name)]=c.influence; c.influence=0.0
    return snap

def restore_existing_ik(arm_obj, snap):
    """suspend_existing_ik 으로 저장된 influence 복원."""
    for (bn,cn),inf in snap.items():
        pb=arm_obj.pose.bones.get(bn)
        if pb:
            for c in pb.constraints:
                if c.name==cn: c.influence=inf


# ================================================================
# FK bake
# ================================================================
def bake_ik_to_fk(arm_obj, saved_mats):
    """IK 평가 결과를 FK rotation_quaternion 으로 변환."""
    result={}
    for bn in ARM_BONES:
        pb=arm_obj.pose.bones.get(bn); bd=arm_obj.data.bones.get(bn)
        if not pb or not bd: continue
        rest=bd.matrix_local.to_3x3(); pose_a=saved_mats[bn].to_3x3()
        if pb.parent:
            pn=pb.parent.name
            p_pose_a=saved_mats.get(pn, pb.parent.matrix).to_3x3()
            p_bd=arm_obj.data.bones.get(pn)
            p_rest=p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
            lr=rest.inverted()@p_rest@p_pose_a.inverted()@pose_a
        else:
            lr=rest.inverted()@pose_a
        result[bn]=lr.to_quaternion()
    return result

def apply_arm_fk(arm_obj, fk_dict):
    for bn,q in fk_dict.items():
        pb=arm_obj.pose.bones.get(bn)
        if pb: pb.rotation_mode='QUATERNION'; pb.rotation_quaternion=q.copy()

def reset_hand_bones(arm_obj):
    for bn in [BONE_L_HAND,BONE_R_HAND]:
        pb=arm_obj.pose.bones.get(bn)
        if pb: pb.rotation_mode='QUATERNION'; pb.rotation_quaternion=mathutils.Quaternion()

def reset_all_bones(arm_obj):
    for pb in arm_obj.pose.bones:
        pb.rotation_mode='QUATERNION'; pb.rotation_quaternion=mathutils.Quaternion()
        pb.location=mathutils.Vector((0,0,0)); pb.scale=mathutils.Vector((1,1,1))


# ================================================================
# 팔 자세 검증
# ================================================================
def verify_arm(arm_obj, base_mats, label, fail_on_error=True):
    ok=True
    for bn in ARM_BONES:
        pb=arm_obj.pose.bones.get(bn)
        if not pb: continue
        d=mat3_angle_deg(pb.matrix.to_3x3(), base_mats[bn].to_3x3())
        if d>ARM_MATCH_TOL_DEG:
            print(f"  [{'FAIL' if fail_on_error else 'WARN'}] {bn.replace('mixamorig:',''):16s} {d:.2f}° from base")
            ok=False
    if ok: print(f"  [OK] 팔 자세 일치 (tol={ARM_MATCH_TOL_DEG}°)")
    return ok


# ================================================================
# Hand rotation 접근법
# ================================================================
def rest_hand_norm(arm, bd):
    def hl(n):
        b=arm.data.bones.get(n); return mathutils.Vector(b.head_local) if b else None
    h0=hl(bd['hand']); hi=hl(bd['index1']); hm=hl(bd['middle1']); hp=hl(bd['pinky1'])
    if not all([h0,hi,hm,hp]): return None
    fwd=(hm-h0); fwd=fwd.normalized() if fwd.length>1e-6 else None
    if not fwd: return None
    n=fwd.cross((hi-hp).normalized()); return n.normalized() if n.length>1e-6 else None

def target_hand_basis(pts21, pn_sign=1.0):
    """Y=fwd, Z=norm*pn_sign, X=lat=norm×fwd (내부용, lateral은 좌수 좌표계 주의)."""
    fwd=(pts21[H_MID_MCP]-pts21[H_WRIST])
    if fwd.length<1e-6: return None,None,None
    fwd=fwd.normalized()
    lr=(pts21[H_INDEX_MCP]-pts21[H_PINK_MCP])
    if lr.length<1e-6: return None,None,None
    norm=fwd.cross(lr.normalized())*pn_sign
    if norm.length<1e-6: return None,None,None
    norm=norm.normalized()
    return fwd, norm.cross(fwd).normalized(), norm   # (fwd, old_lat[left-handed], norm)

def set_hand_rotation_diag(arm_obj, bone_name, wrist_w, mid9_w, palm_n_w, flip_palm):
    b=arm_obj.pose.bones.get(bone_name); bd=arm_obj.data.bones.get(bone_name)
    if not b or not bd: return False
    hd=(mid9_w-wrist_w)
    if hd.length<1e-6: return False
    hd=hd.normalized()
    a_inv=arm_obj.matrix_world.to_3x3().inverted()
    dir_a=(a_inv@hd).normalized()
    rest=bd.matrix_local.to_3x3()
    q=rest.col[1].normalized().rotation_difference(dir_a)
    ny=dir_a; nx=(q@rest.col[0].normalized()).normalized(); nz=(q@rest.col[2].normalized()).normalized()
    pn_a=(a_inv@palm_n_w)
    if pn_a.length>1e-4:
        pn_a=pn_a.normalized()
        if flip_palm: pn_a=-pn_a
        pp=pn_a-pn_a.dot(ny)*ny; zp=nz-nz.dot(ny)*ny
        if pp.length>1e-4 and zp.length>1e-4:
            pp=pp.normalized(); zp=zp.normalized()
            ra=math.atan2(zp.cross(pp).dot(ny), max(-1.,min(1.,zp.dot(pp))))
            rq=mathutils.Quaternion(ny,ra); nx=rq@nx; nz=rq@nz
    da=mathutils.Matrix([[nx.x,ny.x,nz.x],[nx.y,ny.y,nz.y],[nx.z,ny.z,nz.z]])
    if b.parent:
        p_bd=arm_obj.data.bones.get(b.parent.name)
        p_rest=p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        pm=rest.inverted()@p_rest@b.parent.matrix.to_3x3().inverted()@da
    else: pm=rest.inverted()@da
    b.rotation_mode='QUATERNION'; b.rotation_quaternion=pm.to_quaternion(); return True

def set_hand_by_basis(arm_obj, bone_name, fwd_w, norm_w):
    """Y=fwd, Z=norm, X=Y×Z (오른손). lat_w 불필요."""
    b=arm_obj.pose.bones.get(bone_name); bd=arm_obj.data.bones.get(bone_name)
    if not b or not bd: return False
    a_inv=arm_obj.matrix_world.to_3x3().inverted()
    ny=(a_inv@fwd_w).normalized(); nz=a_inv@norm_w
    nz=nz-nz.dot(ny)*ny
    if nz.length<1e-6: return False
    nz=nz.normalized(); nx=ny.cross(nz).normalized()
    da=mathutils.Matrix([[nx.x,ny.x,nz.x],[nx.y,ny.y,nz.y],[nx.z,ny.z,nz.z]])
    rest=bd.matrix_local.to_3x3()
    if b.parent:
        p_bd=arm_obj.data.bones.get(b.parent.name)
        p_rest=p_bd.matrix_local.to_3x3() if p_bd else mathutils.Matrix.Identity(3)
        pm=rest.inverted()@p_rest@b.parent.matrix.to_3x3().inverted()@da
    else: pm=rest.inverted()@da
    b.rotation_mode='QUATERNION'; b.rotation_quaternion=pm.to_quaternion(); return True

def compute_AB_target_world(arm_obj, wrist_w, mid9_w, palm_n_w, flip_palm):
    """
    set_hand_rotation_diag() 가 실제 적용하는 world-space target basis 역산.
    반환: (target_x, target_y, target_z) — 오른손 좌표계 (X=Y×Z)
    """
    fwd_w=(mid9_w-wrist_w)
    if fwd_w.length<1e-6: return None,None,None
    fwd_w=fwd_w.normalized()
    A=arm_obj.matrix_world.to_3x3(); a_inv=A.inverted()
    fwd_a=(a_inv@fwd_w).normalized()
    pn_a=(a_inv@palm_n_w).normalized()
    if flip_palm: pn_a=-pn_a
    pp=pn_a-pn_a.dot(fwd_a)*fwd_a
    if pp.length<1e-4: return fwd_w,None,None
    pp=pp.normalized()
    tz_w=(A@pp).normalized()                 # target Z (effective pn projected)
    tx_w=fwd_w.cross(tz_w).normalized()      # target X = Y × Z (right-handed)
    return tx_w, fwd_w, tz_w


# ================================================================
# 리포트 — target/actual 모두 오른손 좌표계, determinant 포함
# ================================================================
def report_hand(arm_obj, bone_name, tx, ty, tz, label):
    """
    tx/ty/tz: 오른손 좌표계 target (X=Y×Z 보장).
    모든 오차와 determinant 출력.
    """
    bpy.context.view_layer.update()
    b=arm_obj.pose.bones.get(bone_name)
    if not b: print(f"  [{label}] 뼈 없음"); return
    wm=(arm_obj.matrix_world@b.matrix).to_3x3()
    ax=wm.col[0].normalized(); ay=wm.col[1].normalized(); az=wm.col[2].normalized()
    tx=tx.normalized(); ty=ty.normalized(); tz=tz.normalized()

    t_det=det3(tx,ty,tz); a_det=det3(ax,ay,az)
    fwd_err=angle_deg(ay,ty); norm_err=angle_deg(az,tz); lat_err=angle_deg(ax,tx)

    print(f"\n  [{label}]")
    print(f"    target  X: {tx.x:+.3f} {tx.y:+.3f} {tx.z:+.3f}")
    print(f"    target  Y: {ty.x:+.3f} {ty.y:+.3f} {ty.z:+.3f}  (forward)")
    print(f"    target  Z: {tz.x:+.3f} {tz.y:+.3f} {tz.z:+.3f}  (normal)")
    print(f"    actual  X: {ax.x:+.3f} {ax.y:+.3f} {ax.z:+.3f}")
    print(f"    actual  Y: {ay.x:+.3f} {ay.y:+.3f} {ay.z:+.3f}")
    print(f"    actual  Z: {az.x:+.3f} {az.y:+.3f} {az.z:+.3f}")
    print(f"    target det : {t_det:+.4f}  ({'RIGHT-HANDED' if t_det>0 else 'LEFT-HANDED !'})")
    print(f"    actual det : {a_det:+.4f}  ({'RIGHT-HANDED' if a_det>0 else 'LEFT-HANDED !'})")
    print(f"    forward_err : {fwd_err:6.1f}°")
    print(f"    normal_err  : {norm_err:6.1f}°  ← 핵심")
    print(f"    lateral_err : {lat_err:6.1f}°")


# ================================================================
# 액션 저장 — 저장 즉시 action=None 복원
# ================================================================
def save_action(arm_obj, name, frame=1):
    old=bpy.data.actions.get(name)
    if old: bpy.data.actions.remove(old)
    act=bpy.data.actions.new(name)
    arm_obj.animation_data_create(); arm_obj.animation_data.action=act
    for pb in arm_obj.pose.bones:
        pb.rotation_mode='QUATERNION'; pb.keyframe_insert('rotation_quaternion',frame=frame)
    arm_obj.animation_data.action=None   # 즉시 해제
    print(f"  → 저장: '{name}'  (action=None 복원)")
    return act


# ================================================================
# 메인
# ================================================================
def main():
    arm_obj=next((o for o in bpy.data.objects if o.type=='ARMATURE'),None)
    if not arm_obj: print("[오류] 아마추어 없음"); return
    print(f"[아마추어] {arm_obj.name}")

    bpy.context.view_layer.objects.active=arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    with open(JSON_PATH,encoding='utf-8') as f: rec=json.load(f)
    frames_data=rec.get('Frames',[])
    print(f"[JSON] {len(frames_data)}프레임")

    lb=arm_obj.pose.bones.get(BONE_L_ARM); rb=arm_obj.pose.bones.get(BONE_R_ARM)
    if lb and rb:
        l_sh_w=arm_obj.matrix_world@lb.head; r_sh_w=arm_obj.matrix_world@rb.head
        char_origin=(l_sh_w+r_sh_w)/2; char_sw=(l_sh_w-r_sh_w).length
    else:
        char_origin=mathutils.Vector((0,0,0)); char_sw=1.5
    print(f"[캐릭터] 어깨 중심:{char_origin}  너비:{char_sw:.3f}")

    # ── 양손 유효 프레임 수집 ────────────────────────────────────
    both_valid=[]
    for fd_idx,fd in enumerate(frames_data):
        if not fd.get('HasPose',False): continue
        pp=fd.get('PosePoints') or []
        mll=fd.get('LeftHandPoints') or []; mrr=fd.get('RightHandPoints') or []
        hml=fd.get('HasLeftHand',False); hmr=fd.get('HasRightHand',False)
        ch_l,has_l=(mrr,hmr) if SWAP_HAND_LABELS else (mll,hml)
        ch_r,has_r=(mll,hml) if SWAP_HAND_LABELS else (mrr,hmr)
        if not (has_l and has_r): continue
        ls=get_pt(pp,P_L_SHOULDER); rs=get_pt(pp,P_R_SHOULDER)
        if not ls or not rs: continue
        cx=(ls[0]+rs[0])/2; cy=(ls[1]+rs[1])/2
        sw=math.sqrt((ls[0]-rs[0])**2+(ls[1]-rs[1])**2)
        if sw<1e-4: continue
        both_valid.append((fd_idx,cx,cy,sw,ch_l,ch_r,pp,fd))

    n_both=len(both_valid)
    print(f"[탐색] 양손 유효 프레임: {n_both}개")
    if not both_valid: print("[오류] 없음"); return

    sel=min(TEST_BOTH_VALID_INDEX, n_both-1)
    fd_idx,cx,cy,sw,ch_l_pts,ch_r_pts,pose_pts,fd_raw=both_valid[sel]
    ts=fd_raw.get('TimestampMs','N/A')
    print(f"\n[선택 프레임]")
    print(f"  TEST_BOTH_VALID_INDEX : {TEST_BOTH_VALID_INDEX}  (실제 사용: {sel})")
    print(f"  원본 JSON frame index : {fd_idx}")
    print(f"  TimestampMs           : {ts}")
    print(f"  전체 양손 유효 수     : {n_both}")
    print(f"  LEFT/RIGHT 동일 프레임: True (보장)")

    # ── 랜드마크 변환 ────────────────────────────────────────────
    def chp(idx,pts):
        pt=get_pt(pts,idx)
        return convert_hand_pt(*pt,cx,cy,sw,char_origin,char_sw) if pt else None
    def bp(idx):
        pt=get_pt(pose_pts,idx)
        return to_blender(*pt,cx,cy,sw,char_origin,char_sw) if pt else None

    l_pts21=[chp(i,ch_l_pts) for i in range(21)]
    r_pts21=[chp(i,ch_r_pts) for i in range(21)]
    if any(p is None for p in l_pts21+r_pts21): print("[오류] 랜드마크 변환 실패"); return

    pose_lw=bp(P_L_WRIST); pose_rw=bp(P_R_WRIST)
    lw3=l_pts21[H_WRIST]; rw3=r_pts21[H_WRIST]
    lh9=l_pts21[H_MID_MCP]; rh9=r_pts21[H_MID_MCP]

    l_pn_raw=(l_pts21[H_INDEX_MCP]-lw3).cross(l_pts21[H_PINK_MCP]-lw3)
    r_pn_raw=(r_pts21[H_INDEX_MCP]-rw3).cross(r_pts21[H_PINK_MCP]-rw3)
    if l_pn_raw.length<1e-6 or r_pn_raw.length<1e-6: print("[오류] palm_n 길이 0"); return
    l_pn=l_pn_raw.normalized(); r_pn=r_pn_raw.normalized()

    print(f"\n[손목] Pose vs Hand landmark 불일치:")
    if pose_lw: print(f"  LEFT  {(lw3-pose_lw).length:.4f} m  ← 진단은 Hand pt0 사용")
    if pose_rw: print(f"  RIGHT {(rw3-pose_rw).length:.4f} m  ← 진단은 Hand pt0 사용")

    # ── 기존 constraint 저장 ─────────────────────────────────────
    existing_constr_saved=save_constr_state(arm_obj,[BONE_L_FOREARM,BONE_R_FOREARM])
    print(f"\n[기존 constraint 저장]")
    for bn,lst in existing_constr_saved.items():
        print(f"  {bn.replace('mixamorig:','')}: {[x[0] for x in lst]}")

    # ── 진단용 IK 설정 ───────────────────────────────────────────
    col=bpy.data.collections.get("IK_Targets")
    if not col:
        col=bpy.data.collections.new("IK_Targets")
        bpy.context.scene.collection.children.link(col)
    emp_lw=make_diag_empty(DIAG_EMPTY_L,col); emp_rw=make_diag_empty(DIAG_EMPTY_R,col)
    emp_lw.hide_viewport=False; emp_rw.hide_viewport=False

    # Hand/Finger FK 강제 (constraint 제거)
    finger_bones=([BONE_L_HAND,BONE_R_HAND]+
                  [f"mixamorig:{s}{f}{n}"
                   for s in ['Left','Right']
                   for f in ['HandThumb','HandIndex','HandMiddle','HandRing','HandPinky']
                   for n in [1,2,3]])
    for bn in finger_bones:
        pb=arm_obj.pose.bones.get(bn)
        if pb:
            for c in list(pb.constraints): pb.constraints.remove(c)
            pb.rotation_mode='QUATERNION'

    add_diag_ik(arm_obj,BONE_L_FOREARM,emp_lw,2,DIAG_CONSTR_L)
    add_diag_ik(arm_obj,BONE_R_FOREARM,emp_rw,2,DIAG_CONSTR_R)
    arm_obj.animation_data_create(); arm_obj.animation_data.action=None

    # ── IK 평가 ─────────────────────────────────────────────────
    reset_all_bones(arm_obj)
    emp_lw.location=pose_lw if pose_lw else lw3
    emp_rw.location=pose_rw if pose_rw else rw3

    # 기존 IK constraint 임시 중단 (진단 IK만 평가)
    ik_snap=suspend_existing_ik(arm_obj)
    set_diag_ik_influence(arm_obj,1.0)
    bpy.context.view_layer.update()

    print(f"\n[IK 평가 — 진단용 constraint만]")
    for bn in ARM_BONES:
        pb=arm_obj.pose.bones.get(bn)
        if pb:
            tail_w=arm_obj.matrix_world@pb.tail
            print(f"  {bn.replace('mixamorig:',''):18s} tail: {tail_w.x:.3f} {tail_w.y:.3f} {tail_w.z:.3f}")

    # ── IK → FK bake ────────────────────────────────────────────
    saved_mats={}
    for bn in ARM_BONES:
        pb=arm_obj.pose.bones.get(bn); saved_mats[bn]=pb.matrix.copy() if pb else None
        if pb and pb.parent and pb.parent.name not in saved_mats:
            saved_mats[pb.parent.name]=pb.parent.matrix.copy()

    base_arm_fk=bake_ik_to_fk(arm_obj,saved_mats)

    # 진단 IK=0, 기존 IK 복원, 순수 FK 평가
    set_diag_ik_influence(arm_obj,0.0)
    restore_existing_ik(arm_obj,ik_snap)
    apply_arm_fk(arm_obj,base_arm_fk)
    bpy.context.view_layer.update()

    print(f"\n[IK→FK bake 검증]")
    base_fk_mats={}
    bake_all_ok=True
    for bn in ARM_BONES:
        pb=arm_obj.pose.bones.get(bn)
        if not pb: continue
        base_fk_mats[bn]=pb.matrix.copy()
        d=mat3_angle_deg(pb.matrix.to_3x3(), saved_mats[bn].to_3x3())
        s="OK" if d<ARM_MATCH_TOL_DEG else "FAIL"
        print(f"  [{s}] {bn.replace('mixamorig:',''):16s} diff={d:.3f}°")
        if d>=ARM_MATCH_TOL_DEG: bake_all_ok=False
    if not bake_all_ok: print("  [경고] bake 오차 — 결과 신뢰도 낮음")

    # ── Rest Basis ───────────────────────────────────────────────
    a_inv_g=arm_obj.matrix_world.to_3x3().inverted()
    rest_norm_l=rest_hand_norm(arm_obj,REST_BONES_L)
    rest_norm_r=rest_hand_norm(arm_obj,REST_BONES_R)

    tf_l,_,tn_l_p1=target_hand_basis(l_pts21,+1.0)
    tf_r,_,tn_r_p1=target_hand_basis(r_pts21,+1.0)

    def auto_sign(tn_w,rn_a,side):
        if rn_a is None or tn_w is None: return 1.0
        d=(a_inv_g@tn_w).normalized().dot(rn_a)
        s=1.0 if d>=0 else -1.0
        print(f"  [{side} AUTO] dot={d:+.3f} → pn_sign={s:+.0f}"); return s

    # ── 접근법 반복 ───────────────────────────────────────────────
    approaches=[
        ('A',False,False,False,False),
        ('B',True, True, False,False),
        ('C',None, None, True, False),
        ('D',None, None, True, True ),
    ]
    saved_actions={}; approach_arm_mats={}

    print(f"\n{'='*60}")
    print("NOTE: 손가락=Rest Pose | 손목=Hand pt0 | IK 진단용=OFF(baked FK)")
    print(f"{'='*60}")

    for approach_label,flip_l,flip_r,use_basis,do_auto in approaches:
        pn_sign_l=1.0; pn_sign_r=1.0
        print(f"\n{'='*60}")
        print(f"접근법 {approach_label}:", end=" ")
        if not use_basis: print(f"set_hand_rotation  flip_L={flip_l} flip_R={flip_r}")
        else: print(f"set_hand_by_basis  pn_sign={'auto' if do_auto else '+1'}")
        print("="*60)

        arm_obj.animation_data.action=None
        apply_arm_fk(arm_obj,base_arm_fk); reset_hand_bones(arm_obj)
        bpy.context.view_layer.update()

        print("  [팔 검증] Hand 설정 전:")
        arm_ok=verify_arm(arm_obj,base_fk_mats,approach_label)

        # Hand 회전 적용
        if use_basis:
            if do_auto:
                pn_sign_l=auto_sign(tn_l_p1,rest_norm_l,"LEFT")
                pn_sign_r=auto_sign(tn_r_p1,rest_norm_r,"RIGHT")
            tfl,_,tnl=target_hand_basis(l_pts21,pn_sign_l)
            tfr,_,tnr=target_hand_basis(r_pts21,pn_sign_r)
            if tfl: set_hand_by_basis(arm_obj,BONE_L_HAND,tfl,tnl)
            if tfr: set_hand_by_basis(arm_obj,BONE_R_HAND,tfr,tnr)
            # 오른손 좌표계 target (X=Y×Z)
            txl=(tfl.cross(tnl).normalized() if tfl and tnl else None)
            txr=(tfr.cross(tnr).normalized() if tfr and tnr else None)
            tyl,tzl=tfl,tnl; tyr,tzr=tfr,tnr
        else:
            set_hand_rotation_diag(arm_obj,BONE_L_HAND,lw3,lh9,l_pn,flip_l)
            set_hand_rotation_diag(arm_obj,BONE_R_HAND,rw3,rh9,r_pn,flip_r)
            # target basis 역산 (effective_pn 기반)
            txl,tyl,tzl=compute_AB_target_world(arm_obj,lw3,lh9,l_pn,flip_l)
            txr,tyr,tzr=compute_AB_target_world(arm_obj,rw3,rh9,r_pn,flip_r)

        bpy.context.view_layer.update()

        print("  [팔 검증] Hand 설정 후:")
        arm_ok2=verify_arm(arm_obj,base_fk_mats,approach_label,fail_on_error=False)
        if not arm_ok2: print(f"  [WARN] 팔 자세 변동 — 저장하지만 플래그")

        # 리포트
        if txl and tyl and tzl:
            report_hand(arm_obj,BONE_L_HAND,txl,tyl,tzl,f"{approach_label} LEFT")
        if txr and tyr and tzr:
            report_hand(arm_obj,BONE_R_HAND,txr,tyr,tzr,f"{approach_label} RIGHT")

        # 팔 행렬 기록
        approach_arm_mats[approach_label]={bn:arm_obj.pose.bones.get(bn).matrix.copy()
                                            for bn in ARM_BONES if arm_obj.pose.bones.get(bn)}
        # 저장
        act=save_action(arm_obj,f"HandTest3_{approach_label}_ArmLocked")
        saved_actions[approach_label]=act

    # ── A 기준 팔 차이 출력 ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("A 기준 팔 자세 차이 (저장 직전 측정, 오른손 좌표계 기준)")
    print("="*60)
    ref=approach_arm_mats.get('A',{})
    for lbl in ['B','C','D']:
        cur=approach_arm_mats.get(lbl,{}); print(f"  A vs {lbl}:")
        for bn in ARM_BONES:
            if ref.get(bn) and cur.get(bn):
                d=mat3_angle_deg(ref[bn].to_3x3(), cur[bn].to_3x3())
                print(f"    {bn.replace('mixamorig:',''):16s} {d:.3f}°")

    # ── 저장 후 액션 전환 재검증 ─────────────────────────────────
    print(f"\n{'='*60}")
    print("저장 액션 전환 후 팔 자세 재검증 (action=X → frame_set(1) → update)")
    print("="*60)
    for lbl,act in saved_actions.items():
        arm_obj.animation_data.action=act
        bpy.context.scene.frame_set(1); bpy.context.view_layer.update()
        print(f"\n  HandTest3_{lbl}_ArmLocked 활성화 후:")
        verify_arm(arm_obj,base_fk_mats,lbl,fail_on_error=False)

    # A 액션으로 복귀 (IK=0 유지)
    act_a=saved_actions.get('A')
    if act_a: arm_obj.animation_data.action=act_a
    bpy.context.scene.frame_set(1); bpy.context.view_layer.update()

    # ── 기존 constraint 무결 검증 ────────────────────────────────
    print(f"\n[기존 constraint 무결성 검증]")
    verify_constr_state(arm_obj,existing_constr_saved)

    # ── 정리 (IK=0 유지) ────────────────────────────────────────
    emp_lw.hide_viewport=True; emp_rw.hide_viewport=True

    print(f"\n{'='*60}")
    print("완료. HandTest3_A/B/C/D_ArmLocked 시각 비교:")
    print("  IK influence = 0 유지 중 (baked FK 팔 자세)")
    print()
    print("  A: set_hand_rotation  flip=False | Hand pt0 손목 사용")
    print("  B: set_hand_rotation  flip=True")
    print("  C: set_hand_by_basis  pn_sign=+1")
    print("  D: set_hand_by_basis  pn_sign=auto")
    print()
    print("─ 시각 비교 완료 후 진단 도구 제거 명령 (콘솔 실행) ─")
    arm_name=arm_obj.name
    print(f"""
arm = bpy.data.objects.get("{arm_name}")
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
for bn, cn in [("{BONE_L_FOREARM}","{DIAG_CONSTR_L}"), ("{BONE_R_FOREARM}","{DIAG_CONSTR_R}")]:
    pb = arm.pose.bones.get(bn)
    if pb:
        for c in list(pb.constraints):
            if c.name == cn: pb.constraints.remove(c)
for ename in ["{DIAG_EMPTY_L}", "{DIAG_EMPTY_R}"]:
    e = bpy.data.objects.get(ename)
    if e: bpy.data.objects.remove(e)
bpy.context.view_layer.update()
""")
    print("="*60)


main()
