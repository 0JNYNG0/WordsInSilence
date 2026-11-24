using UnityEngine;
using UnityEngine.UI;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// HandPoseIntegrator의 UnifiedFrameArrived를 받아
    /// 상체 스켈레톤 (관절 점 + 뼈대 선)을 카메라 위에 오버레이 표시한다.
    ///
    /// 뼈대 연결:
    ///   코(0)
    ///   좌어깨(11) ── 우어깨(12)
    ///   좌팔꿈치(13)    우팔꿈치(14)
    ///   좌손목(15)      우손목(16)
    /// </summary>
    public sealed class PoseOverlayRenderer : MonoBehaviour
    {
        [SerializeField] HandPoseIntegrator _integrator;
        [SerializeField] RectTransform      _overlayRect;
        [SerializeField] Sprite             _dotSprite;
        [SerializeField] Color              _poseColor         = Color.yellow;
        [SerializeField] Color              _boneColor         = Color.green;
        [SerializeField] Color              _shoulderCenterColor = Color.red;
        [SerializeField] float              _boneWidth         = 3f;

        public bool ShowPoseLandmarks  = true;
        public bool ShowShoulderCenter = true;
        public bool ShowBones          = true;

        // 7점: 코(0), 좌어깨(11), 우어깨(12), 좌팔꿈치(13), 우팔꿈치(14), 좌손목(15), 우손목(16)
        static readonly int[] DotIndices = { 0, 11, 12, 13, 14, 15, 16 };
        const int DotCount = 7;
        const float DotSize = 12f;

        // 뼈대 연결 쌍 (DotIndices 배열 내 인덱스 기준)
        // 0=코, 1=좌어깨, 2=우어깨, 3=좌팔꿈치, 4=우팔꿈치, 5=좌손목, 6=우손목
        static readonly int[,] BonePairs = {
            { 1, 2 }, // 좌어깨 - 우어깨
            { 1, 3 }, // 좌어깨 - 좌팔꿈치
            { 3, 5 }, // 좌팔꿈치 - 좌손목
            { 2, 4 }, // 우어깨 - 우팔꿈치
            { 4, 6 }, // 우팔꿈치 - 우손목
        };
        static readonly int BoneCount = 5;

        Image[] _dots;
        Image   _shoulderCenterDot;
        RectTransform[] _bones;

        // 점 위치 캐시 (뼈대 그릴 때 재사용)
        Vector2[] _dotPositions = new Vector2[DotCount];

        // ─── MonoBehaviour ───────────────────────────────────────────────────

        void Awake()
        {
            CreateDots();
            CreateBones();
        }

        void OnEnable()
        {
            if (_integrator != null)
                _integrator.UnifiedFrameArrived += OnUnifiedFrame;
        }

        void OnDisable()
        {
            if (_integrator != null)
                _integrator.UnifiedFrameArrived -= OnUnifiedFrame;
            HideAll();
        }

        // ─── Event handler ───────────────────────────────────────────────────

        void OnUnifiedFrame(HandPoseFrame frame)
        {
            if (frame.PoseFrame == null) return;

            bool show = ShowPoseLandmarks;

            for (int i = 0; i < DotCount; i++)
            {
                if (_dots[i] == null) continue;
                _dots[i].gameObject.SetActive(show);
                if (!show) continue;

                int idx = DotIndices[i];
                var point = frame.PoseFrame.AllPoints[idx];
                var pos = NormalizedToRect(point.x, point.y);
                _dots[i].rectTransform.anchoredPosition = pos;
                _dotPositions[i] = pos;
            }

            // 뼈대 선
            bool showBones = ShowBones && show;
            for (int b = 0; b < BoneCount; b++)
            {
                if (_bones[b] == null) continue;
                _bones[b].gameObject.SetActive(showBones);
                if (!showBones) continue;

                int fromIdx = BonePairs[b, 0];
                int toIdx = BonePairs[b, 1];
                UpdateBone(_bones[b], _dotPositions[fromIdx], _dotPositions[toIdx]);
            }

            bool showCenter = ShowShoulderCenter && frame.HasValidShoulders;
            if (_shoulderCenterDot != null)
            {
                _shoulderCenterDot.gameObject.SetActive(showCenter);
                if (showCenter)
                {
                    _shoulderCenterDot.rectTransform.anchoredPosition =
                        new Vector2((1f - frame.ShoulderCenter.x) * _overlayRect.rect.width,
                                    frame.ShoulderCenter.y * _overlayRect.rect.height);
                }
            }
        }

        // ─── Internal ────────────────────────────────────────────────────────

        void CreateDots()
        {
            if (_overlayRect == null) return;

            _dots = new Image[DotCount];
            for (int i = 0; i < DotCount; i++)
                _dots[i] = CreateDot($"PoseDot_{i}", _poseColor);

            _shoulderCenterDot = CreateDot("ShoulderCenter", _shoulderCenterColor);
            HideAll();
        }

        void CreateBones()
        {
            if (_overlayRect == null) return;

            _bones = new RectTransform[BoneCount];
            for (int i = 0; i < BoneCount; i++)
                _bones[i] = CreateBoneLine($"Bone_{i}");
        }

        Image CreateDot(string objName, Color color)
        {
            var go = new GameObject(objName, typeof(RectTransform), typeof(Image));
            go.transform.SetParent(_overlayRect, false);

            var img = go.GetComponent<Image>();
            img.sprite = _dotSprite;
            img.color  = color;

            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.zero;
            rt.pivot     = new Vector2(0.5f, 0.5f);
            rt.sizeDelta = new Vector2(DotSize, DotSize);

            return img;
        }

        RectTransform CreateBoneLine(string objName)
        {
            var go = new GameObject(objName, typeof(RectTransform), typeof(Image));
            go.transform.SetParent(_overlayRect, false);

            var img = go.GetComponent<Image>();
            img.color = _boneColor;
            // sprite 없이 흰 사각형 (기본 UI Image)
            img.sprite = null;

            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.zero;
            rt.pivot = new Vector2(0f, 0.5f); // 왼쪽 중앙 기준

            go.SetActive(false);
            return rt;
        }

        void UpdateBone(RectTransform bone, Vector2 from, Vector2 to)
        {
            Vector2 diff = to - from;
            float length = diff.magnitude;
            float angle = Mathf.Atan2(diff.y, diff.x) * Mathf.Rad2Deg;

            bone.anchoredPosition = from;
            bone.sizeDelta = new Vector2(length, _boneWidth);
            bone.localRotation = Quaternion.Euler(0f, 0f, angle);
        }

        Vector2 NormalizedToRect(float nx, float ny)
        {
            // X: 미러 표시 보정 → 좌우 반전
            return new Vector2(
                (1f - nx) * _overlayRect.rect.width,
                ny * _overlayRect.rect.height);
        }

        void HideAll()
        {
            if (_dots != null)
                foreach (var d in _dots)
                    if (d != null) d.gameObject.SetActive(false);
            if (_shoulderCenterDot != null)
                _shoulderCenterDot.gameObject.SetActive(false);
            if (_bones != null)
                foreach (var b in _bones)
                    if (b != null) b.gameObject.SetActive(false);
        }
    }
}
