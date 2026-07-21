using UnityEngine;
using UnityEngine.UI;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// HandPoseIntegrator からの UnifiedFrameArrived を受け取り、
    /// 上体スケルトン (7点: 鼻/両肩/両肘/両手首) をオーバーレイ表示する。
    /// </summary>
    public sealed class PoseOverlayRenderer : MonoBehaviour
    {
        [SerializeField] HandPoseIntegrator _integrator;
        [SerializeField] RectTransform      _overlayRect;
        [SerializeField] Sprite             _dotSprite;
        [SerializeField] Color              _poseColor         = Color.yellow;
        [SerializeField] Color              _shoulderCenterColor = Color.red;

        public bool ShowPoseLandmarks  = true;
        public bool ShowShoulderCenter = true;
        public bool ShowAssociationLines = true;

        // 7点: 鼻(0), 左肩(11), 右肩(12), 左肘(13), 右肘(14), 左手首(15), 右手首(16)
        static readonly int[] DotIndices = { 0, 11, 12, 13, 14, 15, 16 };
        const int DotCount = 7;
        const float DotSize = 12f;

        Image[] _dots;
        Image   _shoulderCenterDot;

        // ─── MonoBehaviour ───────────────────────────────────────────────────

        void Awake()
        {
            CreateDots();
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
            // PoseFrame이 없는 프레임(HandOnly 등)은 도트 상태를 변경하지 않는다.
            // Hand/Pose 두 Provider가 같은 Unity 프레임에서 순차 발행하면
            // HandOnly가 나중에 와서 도트를 덮어쓰는 현상을 방지한다.
            if (frame.PoseFrame == null) return;

            bool show = ShowPoseLandmarks;

            for (int i = 0; i < DotCount; i++)
            {
                if (_dots[i] == null) continue;
                _dots[i].gameObject.SetActive(show);
                if (!show) continue;

                int idx = DotIndices[i];
                var point = frame.PoseFrame.AllPoints[idx];
                _dots[i].rectTransform.anchoredPosition = NormalizedToRect(point.x, point.y);
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

        Vector2 NormalizedToRect(float nx, float ny)
        {
            // X: 미러 표시 보정 → 좌우 반전
            // Y: GetPixels32()가 bottom-up 순서이므로 반전 불필요
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
        }
    }
}
