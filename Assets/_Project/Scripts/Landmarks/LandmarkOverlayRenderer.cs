using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Core;

namespace WordsInSilence.Landmarks
{
    /// <summary>
    /// ILandmarkProvider.FrameArrived を購読し、RectTransform 上に
    /// 21 点のドットオーバーレイを描画する。
    ///
    /// OverlayRect は RawImage_Feed と同じ Rect サイズ・ピボット(center)を持つ
    /// 透明な UI レイヤーを指定すること。
    /// </summary>
    public sealed class LandmarkOverlayRenderer : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] MediaPipeLandmarkProvider _provider;
        [SerializeField] RectTransform _overlayRect;
        [SerializeField] Sprite _dotSprite;
        [SerializeField] TMP_Text _textDetectionRate;

        [Header("Visual Settings")]
        [SerializeField] float _dotSize = 8f;

        static readonly Color LeftHandColor  = new Color(0.25f, 0.45f, 1.00f, 1f);
        static readonly Color RightHandColor = new Color(0.20f, 0.85f, 0.35f, 1f);

        const int MaxHands       = 2;
        const int PointsPerHand  = 21;
        const int RollingWindow  = 100;

        // Dot pool: 2 hands × 21 points = 42
        readonly List<RectTransform> _dotRects  = new List<RectTransform>(MaxHands * PointsPerHand);
        readonly List<Image>         _dotImages = new List<Image>(MaxHands * PointsPerHand);

        // Rolling detection rate (circular queue)
        readonly Queue<bool> _detectionQueue = new Queue<bool>(RollingWindow);
        int _detectedCount;

        // ─── MonoBehaviour ───────────────────────────────────────────────────

        void OnEnable()
        {
            if (_provider != null)
                _provider.FrameArrived += OnFrameArrived;

            BuildDotPool();
        }

        void OnDisable()
        {
            if (_provider != null)
                _provider.FrameArrived -= OnFrameArrived;

            DestroyDotPool();
        }

        // ─── Internal ────────────────────────────────────────────────────────

        void BuildDotPool()
        {
            if (_overlayRect == null) return;

            int total = MaxHands * PointsPerHand;
            for (int i = 0; i < total; i++)
            {
                var go  = new GameObject($"Dot_{i}", typeof(Image));
                go.transform.SetParent(_overlayRect, worldPositionStays: false);

                var img = go.GetComponent<Image>();
                img.sprite = _dotSprite;
                img.raycastTarget = false;

                var rt = go.GetComponent<RectTransform>();
                rt.sizeDelta = new Vector2(_dotSize, _dotSize);
                rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0.5f);

                go.SetActive(false);

                _dotRects.Add(rt);
                _dotImages.Add(img);
            }
        }

        void DestroyDotPool()
        {
            for (int i = 0; i < _dotRects.Count; i++)
            {
                if (_dotRects[i] != null)
                    Destroy(_dotRects[i].gameObject);
            }
            _dotRects.Clear();
            _dotImages.Clear();
        }

        void OnFrameArrived(LandmarkFrame frame)
        {
            // FrameArrived は Update() から呼ばれるため、Main スレッド上で安全に UI 操作できる

            // 全ドットを非表示にリセット
            for (int i = 0; i < _dotRects.Count; i++)
                _dotRects[i].gameObject.SetActive(false);

            // 検出率の rolling window 更新
            bool detected = frame.Hands != null && frame.Hands.Length > 0;
            _detectionQueue.Enqueue(detected);
            if (detected) _detectedCount++;

            if (_detectionQueue.Count > RollingWindow)
            {
                if (_detectionQueue.Dequeue()) _detectedCount--;
            }

            if (_textDetectionRate != null)
            {
                float rate = _detectionQueue.Count > 0
                    ? (float)_detectedCount / _detectionQueue.Count * 100f
                    : 0f;
                _textDetectionRate.text = $"検出率: {rate:F1}%";
            }

            if (frame.Hands == null || _overlayRect == null) return;

            Rect rect = _overlayRect.rect;
            int dotIdx = 0;

            for (int h = 0; h < frame.Hands.Length && h < MaxHands; h++)
            {
                var hand  = frame.Hands[h];
                Color col = hand.IsLeftHand ? LeftHandColor : RightHandColor;

                if (hand.Points == null) continue;

                for (int p = 0; p < hand.Points.Length && p < PointsPerHand; p++)
                {
                    if (dotIdx >= _dotRects.Count) break;

                    var pt = hand.Points[p];

                    // 정규화 좌표 (0~1) → RectTransform 로컬 좌표 (중심 원점)
                    // X: 카메라 미러 표시로 인해 좌우 반전 필요
                    // Y: GetPixels32() bottom-up 순서로 인해 반전 불필요
                    float px = (1f - pt.X) * rect.width  - rect.width  * 0.5f;
                    float py = pt.Y * rect.height - rect.height * 0.5f;

                    _dotRects[dotIdx].anchoredPosition = new Vector2(px, py);
                    _dotRects[dotIdx].gameObject.SetActive(true);
                    _dotImages[dotIdx].color = col;
                    dotIdx++;
                }
            }
        }
    }
}
