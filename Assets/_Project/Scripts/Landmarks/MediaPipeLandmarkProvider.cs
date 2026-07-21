using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Mediapipe;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.Core;
using Mediapipe.Tasks.Vision.HandLandmarker;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Core;

namespace WordsInSilence.Landmarks
{
    /// <summary>
    /// MediaPipe Hand Landmarker を使って ICameraFrameSource パイプラインに
    /// 接続し、手の 21 点ランドマークを毎フレーム検出する。
    ///
    /// Inspector の RawImage には WebCamSpikeController が設定している
    /// WebCamTexture が流れているものを割り当てること。
    /// </summary>
    public sealed class MediaPipeLandmarkProvider : MonoBehaviour, ILandmarkProvider
    {
        [Header("References")]
        [SerializeField] RawImage _rawImage;

        [Header("Detection Settings")]
        [SerializeField] int _numHands = 2;
        [SerializeField] float _minDetectionConfidence = 0.5f;
        [SerializeField] float _minTrackingConfidence = 0.5f;

        HandLandmarker _handLandmarker;
        Texture2D _tempTex;
        bool _isRunning;

        // ─── ILandmarkProvider ───────────────────────────────────────────────

        public bool IsRunning => _isRunning;
        public event Action<LandmarkFrame> FrameArrived;

        public async Task StartAsync(CancellationToken ct)
        {
            if (_isRunning) return;

            if (_handLandmarker == null)
                await InitializeAsync(ct);
            else
                _isRunning = true;
        }

        public Task StopAsync(CancellationToken ct)
        {
            _isRunning = false;
            return Task.CompletedTask;
        }

        // ─── MonoBehaviour ───────────────────────────────────────────────────

        void Start()
        {
            _ = StartAsync(destroyCancellationToken);
        }

        void OnDestroy()
        {
            _isRunning = false;
            _handLandmarker?.Close();
            _handLandmarker = null;
            if (_tempTex != null)
                Destroy(_tempTex);

            MediaPipeRuntime.Release();
        }

        void Update()
        {
            if (!_isRunning || _handLandmarker == null) return;

            var wcTex = _rawImage != null ? _rawImage.texture as WebCamTexture : null;
            if (wcTex == null || !wcTex.didUpdateThisFrame) return;

            EnsureTempTexture(wcTex.width, wcTex.height);
            _tempTex.SetPixels32(wcTex.GetPixels32());
            _tempTex.Apply();

            // VIDEO モードはタイムスタンプが単調増加である必要がある
            long timestampMs = (long)(Time.realtimeSinceStartup * 1000.0);

            try
            {
                // NOTE: Image(format, Texture2D) が無い場合は GetRawTextureData<byte>() +
                //       Image(format, w, h, widthStep, NativeArray<byte>) に置き換える
                using var image = new Mediapipe.Image(ImageFormat.Types.Format.Srgba, _tempTex);
                var result = _handLandmarker.DetectForVideo(image, timestampMs);
                FrameArrived?.Invoke(ConvertToLandmarkFrame(result));
            }
            catch (Exception ex)
            {
                Debug.LogError($"[MediaPipeLandmarkProvider] DetectForVideo failed: {ex.Message}");
            }
        }

        // ─── Internal ────────────────────────────────────────────────────────

        async Task InitializeAsync(CancellationToken ct)
        {
            MediaPipeRuntime.EnsureInitialized();

            if (ct.IsCancellationRequested) return;

            string modelPath = ResolveModelPath();
            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                Debug.LogError(
                    "[MediaPipeLandmarkProvider] hand_landmarker.bytes が見つかりません。\n" +
                    "Editor: Packages/com.github.homuler.mediapipe/PackageResources/MediaPipe/\n" +
                    "Build:  StreamingAssets/hand_landmarker.bytes\n" +
                    "解決パス: " + modelPath);
                return;
            }

            // ファイル I/O はスレッドプールで実行してメインスレッドをブロックしない
            byte[] modelBytes = await Task.Run(() => File.ReadAllBytes(modelPath), ct);
            if (ct.IsCancellationRequested) return;

            // CreateFromOptions はメインスレッドで呼ぶ (await 後は UnitySynchronizationContext で復帰)
            var options = new HandLandmarkerOptions(
                baseOptions: new BaseOptions(modelAssetBuffer: modelBytes),
                runningMode: RunningMode.VIDEO,
                numHands: _numHands,
                minHandDetectionConfidence: _minDetectionConfidence,
                minHandPresenceConfidence: _minDetectionConfidence,
                minTrackingConfidence: _minTrackingConfidence);

            // NOTE: GPU を使わないため gpuResources: null
            //       gpuResources パラメータが無い場合は引数を省略すること
            _handLandmarker = HandLandmarker.CreateFromOptions(options, gpuResources: null);
            _isRunning = true;

            Debug.Log("[MediaPipeLandmarkProvider] HandLandmarker 初期化完了。");
        }

        static string ResolveModelPath()
        {
            const string ModelFileName = "hand_landmarker.bytes";

#if UNITY_EDITOR
            // プロジェクトルート / Packages / ... に配置されたモデルを参照
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string editorPath = Path.Combine(
                projectRoot,
                "Packages/com.github.homuler.mediapipe/PackageResources/MediaPipe",
                ModelFileName);
            if (File.Exists(editorPath)) return editorPath;
#endif
            // ビルド時は StreamingAssets に手動コピーが必要
            return Path.Combine(Application.streamingAssetsPath, ModelFileName);
        }

        void EnsureTempTexture(int width, int height)
        {
            if (_tempTex != null && _tempTex.width == width && _tempTex.height == height)
                return;
            if (_tempTex != null) Destroy(_tempTex);
            _tempTex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        }

        LandmarkFrame ConvertToLandmarkFrame(HandLandmarkerResult result)
        {
            int handCount = result.handLandmarks?.Count ?? 0;
            var hands = new HandLandmarks[handCount];

            for (int i = 0; i < handCount; i++)
            {
                var landmarks = result.handLandmarks[i];
                int pointCount = landmarks.landmarks?.Count ?? 0;
                var points = new LandmarkPoint[pointCount];

                for (int j = 0; j < pointCount; j++)
                {
                    var lm = landmarks.landmarks[j];
                    points[j] = new LandmarkPoint { X = lm.x, Y = lm.y, Z = lm.z };
                }

                bool isLeft = false;
                float confidence = 1f;

                if (result.handedness != null && result.handedness.Count > i)
                {
                    var cats = result.handedness[i].categories;
                    if (cats != null && cats.Count > 0)
                    {
                        isLeft = cats[0].categoryName == "Left";
                        confidence = cats[0].score;
                    }
                }

                hands[i] = new HandLandmarks
                {
                    IsLeftHand = isLeft,
                    Points = points,
                    Confidence = confidence
                };
            }

            return new LandmarkFrame
            {
                Timestamp = Time.realtimeSinceStartup,
                Hands = hands
            };
        }
    }
}
