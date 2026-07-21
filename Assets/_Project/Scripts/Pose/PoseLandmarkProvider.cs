using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Mediapipe;
using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Vision.Core;
using Mediapipe.Tasks.Vision.PoseLandmarker;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Landmarks;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// MediaPipe Pose Landmarker を使って毎フレーム上体ポーズを検出する。
    /// RawImage に流れている WebCamTexture を直接参照する (ICameraFrameSource 非使用)。
    /// </summary>
    public sealed class PoseLandmarkProvider : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] RawImage _rawImage;

        [Header("Detection Settings")]
        [SerializeField] float _minPoseDetectionConfidence = 0.5f;
        [SerializeField] float _minPosePresenceConfidence  = 0.5f;
        [SerializeField] float _minTrackingConfidence      = 0.5f;

        PoseLandmarker _poseLandmarker;
        Texture2D _tempTex;
        bool _isRunning;

        public bool IsRunning => _isRunning;

        /// <summary>新しいポーズフレームが到着したときに発生。null フレームは発行しない。</summary>
        public event Action<UpperBodyPoseFrame> PoseFrameArrived;

        // ─── MonoBehaviour ───────────────────────────────────────────────────

        void Start()
        {
            _ = InitializeAsync(destroyCancellationToken);
        }

        void OnDestroy()
        {
            _isRunning = false;
            _poseLandmarker?.Close();
            _poseLandmarker = null;
            if (_tempTex != null)
                Destroy(_tempTex);
            MediaPipeRuntime.Release();
        }

        void Update()
        {
            if (!_isRunning || _poseLandmarker == null) return;

            var wcTex = _rawImage != null ? _rawImage.texture as WebCamTexture : null;
            if (wcTex == null || !wcTex.didUpdateThisFrame) return;

            EnsureTempTexture(wcTex.width, wcTex.height);
            _tempTex.SetPixels32(wcTex.GetPixels32());
            _tempTex.Apply();

            long timestampMs = (long)(Time.realtimeSinceStartup * 1000.0);

            try
            {
                using var image = new Mediapipe.Image(ImageFormat.Types.Format.Srgba, _tempTex);
                var result = _poseLandmarker.DetectForVideo(image, timestampMs);
                var frame  = ConvertToUpperBodyPoseFrame(result, timestampMs);
                if (frame != null)
                    PoseFrameArrived?.Invoke(frame);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[PoseLandmarkProvider] DetectForVideo failed: {ex.Message}");
            }
        }

        // ─── Internal ────────────────────────────────────────────────────────

        async Task InitializeAsync(CancellationToken ct)
        {
            // Glog は MediaPipeRuntime の共有レファレンスカウンタで管理
            MediaPipeRuntime.EnsureInitialized();

            if (ct.IsCancellationRequested) return;

            string modelPath = ResolveModelPath("pose_landmarker_lite.bytes");
            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                Debug.LogError(
                    "[PoseLandmarkProvider] pose_landmarker_lite.bytes が見つかりません。\n" +
                    "Editor: Packages/com.github.homuler.mediapipe/PackageResources/MediaPipe/\n" +
                    "Build:  StreamingAssets/pose_landmarker_lite.bytes\n" +
                    "解決パス: " + modelPath);
                MediaPipeRuntime.Release();
                return;
            }

            byte[] modelBytes = await Task.Run(() => File.ReadAllBytes(modelPath), ct);
            if (ct.IsCancellationRequested) return;

            var options = new PoseLandmarkerOptions(
                baseOptions: new BaseOptions(modelAssetBuffer: modelBytes),
                runningMode: RunningMode.VIDEO,
                numPoses: 1,
                minPoseDetectionConfidence: _minPoseDetectionConfidence,
                minPosePresenceConfidence:  _minPosePresenceConfidence,
                minTrackingConfidence:      _minTrackingConfidence,
                outputSegmentationMasks:    false);

            _poseLandmarker = PoseLandmarker.CreateFromOptions(options, gpuResources: null);
            _isRunning = true;

            Debug.Log("[PoseLandmarkProvider] PoseLandmarker 初期化完了。");
        }

        static string ResolveModelPath(string fileName)
        {
#if UNITY_EDITOR
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string editorPath  = Path.Combine(
                projectRoot,
                "Packages/com.github.homuler.mediapipe/PackageResources/MediaPipe",
                fileName);
            if (File.Exists(editorPath)) return editorPath;
#endif
            return Path.Combine(Application.streamingAssetsPath, fileName);
        }

        void EnsureTempTexture(int width, int height)
        {
            if (_tempTex != null && _tempTex.width == width && _tempTex.height == height)
                return;
            if (_tempTex != null) Destroy(_tempTex);
            _tempTex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        }

        UpperBodyPoseFrame ConvertToUpperBodyPoseFrame(PoseLandmarkerResult result, long timestampMs)
        {
            if (result.poseLandmarks == null || result.poseLandmarks.Count == 0)
                return null;

            var landmarks = result.poseLandmarks[0];
            if (landmarks.landmarks == null || landmarks.landmarks.Count < 33)
                return null;

            var points     = new Vector3[33];
            var visibility = new float[33];

            for (int i = 0; i < 33; i++)
            {
                var lm    = landmarks.landmarks[i];
                points[i] = new Vector3(lm.x, lm.y, lm.z);
                visibility[i] = lm.visibility ?? 0f;
            }

            return new UpperBodyPoseFrame
            {
                Timestamp   = Time.realtimeSinceStartup,
                TimestampMs = timestampMs,
                AllPoints   = points,
                Visibility  = visibility
            };
        }
    }
}
