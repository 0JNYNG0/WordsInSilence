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
using WordsInSilence.Core;
using WordsInSilence.Landmarks;
using WordsInSilence.Recognition.Diagnostics;

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
        long _providerFrameIndex;

        // F51 opt-in upright inference, cached once at first frame; off by default
        bool _uprightOptIn;
        bool _uprightInitialized;
        Color32[] _uprightSource;
        Color32[] _uprightFlipped;
        bool _geometryRefused;
        int _refusedRotation = int.MinValue;
        bool _refusedMirror;

        // F54 passive telemetry; null unless WIS_LIVE_PIPELINE_TIMING=1
        LivePipelineTiming.Channel _timing;
        bool _timingInitialized;

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
            PoseLandmarker landmarker = _poseLandmarker;
            _poseLandmarker = null;
            if (landmarker != null)
            {
                try
                {
                    // Close() alone leaves the TaskRunner alive; Dispose is an
                    // explicit interface implementation on BaseVisionTaskApi
                    ((System.IDisposable)landmarker).Dispose();
                }
                catch (System.Exception exception)
                {
                    Debug.LogWarning("[PoseLandmarkProvider] dispose reported "
                                     + exception.GetType().Name);
                }
            }
            if (_tempTex != null)
                Destroy(_tempTex);
            ReleaseRuntimeOnce();
        }

        // exact-once shared-runtime accounting for THIS owner. The missing-model branch
        // and OnDestroy both call it, and the second call is a no-op, so this owner can
        // no longer shut the shared runtime down while another provider is live.
        bool _runtimeAcquired;

        void AcquireRuntimeOnce()
        {
            if (_runtimeAcquired) return;
            _runtimeAcquired = true;
            MediaPipeRuntime.EnsureInitialized();
        }

        void ReleaseRuntimeOnce()
        {
            if (!_runtimeAcquired) return;
            _runtimeAcquired = false;
            MediaPipeRuntime.Release();
        }

        void Update()
        {
            if (!_isRunning || _poseLandmarker == null) return;

            if (!_timingInitialized)
            {
                _timing = LivePipelineTiming.CreateOrNull("pose_provider");
                _timingInitialized = true;
            }
            long workStartTicks = 0L;
            if (_timing != null)
            {
                _timing.MarkEntry(Time.frameCount);
                workStartTicks = LivePipelineTiming.Now();
            }

            var wcTex = _rawImage != null ? _rawImage.texture as WebCamTexture : null;
            bool didUpdateThisFrame = wcTex != null && wcTex.didUpdateThisFrame;
            if (_timing != null)
            {
                _timing.Increment(didUpdateThisFrame
                    ? TimingCounter.FrameUpdated : TimingCounter.FrameSkipped);
                _timing.Flush();
            }
            if (!didUpdateThisFrame) return;

            if (!_uprightInitialized)
            {
                _uprightOptIn = UprightInferencePixelBridge.ReadOptIn();
                _uprightInitialized = true;
            }

            long pixelStartTicks = _timing == null ? 0L : LivePipelineTiming.Now();
            EnsureTempTexture(wcTex.width, wcTex.height);
            if (_uprightOptIn)
            {
                int rotation = wcTex.videoRotationAngle;
                bool mirrored = wcTex.videoVerticallyMirrored;
                if (!UprightInferencePixelBridge.IsSupportedCameraGeometry(rotation, mirrored))
                {
                    if (!_geometryRefused || rotation != _refusedRotation
                        || mirrored != _refusedMirror)
                    {
                        Debug.LogError("[PoseLandmarkProvider] "
                            + UprightInferencePixelBridge.DescribeUnsupportedGeometry(
                                rotation, mirrored));
                        _geometryRefused = true;
                        _refusedRotation = rotation;
                        _refusedMirror = mirrored;
                    }
                    return;
                }
                _geometryRefused = false;
                _refusedRotation = rotation;
                _refusedMirror = mirrored;
                int pixelCount = wcTex.width * wcTex.height;
                _uprightSource = UprightInferencePixelBridge.EnsureBuffer(_uprightSource, pixelCount);
                _uprightFlipped = UprightInferencePixelBridge.EnsureBuffer(_uprightFlipped, pixelCount);
                wcTex.GetPixels32(_uprightSource);
                UprightInferencePixelBridge.ReversePixelRows(
                    _uprightSource, wcTex.width, wcTex.height, _uprightFlipped);
                _tempTex.SetPixels32(_uprightFlipped);
            }
            else
            {
                _tempTex.SetPixels32(wcTex.GetPixels32());
            }
            _tempTex.Apply();
            if (_timing != null) _timing.AddSpan(TimingStage.Pixels, pixelStartTicks);

            long timestampMs = (long)(Time.realtimeSinceStartup * 1000.0);

            try
            {
                using var image = new Mediapipe.Image(ImageFormat.Types.Format.Srgba, _tempTex);

                // Capture geometry at the exact moment of DetectForVideo submission
                var geometryResult = CaptureGeometryResult(wcTex, _tempTex, timestampMs);
                long detectStartTicks = _timing == null ? 0L : LivePipelineTiming.Now();
                var result = _poseLandmarker.DetectForVideo(image, timestampMs);
                long dispatchStartTicks = 0L;
                if (_timing != null)
                {
                    _timing.AddSpan(TimingStage.Detect, detectStartTicks);
                    dispatchStartTicks = LivePipelineTiming.Now();
                }
                var frame  = ConvertToUpperBodyPoseFrame(result, timestampMs);
                if (frame != null)
                {
                    frame.InferenceGeometryResult = geometryResult;
                    _providerFrameIndex++;
                    PoseFrameArrived?.Invoke(frame);
                }
                if (_timing != null)
                {
                    _timing.AddSpan(TimingStage.Dispatch, dispatchStartTicks);
                    _timing.Increment(frame != null
                        ? TimingCounter.Detections : TimingCounter.EmptyResults);
                    _timing.AddSpan(TimingStage.Total, workStartTicks);
                }
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
            AcquireRuntimeOnce();

            if (ct.IsCancellationRequested) return;

            string modelPath = ResolveModelPath("pose_landmarker_lite.bytes");
            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                Debug.LogError(
                    "[PoseLandmarkProvider] pose_landmarker_lite.bytes が見つかりません。\n" +
                    "Editor: Packages/com.github.homuler.mediapipe/PackageResources/MediaPipe/\n" +
                    "Build:  StreamingAssets/pose_landmarker_lite.bytes\n" +
                    "解決パス: " + modelPath);
                ReleaseRuntimeOnce();
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

        Step9DGeometryAcquisitionResult CaptureGeometryResult(WebCamTexture wcTex, Texture2D tempTex, long timestampMs)
        {
            try
            {
                if (wcTex.width <= 0 || wcTex.height <= 0)
                    return Step9DGeometryAcquisitionResult.ForUnavailable(
                        Step9DGeometryAcquisitionStatus.UnavailableInvalidDimensions,
                        "pose: webcam dimensions not yet valid: " + wcTex.width + "x" + wcTex.height);

                var geometry = new Step9DInferenceFrameGeometry(
                    provenance: Step9DInferenceGeometryProvenance.ObservedAtPoseInference,
                    requestedWidthOrNull: null,
                    requestedHeightOrNull: null,
                    requestedFpsOrNull: null,
                    observedWebcamWidth: wcTex.width,
                    observedWebcamHeight: wcTex.height,
                    observedVideoRotationAngleOrNull: wcTex.videoRotationAngle,
                    observedVideoVerticallyMirroredOrNull: wcTex.videoVerticallyMirrored,
                    inferenceTextureWidth: tempTex.width,
                    inferenceTextureHeight: tempTex.height,
                    inferenceImageWidth: tempTex.width,
                    inferenceImageHeight: tempTex.height,
                    imageFormat: "SRGBA",
                    preInferenceHorizontalFlip: Step9DPreInferenceTransformState.None,
                    preInferenceVerticalFlip: _uprightOptIn
                        ? Step9DPreInferenceTransformState.Applied
                        : Step9DPreInferenceTransformState.None,
                    preInferenceRotationDegreesOrNull: null,
                    crop: Step9DPreInferenceTransformState.None,
                    resize: Step9DPreInferenceTransformState.None,
                    letterbox: Step9DPreInferenceTransformState.None,
                    providerFrameIndexOrNull: _providerFrameIndex,
                    mediaPipeTimestampMs: timestampMs,
                    unityFrameCountOrNull: Time.frameCount,
                    pixelBufferRowOrder: _uprightOptIn
                        ? UprightInferencePixelBridge.RowOrderTopDown
                        : UprightInferencePixelBridge.RowOrderBottomUp);
                return Step9DGeometryAcquisitionResult.ForAvailable(geometry);
            }
            catch (Exception ex)
            {
                return Step9DGeometryAcquisitionResult.ForUnavailable(
                    Step9DGeometryAcquisitionStatus.UnavailableCaptureException,
                    "pose: " + ex.GetType().Name + ": " + ex.Message);
            }
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
                // upright inference predicts in the flipped grid: map y back to the unchanged
                // source-texture convention. All 33 points, z untouched.
                points[i] = _uprightOptIn
                    ? UprightInferencePixelBridge.ToSourceTexturePoint(
                        new Vector3(lm.x, lm.y, lm.z))
                    : new Vector3(lm.x, lm.y, lm.z);
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
