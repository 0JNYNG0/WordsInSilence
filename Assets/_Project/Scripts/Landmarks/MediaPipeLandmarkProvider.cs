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
using WordsInSilence.Recognition.Diagnostics;

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
            HandLandmarker landmarker = _handLandmarker;
            _handLandmarker = null;
            if (landmarker != null)
            {
                try
                {
                    // the package's intended path: Close() alone leaves the TaskRunner
                    // alive, and Dispose is an explicit interface implementation
                    ((System.IDisposable)landmarker).Dispose();
                }
                catch (System.Exception exception)
                {
                    // a managed failure is reported by type only; a native access
                    // violation cannot be caught here and is not claimed to be
                    Debug.LogWarning("[MediaPipeLandmarkProvider] dispose reported "
                                     + exception.GetType().Name);
                }
            }
            if (_tempTex != null)
                Destroy(_tempTex);

            ReleaseRuntimeOnce();
        }

        // exact-once shared-runtime accounting for THIS owner, so a cancelled or
        // failed initialization cannot leak a count and no path can release twice
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
            if (!_isRunning || _handLandmarker == null) return;

            if (!_timingInitialized)
            {
                _timing = LivePipelineTiming.CreateOrNull("hand_provider");
                _timingInitialized = true;
            }
            long workStartTicks = 0L;
            if (_timing != null)
            {
                // every entry is marked, including entries with no new camera frame, so a
                // blocked main thread shows up as a large entry gap
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
                    // refuse visibly instead of claiming upright correctness on an unmapped
                    // geometry; logged once per geometry change, not once per frame
                    if (!_geometryRefused || rotation != _refusedRotation
                        || mirrored != _refusedMirror)
                    {
                        Debug.LogError("[MediaPipeLandmarkProvider] "
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

            // VIDEO モードはタイムスタンプが単調増加である必要がある
            long timestampMs = (long)(Time.realtimeSinceStartup * 1000.0);

            try
            {
                // NOTE: Image(format, Texture2D) が無い場合は GetRawTextureData<byte>() +
                //       Image(format, w, h, widthStep, NativeArray<byte>) に置き換える
                using var image = new Mediapipe.Image(ImageFormat.Types.Format.Srgba, _tempTex);

                // Capture geometry at the exact moment of DetectForVideo submission
                var geometryResult = CaptureGeometryResult(wcTex, _tempTex, timestampMs);
                long detectStartTicks = _timing == null ? 0L : LivePipelineTiming.Now();
                var result = _handLandmarker.DetectForVideo(image, timestampMs);
                long dispatchStartTicks = 0L;
                if (_timing != null)
                {
                    _timing.AddSpan(TimingStage.Detect, detectStartTicks);
                    dispatchStartTicks = LivePipelineTiming.Now();
                }

                var frame = ConvertToLandmarkFrame(result);
                frame.InferenceGeometryResult = geometryResult;
                _providerFrameIndex++;
                FrameArrived?.Invoke(frame);
                if (_timing != null)
                {
                    _timing.AddSpan(TimingStage.Dispatch, dispatchStartTicks);
                    // counts only: whether a hand was found, never any coordinate
                    _timing.Increment(frame.Hands != null && frame.Hands.Length > 0
                        ? TimingCounter.Detections : TimingCounter.EmptyResults);
                    _timing.AddSpan(TimingStage.Total, workStartTicks);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[MediaPipeLandmarkProvider] DetectForVideo failed: {ex.Message}");
            }
        }

        // ─── Internal ────────────────────────────────────────────────────────

        async Task InitializeAsync(CancellationToken ct)
        {
            AcquireRuntimeOnce();

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

        Step9DGeometryAcquisitionResult CaptureGeometryResult(WebCamTexture wcTex, Texture2D tempTex, long timestampMs)
        {
            try
            {
                if (wcTex.width <= 0 || wcTex.height <= 0)
                    return Step9DGeometryAcquisitionResult.ForUnavailable(
                        Step9DGeometryAcquisitionStatus.UnavailableInvalidDimensions,
                        "hand: webcam dimensions not yet valid: " + wcTex.width + "x" + wcTex.height);

                var geometry = new Step9DInferenceFrameGeometry(
                    provenance: Step9DInferenceGeometryProvenance.ObservedAtHandInference,
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
                    "hand: " + ex.GetType().Name + ": " + ex.Message);
            }
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
                    // upright inference predicts in the flipped grid: map y back to the
                    // unchanged source-texture convention. Labels are NEVER swapped.
                    Vector3 mapped = _uprightOptIn
                        ? UprightInferencePixelBridge.ToSourceTexturePoint(
                            new Vector3(lm.x, lm.y, lm.z))
                        : new Vector3(lm.x, lm.y, lm.z);
                    points[j] = new LandmarkPoint { X = mapped.x, Y = mapped.y, Z = mapped.z };
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
