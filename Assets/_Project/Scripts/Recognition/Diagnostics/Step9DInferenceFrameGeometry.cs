using System;
using System.Globalization;

namespace WordsInSilence.Recognition.Diagnostics
{
    public enum Step9DInferenceGeometryProvenance
    {
        ObservedAtHandInference,
        ObservedAtPoseInference,
        TestInjected,
    }

    public enum Step9DPreInferenceTransformState
    {
        Unknown,
        None,
        Applied,
    }

    public enum Step9DGeometryAcquisitionStatus
    {
        Available,
        UnavailableBeforeFrameReady,
        UnavailableInvalidDimensions,
        UnavailableCaptureException,
    }

    /// <summary>
    /// Typed result of geometry acquisition at a provider's DetectForVideo site.
    /// When Available, Geometry is non-null. Otherwise, DiagnosticCode describes why.
    /// </summary>
    public sealed class Step9DGeometryAcquisitionResult
    {
        public Step9DGeometryAcquisitionStatus Status { get; }
        public Step9DInferenceFrameGeometry Geometry { get; }
        public string DiagnosticCode { get; }

        Step9DGeometryAcquisitionResult(
            Step9DGeometryAcquisitionStatus status,
            Step9DInferenceFrameGeometry geometry,
            string diagnosticCode)
        {
            Status = status;
            Geometry = geometry;
            DiagnosticCode = diagnosticCode ?? string.Empty;
        }

        public static Step9DGeometryAcquisitionResult ForAvailable(Step9DInferenceFrameGeometry geometry)
        {
            if (geometry == null)
                throw new ArgumentNullException(nameof(geometry));
            return new Step9DGeometryAcquisitionResult(
                Step9DGeometryAcquisitionStatus.Available, geometry, string.Empty);
        }

        public static Step9DGeometryAcquisitionResult ForUnavailable(
            Step9DGeometryAcquisitionStatus status, string diagnosticCode)
        {
            if (status == Step9DGeometryAcquisitionStatus.Available)
                throw new ArgumentException("use ForAvailable for Available status", nameof(status));
            return new Step9DGeometryAcquisitionResult(
                status, null, diagnosticCode ?? string.Empty);
        }

        internal Step9DGeometryAcquisitionResult Clone()
        {
            return new Step9DGeometryAcquisitionResult(
                Status, Geometry?.Clone(), DiagnosticCode);
        }
    }

    /// <summary>
    /// Immutable record of actual inference image geometry captured at the exact
    /// moment pixels were submitted to MediaPipe DetectForVideo. Created by the
    /// provider that owns the WebCamTexture and Texture2D at inference time.
    /// </summary>
    public sealed class Step9DInferenceFrameGeometry
    {
        public Step9DInferenceGeometryProvenance Provenance { get; }

        public int? RequestedWidthOrNull { get; }
        public int? RequestedHeightOrNull { get; }
        public int? RequestedFpsOrNull { get; }

        public int ObservedWebcamWidth { get; }
        public int ObservedWebcamHeight { get; }
        public int? ObservedVideoRotationAngleOrNull { get; }
        public bool? ObservedVideoVerticallyMirroredOrNull { get; }

        public int InferenceTextureWidth { get; }
        public int InferenceTextureHeight { get; }
        public int InferenceImageWidth { get; }
        public int InferenceImageHeight { get; }
        public string ImageFormat { get; }

        public Step9DPreInferenceTransformState PreInferenceHorizontalFlip { get; }
        public Step9DPreInferenceTransformState PreInferenceVerticalFlip { get; }
        public int? PreInferenceRotationDegreesOrNull { get; }
        public Step9DPreInferenceTransformState Crop { get; }
        public Step9DPreInferenceTransformState Resize { get; }
        public Step9DPreInferenceTransformState Letterbox { get; }

        public long? ProviderFrameIndexOrNull { get; }
        public long MediaPipeTimestampMs { get; }
        public int? UnityFrameCountOrNull { get; }

        /// <summary>
        /// Source-contract: pixel-buffer row order as determined from source code analysis.
        /// This is NOT a runtime-observed property but a typed source-code contract.
        /// </summary>
        public string PixelBufferRowOrder { get; }

        public Step9DInferenceFrameGeometry(
            Step9DInferenceGeometryProvenance provenance,
            int? requestedWidthOrNull,
            int? requestedHeightOrNull,
            int? requestedFpsOrNull,
            int observedWebcamWidth,
            int observedWebcamHeight,
            int? observedVideoRotationAngleOrNull,
            bool? observedVideoVerticallyMirroredOrNull,
            int inferenceTextureWidth,
            int inferenceTextureHeight,
            int inferenceImageWidth,
            int inferenceImageHeight,
            string imageFormat,
            Step9DPreInferenceTransformState preInferenceHorizontalFlip,
            Step9DPreInferenceTransformState preInferenceVerticalFlip,
            int? preInferenceRotationDegreesOrNull,
            Step9DPreInferenceTransformState crop,
            Step9DPreInferenceTransformState resize,
            Step9DPreInferenceTransformState letterbox,
            long? providerFrameIndexOrNull,
            long mediaPipeTimestampMs,
            int? unityFrameCountOrNull,
            string pixelBufferRowOrder)
        {
            if (provenance != Step9DInferenceGeometryProvenance.TestInjected)
            {
                if (observedWebcamWidth <= 0)
                    throw new Step9DRawIntermediateDiagnosticException(
                        Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                        "observed webcam width must be positive for non-test provenance");
                if (observedWebcamHeight <= 0)
                    throw new Step9DRawIntermediateDiagnosticException(
                        Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                        "observed webcam height must be positive for non-test provenance");
            }
            if (inferenceTextureWidth <= 0)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                    "inference texture width must be positive");
            if (inferenceTextureHeight <= 0)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                    "inference texture height must be positive");
            if (inferenceImageWidth <= 0)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                    "inference image width must be positive");
            if (inferenceImageHeight <= 0)
                throw new Step9DRawIntermediateDiagnosticException(
                    Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE,
                    "inference image height must be positive");
            if (string.IsNullOrWhiteSpace(imageFormat))
                throw new ArgumentException("image format is required", nameof(imageFormat));

            Provenance = provenance;
            RequestedWidthOrNull = requestedWidthOrNull;
            RequestedHeightOrNull = requestedHeightOrNull;
            RequestedFpsOrNull = requestedFpsOrNull;
            ObservedWebcamWidth = observedWebcamWidth;
            ObservedWebcamHeight = observedWebcamHeight;
            ObservedVideoRotationAngleOrNull = observedVideoRotationAngleOrNull;
            ObservedVideoVerticallyMirroredOrNull = observedVideoVerticallyMirroredOrNull;
            InferenceTextureWidth = inferenceTextureWidth;
            InferenceTextureHeight = inferenceTextureHeight;
            InferenceImageWidth = inferenceImageWidth;
            InferenceImageHeight = inferenceImageHeight;
            ImageFormat = imageFormat;
            PreInferenceHorizontalFlip = preInferenceHorizontalFlip;
            PreInferenceVerticalFlip = preInferenceVerticalFlip;
            PreInferenceRotationDegreesOrNull = preInferenceRotationDegreesOrNull;
            Crop = crop;
            Resize = resize;
            Letterbox = letterbox;
            ProviderFrameIndexOrNull = providerFrameIndexOrNull;
            MediaPipeTimestampMs = mediaPipeTimestampMs;
            UnityFrameCountOrNull = unityFrameCountOrNull;
            PixelBufferRowOrder = pixelBufferRowOrder ?? "unknown";
        }

        internal Step9DInferenceFrameGeometry Clone() => new Step9DInferenceFrameGeometry(
            Provenance,
            RequestedWidthOrNull, RequestedHeightOrNull, RequestedFpsOrNull,
            ObservedWebcamWidth, ObservedWebcamHeight,
            ObservedVideoRotationAngleOrNull, ObservedVideoVerticallyMirroredOrNull,
            InferenceTextureWidth, InferenceTextureHeight,
            InferenceImageWidth, InferenceImageHeight, ImageFormat,
            PreInferenceHorizontalFlip, PreInferenceVerticalFlip,
            PreInferenceRotationDegreesOrNull, Crop, Resize, Letterbox,
            ProviderFrameIndexOrNull, MediaPipeTimestampMs,
            UnityFrameCountOrNull, PixelBufferRowOrder);

        /// <summary>
        /// Full contract comparison. Compares all observable fields that affect
        /// pixel semantics. Provenance, requested values, timestamps and frame
        /// indices are not compared (they differ by design between hand/pose).
        /// Unknown != None.
        /// </summary>
        public bool IsFullContractCompatible(Step9DInferenceFrameGeometry other)
        {
            if (other == null) return false;
            if (ObservedWebcamWidth != other.ObservedWebcamWidth) return false;
            if (ObservedWebcamHeight != other.ObservedWebcamHeight) return false;
            if (ObservedVideoRotationAngleOrNull.HasValue && other.ObservedVideoRotationAngleOrNull.HasValue
                && ObservedVideoRotationAngleOrNull.Value != other.ObservedVideoRotationAngleOrNull.Value) return false;
            if (ObservedVideoVerticallyMirroredOrNull.HasValue && other.ObservedVideoVerticallyMirroredOrNull.HasValue
                && ObservedVideoVerticallyMirroredOrNull.Value != other.ObservedVideoVerticallyMirroredOrNull.Value) return false;
            if (InferenceTextureWidth != other.InferenceTextureWidth) return false;
            if (InferenceTextureHeight != other.InferenceTextureHeight) return false;
            if (InferenceImageWidth != other.InferenceImageWidth) return false;
            if (InferenceImageHeight != other.InferenceImageHeight) return false;
            if (!string.Equals(ImageFormat, other.ImageFormat, StringComparison.Ordinal)) return false;
            if (PreInferenceHorizontalFlip != other.PreInferenceHorizontalFlip) return false;
            if (PreInferenceVerticalFlip != other.PreInferenceVerticalFlip) return false;
            if (PreInferenceRotationDegreesOrNull.HasValue && other.PreInferenceRotationDegreesOrNull.HasValue
                && PreInferenceRotationDegreesOrNull.Value != other.PreInferenceRotationDegreesOrNull.Value) return false;
            if (Crop != other.Crop) return false;
            if (Resize != other.Resize) return false;
            if (Letterbox != other.Letterbox) return false;
            if (!string.Equals(PixelBufferRowOrder, other.PixelBufferRowOrder, StringComparison.Ordinal)) return false;
            return true;
        }

        /// <summary>
        /// Returns a description of the first contract field that differs, or null if compatible.
        /// </summary>
        public string DescribeFirstMismatch(Step9DInferenceFrameGeometry other)
        {
            if (other == null) return "other is null";
            if (ObservedWebcamWidth != other.ObservedWebcamWidth) return F("observed_webcam_width", ObservedWebcamWidth, other.ObservedWebcamWidth);
            if (ObservedWebcamHeight != other.ObservedWebcamHeight) return F("observed_webcam_height", ObservedWebcamHeight, other.ObservedWebcamHeight);
            if (ObservedVideoRotationAngleOrNull.HasValue && other.ObservedVideoRotationAngleOrNull.HasValue
                && ObservedVideoRotationAngleOrNull.Value != other.ObservedVideoRotationAngleOrNull.Value)
                return F("observed_video_rotation_angle", ObservedVideoRotationAngleOrNull.Value, other.ObservedVideoRotationAngleOrNull.Value);
            if (ObservedVideoVerticallyMirroredOrNull.HasValue && other.ObservedVideoVerticallyMirroredOrNull.HasValue
                && ObservedVideoVerticallyMirroredOrNull.Value != other.ObservedVideoVerticallyMirroredOrNull.Value)
                return "observed_video_vertically_mirrored: " + ObservedVideoVerticallyMirroredOrNull.Value + " vs " + other.ObservedVideoVerticallyMirroredOrNull.Value;
            if (InferenceTextureWidth != other.InferenceTextureWidth) return F("inference_texture_width", InferenceTextureWidth, other.InferenceTextureWidth);
            if (InferenceTextureHeight != other.InferenceTextureHeight) return F("inference_texture_height", InferenceTextureHeight, other.InferenceTextureHeight);
            if (InferenceImageWidth != other.InferenceImageWidth) return F("inference_image_width", InferenceImageWidth, other.InferenceImageWidth);
            if (InferenceImageHeight != other.InferenceImageHeight) return F("inference_image_height", InferenceImageHeight, other.InferenceImageHeight);
            if (!string.Equals(ImageFormat, other.ImageFormat, StringComparison.Ordinal)) return "image_format: " + ImageFormat + " vs " + other.ImageFormat;
            if (PreInferenceHorizontalFlip != other.PreInferenceHorizontalFlip) return "pre_inference_horizontal_flip: " + PreInferenceHorizontalFlip + " vs " + other.PreInferenceHorizontalFlip;
            if (PreInferenceVerticalFlip != other.PreInferenceVerticalFlip) return "pre_inference_vertical_flip: " + PreInferenceVerticalFlip + " vs " + other.PreInferenceVerticalFlip;
            if (PreInferenceRotationDegreesOrNull.HasValue && other.PreInferenceRotationDegreesOrNull.HasValue
                && PreInferenceRotationDegreesOrNull.Value != other.PreInferenceRotationDegreesOrNull.Value)
                return F("pre_inference_rotation_degrees", PreInferenceRotationDegreesOrNull.Value, other.PreInferenceRotationDegreesOrNull.Value);
            if (Crop != other.Crop) return "crop: " + Crop + " vs " + other.Crop;
            if (Resize != other.Resize) return "resize: " + Resize + " vs " + other.Resize;
            if (Letterbox != other.Letterbox) return "letterbox: " + Letterbox + " vs " + other.Letterbox;
            if (!string.Equals(PixelBufferRowOrder, other.PixelBufferRowOrder, StringComparison.Ordinal))
                return "pixel_buffer_row_order: " + PixelBufferRowOrder + " vs " + other.PixelBufferRowOrder;
            return null;
        }

        static string F(string field, int a, int b) => field + ": " + a.ToString(CultureInfo.InvariantCulture) + " vs " + b.ToString(CultureInfo.InvariantCulture);

        /// <summary>Backward-compatible dimensional comparison (subset of full contract).</summary>
        public bool IsDimensionallyCompatible(Step9DInferenceFrameGeometry other)
        {
            if (other == null) return false;
            return ObservedWebcamWidth == other.ObservedWebcamWidth
                && ObservedWebcamHeight == other.ObservedWebcamHeight
                && InferenceTextureWidth == other.InferenceTextureWidth
                && InferenceTextureHeight == other.InferenceTextureHeight
                && InferenceImageWidth == other.InferenceImageWidth
                && InferenceImageHeight == other.InferenceImageHeight
                && string.Equals(ImageFormat, other.ImageFormat, StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// Merged hand+pose geometry result from the integrator. Carries both acquisition
    /// results, full contract compatibility verdict, and timestamp skew.
    /// </summary>
    public sealed class Step9DMergedInferenceGeometry
    {
        public Step9DGeometryAcquisitionResult HandResult { get; }
        public Step9DGeometryAcquisitionResult PoseResult { get; }

        public Step9DInferenceFrameGeometry Hand => HandResult?.Geometry;
        public Step9DInferenceFrameGeometry Pose => PoseResult?.Geometry;

        public bool BothAvailable { get; }
        public bool FullContractCompatible { get; }
        public string FirstMismatchDescription { get; }
        public long? HandPoseTimestampSkewMs { get; }

        /// <summary>Backward compat.</summary>
        public bool BothPresent => BothAvailable;
        /// <summary>Backward compat.</summary>
        public bool DimensionallyCompatible => FullContractCompatible;

        public Step9DMergedInferenceGeometry(
            Step9DGeometryAcquisitionResult handResult,
            Step9DGeometryAcquisitionResult poseResult)
        {
            HandResult = handResult;
            PoseResult = poseResult;

            bool handAvailable = handResult != null && handResult.Status == Step9DGeometryAcquisitionStatus.Available;
            bool poseAvailable = poseResult != null && poseResult.Status == Step9DGeometryAcquisitionStatus.Available;
            BothAvailable = handAvailable && poseAvailable;

            if (BothAvailable)
            {
                FullContractCompatible = handResult.Geometry.IsFullContractCompatible(poseResult.Geometry);
                FirstMismatchDescription = handResult.Geometry.DescribeFirstMismatch(poseResult.Geometry);
                HandPoseTimestampSkewMs = Math.Abs(
                    handResult.Geometry.MediaPipeTimestampMs - poseResult.Geometry.MediaPipeTimestampMs);
            }
            else
            {
                FullContractCompatible = false;
                FirstMismatchDescription = null;
                HandPoseTimestampSkewMs = null;
            }
        }

        internal Step9DMergedInferenceGeometry Clone() =>
            new Step9DMergedInferenceGeometry(HandResult?.Clone(), PoseResult?.Clone());

        /// <summary>Returns the representative geometry (hand if available, else pose).</summary>
        public Step9DInferenceFrameGeometry Representative => Hand ?? Pose;
    }

    /// <summary>
    /// Summary of capture-wide geometry consistency checks.
    /// Created during FinalizeCapture when geometry is required.
    /// </summary>
    public sealed class Step9DCaptureGeometryConsistencySummary
    {
        public int CheckedCount { get; }
        public long FirstHandTimestampMs { get; }
        public long LastHandTimestampMs { get; }
        public long FirstPoseTimestampMs { get; }
        public long LastPoseTimestampMs { get; }
        public long MaxHandPoseAbsSkewMs { get; }

        public Step9DCaptureGeometryConsistencySummary(
            int checkedCount,
            long firstHandTimestampMs,
            long lastHandTimestampMs,
            long firstPoseTimestampMs,
            long lastPoseTimestampMs,
            long maxHandPoseAbsSkewMs)
        {
            CheckedCount = checkedCount;
            FirstHandTimestampMs = firstHandTimestampMs;
            LastHandTimestampMs = lastHandTimestampMs;
            FirstPoseTimestampMs = firstPoseTimestampMs;
            LastPoseTimestampMs = lastPoseTimestampMs;
            MaxHandPoseAbsSkewMs = maxHandPoseAbsSkewMs;
        }
    }
}
