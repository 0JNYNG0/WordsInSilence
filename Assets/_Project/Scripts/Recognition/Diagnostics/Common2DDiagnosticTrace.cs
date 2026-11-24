#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.Diagnostics
{
    /// <summary>
    /// Step 10P v1.0f2 common-2D diagnostic trace: the editor-only observation buffer for
    /// the exact numeric path from the finalized raw observation segment through
    /// AdaptCommon2D, the feature builder and the frozen model.
    ///
    /// Editor only. The whole file is compiled out of a player build, so no shipped code
    /// path can reach it.
    ///
    /// It observes and never decides. Every array it holds is a defensive copy of a value
    /// the real calculation already produced; nothing here recomputes a shadow result,
    /// and nothing it stores can flow back into the classifier, the model arrays or the
    /// capture state. When the buffer is not armed, no allocation of any kind happens.
    /// </summary>
    public enum Common2DTraceStatus
    {
        Disabled,
        Armed,
        Collecting,
        Complete,
        Committed,
        Duplicate,
        Failed,
    }

    /// <summary>Closed vocabulary. No free text ever reaches the ordinary log marker.</summary>
    public enum Common2DTraceFailureReason
    {
        None,
        NotArmed,
        SessionMismatch,
        IncompletePayload,
        SerializationFailed,
        WriteFailed,
        AlreadyCommitted,
        ExportRootUnavailable,
        CollectionFailed,
        InvalidBinding,
        InvalidPayloadShape,
        InvalidFreeText,
        TerminalNotAccepted,
        ExportRootRejected,
        MarkerTooLong,
        StagingConflict,
    }

    /// <summary>One observation convention, copied field by field.</summary>
    public sealed class Common2DConventionRecord
    {
        public string Id { get; }
        public string CoordinateSpace { get; }
        public string AxisAndDepth { get; }
        public string Handedness { get; }
        public string Mirror { get; }

        public Common2DConventionRecord(RawObservationConvention convention)
        {
            if (convention == null)
            {
                Id = string.Empty;
                CoordinateSpace = string.Empty;
                AxisAndDepth = string.Empty;
                Handedness = string.Empty;
                Mirror = string.Empty;
                return;
            }
            Id = convention.Id ?? string.Empty;
            CoordinateSpace = convention.CoordinateSpace.ToString();
            AxisAndDepth = convention.AxisAndDepth ?? string.Empty;
            Handedness = convention.Handedness ?? string.Empty;
            Mirror = convention.Mirror ?? string.Empty;
        }
    }

    /// <summary>
    /// One provider's inference frame geometry, exactly the fields
    /// TryResolveSegmentImageGeometry consults.
    /// </summary>
    public sealed class Common2DGeometryRecord
    {
        public bool Present { get; }
        public string Provenance { get; }
        public int ObservedWebcamWidth { get; }
        public int ObservedWebcamHeight { get; }
        public string ObservedVideoRotationAngleOrNull { get; }
        public string ObservedVideoVerticallyMirroredOrNull { get; }
        public int InferenceTextureWidth { get; }
        public int InferenceTextureHeight { get; }
        public int InferenceImageWidth { get; }
        public int InferenceImageHeight { get; }
        public string ImageFormat { get; }
        public string PreInferenceHorizontalFlip { get; }
        public string PreInferenceVerticalFlip { get; }
        public string PreInferenceRotationDegreesOrNull { get; }
        public string Crop { get; }
        public string Resize { get; }
        public string Letterbox { get; }
        public string PixelBufferRowOrder { get; }
        public string ProviderFrameIndexOrNull { get; }
        public long MediaPipeTimestampMs { get; }
        public string UnityFrameCountOrNull { get; }

        public Common2DGeometryRecord(Step9DInferenceFrameGeometry geometry)
        {
            if (geometry == null)
            {
                Present = false;
                Provenance = string.Empty;
                ObservedVideoRotationAngleOrNull = Absent;
                ObservedVideoVerticallyMirroredOrNull = Absent;
                ImageFormat = string.Empty;
                PreInferenceHorizontalFlip = string.Empty;
                PreInferenceVerticalFlip = string.Empty;
                PreInferenceRotationDegreesOrNull = Absent;
                Crop = string.Empty;
                Resize = string.Empty;
                Letterbox = string.Empty;
                PixelBufferRowOrder = string.Empty;
                ProviderFrameIndexOrNull = Absent;
                UnityFrameCountOrNull = Absent;
                return;
            }
            Present = true;
            Provenance = geometry.Provenance.ToString();
            ObservedWebcamWidth = geometry.ObservedWebcamWidth;
            ObservedWebcamHeight = geometry.ObservedWebcamHeight;
            ObservedVideoRotationAngleOrNull = Nullable(geometry.ObservedVideoRotationAngleOrNull);
            ObservedVideoVerticallyMirroredOrNull =
                geometry.ObservedVideoVerticallyMirroredOrNull.HasValue
                    ? (geometry.ObservedVideoVerticallyMirroredOrNull.Value ? "true" : "false")
                    : Absent;
            InferenceTextureWidth = geometry.InferenceTextureWidth;
            InferenceTextureHeight = geometry.InferenceTextureHeight;
            InferenceImageWidth = geometry.InferenceImageWidth;
            InferenceImageHeight = geometry.InferenceImageHeight;
            ImageFormat = geometry.ImageFormat ?? string.Empty;
            PreInferenceHorizontalFlip = geometry.PreInferenceHorizontalFlip.ToString();
            PreInferenceVerticalFlip = geometry.PreInferenceVerticalFlip.ToString();
            PreInferenceRotationDegreesOrNull = Nullable(geometry.PreInferenceRotationDegreesOrNull);
            Crop = geometry.Crop.ToString();
            Resize = geometry.Resize.ToString();
            Letterbox = geometry.Letterbox.ToString();
            PixelBufferRowOrder = geometry.PixelBufferRowOrder ?? string.Empty;
            ProviderFrameIndexOrNull = geometry.ProviderFrameIndexOrNull.HasValue
                ? geometry.ProviderFrameIndexOrNull.Value.ToString(CultureInfo.InvariantCulture)
                : Absent;
            MediaPipeTimestampMs = geometry.MediaPipeTimestampMs;
            UnityFrameCountOrNull = Nullable(geometry.UnityFrameCountOrNull);
        }

        public const string Absent = "absent";

        static string Nullable(int? value)
        {
            return value.HasValue
                ? value.Value.ToString(CultureInfo.InvariantCulture)
                : Absent;
        }
    }

    /// <summary>The agreement fields TryResolveSegmentImageGeometry actually evaluates.</summary>
    public sealed class Common2DGeometryAgreementRecord
    {
        public bool Present { get; }
        public bool BothAvailable { get; }
        public bool FullContractCompatible { get; }
        public string FirstMismatchDescription { get; }
        public string HandPoseTimestampSkewMs { get; }
        public string HandResultStatus { get; }
        public string PoseResultStatus { get; }
        public string HandResultDiagnosticCode { get; }
        public string PoseResultDiagnosticCode { get; }

        public Common2DGeometryAgreementRecord(Step9DMergedInferenceGeometry merged)
        {
            if (merged == null)
            {
                Present = false;
                FirstMismatchDescription = string.Empty;
                HandPoseTimestampSkewMs = Common2DGeometryRecord.Absent;
                HandResultStatus = string.Empty;
                PoseResultStatus = string.Empty;
                HandResultDiagnosticCode = string.Empty;
                PoseResultDiagnosticCode = string.Empty;
                return;
            }
            Present = true;
            BothAvailable = merged.BothAvailable;
            FullContractCompatible = merged.FullContractCompatible;
            FirstMismatchDescription = merged.FirstMismatchDescription ?? string.Empty;
            HandPoseTimestampSkewMs = merged.HandPoseTimestampSkewMs.HasValue
                ? merged.HandPoseTimestampSkewMs.Value.ToString(CultureInfo.InvariantCulture)
                : Common2DGeometryRecord.Absent;
            HandResultStatus = merged.HandResult == null
                ? string.Empty : merged.HandResult.Status.ToString();
            PoseResultStatus = merged.PoseResult == null
                ? string.Empty : merged.PoseResult.Status.ToString();
            HandResultDiagnosticCode = merged.HandResult == null
                ? string.Empty : (merged.HandResult.DiagnosticCode ?? string.Empty);
            PoseResultDiagnosticCode = merged.PoseResult == null
                ? string.Empty : (merged.PoseResult.DiagnosticCode ?? string.Empty);
        }
    }

    /// <summary>
    /// One observation of the finalized segment with all seventeen v1.0f1 fields. Every
    /// array is a fresh copy; the live observation is never referenced after construction.
    /// </summary>
    public sealed class Common2DRawObservationRecord
    {
        public const int FieldCount = 17;

        public double TimestampSeconds { get; }
        public double[] PoseCoordinates { get; }
        public double[] LeftHandCoordinates { get; }
        public double[] RightHandCoordinates { get; }
        public bool[] PoseValidMask { get; }
        public bool[] LeftHandValidMask { get; }
        public bool[] RightHandValidMask { get; }
        public bool[] PoseReappearanceMask { get; }
        public bool[] LeftHandReappearanceMask { get; }
        public bool[] RightHandReappearanceMask { get; }
        public bool[] PoseStrongDepthBoundaryMask { get; }
        public bool[] LeftHandStrongDepthBoundaryMask { get; }
        public bool[] RightHandStrongDepthBoundaryMask { get; }
        public Common2DConventionRecord Convention { get; }
        public Common2DGeometryRecord MergedGeometryHand { get; }
        public Common2DGeometryRecord MergedGeometryPose { get; }
        public Common2DGeometryAgreementRecord MergedGeometryAgreement { get; }

        public Common2DRawObservationRecord(RawMediaPipeObservation observation)
        {
            if (observation == null) throw new ArgumentNullException(nameof(observation));

            TimestampSeconds = observation.TimestampSeconds;
            PoseCoordinates = observation.Pose.CopyCoordinates();
            LeftHandCoordinates = observation.LeftHand.CopyCoordinates();
            RightHandCoordinates = observation.RightHand.CopyCoordinates();
            PoseValidMask = observation.Pose.CopyValidMask();
            LeftHandValidMask = observation.LeftHand.CopyValidMask();
            RightHandValidMask = observation.RightHand.CopyValidMask();
            PoseReappearanceMask = CopyReappearance(observation.Pose);
            LeftHandReappearanceMask = CopyReappearance(observation.LeftHand);
            RightHandReappearanceMask = CopyReappearance(observation.RightHand);
            PoseStrongDepthBoundaryMask = CopyStrongDepthBoundary(observation.Pose);
            LeftHandStrongDepthBoundaryMask = CopyStrongDepthBoundary(observation.LeftHand);
            RightHandStrongDepthBoundaryMask = CopyStrongDepthBoundary(observation.RightHand);
            Convention = new Common2DConventionRecord(observation.Convention);
            Step9DMergedInferenceGeometry merged = observation.MergedGeometry;
            MergedGeometryHand = new Common2DGeometryRecord(merged == null ? null : merged.Hand);
            MergedGeometryPose = new Common2DGeometryRecord(merged == null ? null : merged.Pose);
            MergedGeometryAgreement = new Common2DGeometryAgreementRecord(merged);
        }

        /// <summary>Every one of the seventeen fields has its exact declared shape.</summary>
        public bool HasExactShapes()
        {
            const int Pose = RawMediaPipeObservation.PoseLandmarkCount;
            const int Hand = RawMediaPipeObservation.HandLandmarkCount;
            return !double.IsNaN(TimestampSeconds) &&
                   PoseCoordinates != null && PoseCoordinates.Length == Pose * 3 &&
                   LeftHandCoordinates != null && LeftHandCoordinates.Length == Hand * 3 &&
                   RightHandCoordinates != null && RightHandCoordinates.Length == Hand * 3 &&
                   PoseValidMask != null && PoseValidMask.Length == Pose &&
                   LeftHandValidMask != null && LeftHandValidMask.Length == Hand &&
                   RightHandValidMask != null && RightHandValidMask.Length == Hand &&
                   PoseReappearanceMask != null && PoseReappearanceMask.Length == Pose &&
                   LeftHandReappearanceMask != null && LeftHandReappearanceMask.Length == Hand &&
                   RightHandReappearanceMask != null && RightHandReappearanceMask.Length == Hand &&
                   PoseStrongDepthBoundaryMask != null && PoseStrongDepthBoundaryMask.Length == Pose &&
                   LeftHandStrongDepthBoundaryMask != null && LeftHandStrongDepthBoundaryMask.Length == Hand &&
                   RightHandStrongDepthBoundaryMask != null && RightHandStrongDepthBoundaryMask.Length == Hand &&
                   Convention != null && MergedGeometryHand != null &&
                   MergedGeometryPose != null && MergedGeometryAgreement != null;
        }

        static bool[] CopyReappearance(RawLandmarkObservationSet set)
        {
            var values = new bool[set.LandmarkCount];
            for (int i = 0; i < values.Length; i++) values[i] = set.IsReappearance(i);
            return values;
        }

        static bool[] CopyStrongDepthBoundary(RawLandmarkObservationSet set)
        {
            var values = new bool[set.LandmarkCount];
            for (int i = 0; i < values.Length; i++) values[i] = set.IsStrongDepthBoundary(i);
            return values;
        }
    }

    /// <summary>
    /// One capture session's complete trace. Owned entirely by the buffer; the exporter
    /// reads it and writes nothing back.
    /// </summary>
    public sealed class Common2DDiagnosticTrace
    {
        public string CaptureSessionId { get; internal set; }
        public string RequestedTarget { get; internal set; }
        public string ArtifactVersion { get; internal set; }
        public string ArtifactSha256 { get; internal set; }
        public int SourceFrameCount { get; internal set; }
        public string DomainContract { get; internal set; }

        public List<Common2DRawObservationRecord> RawObservations { get; }
            = new List<Common2DRawObservationRecord>();

        public int NormalizedFrameCount { get; internal set; }
        public double[] NormalizedCoordinates { get; internal set; }      // [T,16,3]
        public bool[] InputMask { get; internal set; }                    // [T,16]
        public double[] ResampledCoordinates { get; internal set; }       // [32,16,2]
        public bool[] ResampledMask { get; internal set; }                // [32,16]
        public double[] FeatureVector { get; internal set; }              // [1536]
        public double[] StandardizedFeatureVector { get; internal set; }  // [1536]
        public double[] PcaProjection { get; internal set; }              // [4]
        public double[] LrComponentContributions { get; internal set; }   // [4]
        public double LrIntercept { get; internal set; }
        public double DecisionValue { get; internal set; }
        public double GreetingProbability { get; internal set; }
        public double ThankYouProbability { get; internal set; }

        public bool ModelPathRecorded { get; internal set; }
        public bool AttemptComplete { get; internal set; }

        /// <summary>
        /// Set only after the developer coordinator has accepted and committed the
        /// terminal take. Nothing may be exported before it.
        /// </summary>
        public bool TerminalAccepted { get; internal set; }

        /// <summary>The owner that armed the buffer for this capture.</summary>
        public object Owner { get; internal set; }

        /// <summary>
        /// The target the seal was taken under. It must still equal the session target
        /// at publish time, so a completed trace cannot be relabelled before its first
        /// export.
        /// </summary>
        public string SealTarget { get; internal set; }

        /// <summary>
        /// Full semantic validation. Every binding, every declared shape and count, the
        /// raw-to-normalized frame agreement and the accepted terminal seal are checked
        /// here, so a payload that is merely non-null cannot be published.
        /// </summary>
        public bool Validate(out Common2DTraceFailureReason reason)
        {
            reason = Common2DTraceFailureReason.None;

            if (!Common2DClosedVocabulary.IsApprovedSessionId(CaptureSessionId) ||
                !Common2DClosedVocabulary.IsApprovedTarget(RequestedTarget) ||
                !Common2DClosedVocabulary.IsApprovedArtifactVersion(ArtifactVersion) ||
                !Common2DClosedVocabulary.IsExactFrozenArtifactDigest(ArtifactSha256) ||
                !Common2DClosedVocabulary.IsApprovedDomainContract(DomainContract))
            {
                reason = Common2DTraceFailureReason.InvalidBinding;
                return false;
            }
            // The seal target must still be the session target: a completed trace may
            // not be relabelled to another target before its first export.
            if (!string.Equals(SealTarget, RequestedTarget, StringComparison.Ordinal))
            {
                reason = Common2DTraceFailureReason.InvalidBinding;
                return false;
            }
            if (!AttemptComplete)
            {
                reason = Common2DTraceFailureReason.IncompletePayload;
                return false;
            }
            if (!TerminalAccepted)
            {
                reason = Common2DTraceFailureReason.TerminalNotAccepted;
                return false;
            }
            if (!ModelPathRecorded)
            {
                reason = Common2DTraceFailureReason.IncompletePayload;
                return false;
            }
            // The raw sequence, the declared source frame count and the normalized
            // frame count must all be the same number of frames.
            if (RawObservations.Count <= 0 ||
                SourceFrameCount != RawObservations.Count ||
                NormalizedFrameCount != RawObservations.Count)
            {
                reason = Common2DTraceFailureReason.InvalidPayloadShape;
                return false;
            }
            for (int i = 0; i < RawObservations.Count; i++)
            {
                if (!RawObservations[i].HasExactShapes())
                {
                    reason = Common2DTraceFailureReason.InvalidPayloadShape;
                    return false;
                }
                if (!Common2DClosedVocabulary.IsApprovedConvention(
                        RawObservations[i].Convention) ||
                    !Common2DClosedVocabulary.IsApprovedGeometry(
                        RawObservations[i].MergedGeometryHand) ||
                    !Common2DClosedVocabulary.IsApprovedGeometry(
                        RawObservations[i].MergedGeometryPose) ||
                    !Common2DClosedVocabulary.IsApprovedAgreement(
                        RawObservations[i].MergedGeometryAgreement))
                {
                    reason = Common2DTraceFailureReason.InvalidFreeText;
                    return false;
                }
            }

            if (NormalizedFrameCount <= 0 ||
                !Exact(NormalizedCoordinates, NormalizedFrameCount * Common2DLandmarks * 3) ||
                !Exact(InputMask, NormalizedFrameCount * Common2DLandmarks) ||
                !Exact(ResampledCoordinates, ResampleT * Common2DLandmarks * 2) ||
                !Exact(ResampledMask, ResampleT * Common2DLandmarks) ||
                !Exact(FeatureVector, FeatureLength) ||
                !Exact(StandardizedFeatureVector, FeatureLength) ||
                !Exact(PcaProjection, PcaComponents) ||
                !Exact(LrComponentContributions, PcaComponents))
            {
                reason = Common2DTraceFailureReason.InvalidPayloadShape;
                return false;
            }
            return true;
        }

        /// <summary>
        /// Editor-only negative-case seam. Returns a copy of this trace with exactly the
        /// requested fields replaced so a validation test can prove the exporter and the
        /// marker refuse an invalid binding, a relabelled target or a wrong-length array.
        ///
        /// Every semantic field Validate reads is carried over, including SealTarget: a
        /// clone with no override must validate exactly as its source does, so one
        /// silently dropped field can never be the reason an unrelated negative fails.
        /// The original trace is untouched, the setters stay internal, and nothing here
        /// is reachable from production code.
        /// </summary>
        public Common2DDiagnosticTrace CloneWithOverride(
            string captureSessionId = null,
            string requestedTarget = null,
            double[] featureVector = null,
            double[] pcaProjection = null,
            string sealTarget = null,
            string artifactSha256 = null,
            int? sourceFrameCount = null,
            int? normalizedFrameCount = null)
        {
            var clone = new Common2DDiagnosticTrace
            {
                CaptureSessionId = captureSessionId ?? CaptureSessionId,
                RequestedTarget = requestedTarget ?? RequestedTarget,
                SealTarget = sealTarget ?? SealTarget,
                ArtifactVersion = ArtifactVersion,
                ArtifactSha256 = artifactSha256 ?? ArtifactSha256,
                SourceFrameCount = sourceFrameCount ?? SourceFrameCount,
                DomainContract = DomainContract,
                NormalizedFrameCount = normalizedFrameCount ?? NormalizedFrameCount,
                NormalizedCoordinates = NormalizedCoordinates,
                InputMask = InputMask,
                ResampledCoordinates = ResampledCoordinates,
                ResampledMask = ResampledMask,
                FeatureVector = featureVector ?? FeatureVector,
                StandardizedFeatureVector = StandardizedFeatureVector,
                PcaProjection = pcaProjection ?? PcaProjection,
                LrComponentContributions = LrComponentContributions,
                LrIntercept = LrIntercept,
                DecisionValue = DecisionValue,
                GreetingProbability = GreetingProbability,
                ThankYouProbability = ThankYouProbability,
                ModelPathRecorded = ModelPathRecorded,
                AttemptComplete = AttemptComplete,
                TerminalAccepted = TerminalAccepted,
                Owner = Owner,
            };
            clone.RawObservations.AddRange(RawObservations);
            return clone;
        }

        public const int Common2DLandmarks = 16;
        public const int ResampleT = 32;
        public const int FeatureLength = 1536;
        public const int PcaComponents = 4;

        static bool Exact(double[] values, int length)
        {
            return values != null && values.Length == length;
        }

        static bool Exact(bool[] values, int length)
        {
            return values != null && values.Length == length;
        }
    }

    /// <summary>
    /// The closed vocabularies every binding and every payload string must satisfy.
    /// A value outside them is rejected as a typed failure; it is never sanitized by
    /// character substitution and never truncated into something that looks valid.
    /// </summary>
    public static class Common2DClosedVocabulary
    {
        public const string ArtifactVersion = "common_2d_v1_6";
        public const string DomainContract =
            "raw_post_gap_pixel_plane_z0_torso_normalize_2d";
        public const int MaxSessionIdLength = 64;

        public static readonly string[] ApprovedTargets = { "GREETING", "THANK_YOU" };

        /// <summary>
        /// The two approved observation conventions, as whole tuples. These are fixed
        /// source constants of RawObservationConvention, not operator text, so a payload
        /// convention is accepted only when all five fields match one tuple exactly.
        /// </summary>
        public static readonly string[][] ApprovedConventions =
        {
            new[]
            {
                "approved_step7b_front_camera_metric_3d",
                "ApprovedFrontCameraMetric3D",
                "FRONT_camera_meters; x/y/z preserved",
                "approved Step 7B left_hand/right_hand arrays",
                "not mirrored",
            },
            new[]
            {
                "mediapipe_tasks_webcam_normalized_image_v0_16_3",
                "MediaPipeNormalizedImage",
                "MediaPipe NormalizedLandmark x-right/y-down/z-source-relative; " +
                "no axis rewrite",
                "MediaPipe handedness categoryName mapped directly",
                "raw WebCamTexture pixels; RawImage UV mirror is display-only",
            },
        };

        public static readonly string[] ApprovedTransformStates =
        {
            "Unknown", "None", "Applied",
        };

        public static readonly string[] ApprovedProvenance =
        {
            "ObservedAtHandInference", "ObservedAtPoseInference", "TestInjected",
        };

        public static readonly string[] ApprovedAcquisitionStatus =
        {
            "", "Available", "UnavailableBeforeFrameReady",
            "UnavailableInvalidDimensions", "UnavailableCaptureException",
        };

        public static readonly string[] ApprovedImageFormats =
        {
            "SRGBA", "SRGB", "GRAY8", "VEC32F1", "VEC32F2", "UNKNOWN",
        };

        public static readonly string[] ApprovedRowOrders =
        {
            "TopDown", "BottomUp", "Unknown",
        };

        /// <summary>Session ids are an approved format, not arbitrary free text.</summary>
        public static bool IsApprovedSessionId(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length > MaxSessionIdLength)
                return false;
            for (int i = 0; i < value.Length; i++)
            {
                char character = value[i];
                bool safe = (character >= 'a' && character <= 'z') ||
                            (character >= 'A' && character <= 'Z') ||
                            (character >= '0' && character <= '9') ||
                            character == '-' || character == '_';
                if (!safe) return false;
            }
            return true;
        }

        public static bool IsApprovedTarget(string value)
        {
            return Contains(ApprovedTargets, value);
        }

        public static bool IsApprovedArtifactVersion(string value)
        {
            return string.Equals(value, ArtifactVersion, StringComparison.Ordinal);
        }

        public static bool IsApprovedDomainContract(string value)
        {
            return string.Equals(value, DomainContract, StringComparison.Ordinal);
        }

        /// <summary>
        /// The artifact digest must be the exact frozen combined digest, not merely a
        /// well-formed 64-hex string.
        /// </summary>
        public static bool IsExactFrozenArtifactDigest(string value)
        {
            return IsLowerHex64(value) && string.Equals(
                value, WordsInSilence.Recognition.Common.FrozenCommonSignModel
                    .ExpectedCombinedSha256, StringComparison.Ordinal);
        }

        public static bool IsLowerHex64(string value)
        {
            if (value == null || value.Length != 64) return false;
            for (int i = 0; i < 64; i++)
            {
                char character = value[i];
                bool hex = (character >= '0' && character <= '9') ||
                           (character >= 'a' && character <= 'f');
                if (!hex) return false;
            }
            return true;
        }

        /// <summary>
        /// Accepts a convention only when every field equals one approved tuple. A
        /// single altered character anywhere in the record is rejected, so no operator
        /// or provider text can be smuggled through these fields.
        /// </summary>
        public static bool IsApprovedConvention(Common2DConventionRecord convention)
        {
            if (convention == null) return false;
            for (int i = 0; i < ApprovedConventions.Length; i++)
            {
                string[] approved = ApprovedConventions[i];
                if (string.Equals(convention.Id, approved[0], StringComparison.Ordinal) &&
                    string.Equals(convention.CoordinateSpace, approved[1], StringComparison.Ordinal) &&
                    string.Equals(convention.AxisAndDepth, approved[2], StringComparison.Ordinal) &&
                    string.Equals(convention.Handedness, approved[3], StringComparison.Ordinal) &&
                    string.Equals(convention.Mirror, approved[4], StringComparison.Ordinal))
                    return true;
            }
            return false;
        }

        public static bool IsApprovedGeometry(Common2DGeometryRecord geometry)
        {
            if (geometry == null) return false;
            if (!geometry.Present)
            {
                // An absent provider geometry is rejected upstream by
                // TryResolveSegmentImageGeometry; it may never reach a payload.
                return false;
            }
            return Contains(ApprovedProvenance, geometry.Provenance) &&
                   Contains(ApprovedImageFormats, geometry.ImageFormat) &&
                   Contains(ApprovedRowOrders, geometry.PixelBufferRowOrder) &&
                   Contains(ApprovedTransformStates, geometry.PreInferenceHorizontalFlip) &&
                   Contains(ApprovedTransformStates, geometry.PreInferenceVerticalFlip) &&
                   Contains(ApprovedTransformStates, geometry.Crop) &&
                   Contains(ApprovedTransformStates, geometry.Resize) &&
                   Contains(ApprovedTransformStates, geometry.Letterbox) &&
                   IsIntegerOrAbsent(geometry.ObservedVideoRotationAngleOrNull) &&
                   IsBooleanOrAbsent(geometry.ObservedVideoVerticallyMirroredOrNull) &&
                   IsIntegerOrAbsent(geometry.PreInferenceRotationDegreesOrNull) &&
                   IsIntegerOrAbsent(geometry.ProviderFrameIndexOrNull) &&
                   IsIntegerOrAbsent(geometry.UnityFrameCountOrNull);
        }

        public static bool IsApprovedAgreement(Common2DGeometryAgreementRecord agreement)
        {
            if (agreement == null || !agreement.Present) return false;
            // FirstMismatchDescription and the two diagnostic codes are provider free
            // text. Only the empty value is publishable; anything else is a typed
            // failure and is never echoed into the payload or the marker.
            return string.IsNullOrEmpty(agreement.FirstMismatchDescription) &&
                   string.IsNullOrEmpty(agreement.HandResultDiagnosticCode) &&
                   string.IsNullOrEmpty(agreement.PoseResultDiagnosticCode) &&
                   Contains(ApprovedAcquisitionStatus, agreement.HandResultStatus) &&
                   Contains(ApprovedAcquisitionStatus, agreement.PoseResultStatus) &&
                   IsIntegerOrAbsent(agreement.HandPoseTimestampSkewMs);
        }

        /// <summary>A bounded enum-shaped token: letters, digits and underscore only.</summary>
        public static bool IsClosedToken(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length > 64) return false;
            for (int i = 0; i < value.Length; i++)
            {
                char character = value[i];
                bool safe = (character >= 'a' && character <= 'z') ||
                            (character >= 'A' && character <= 'Z') ||
                            (character >= '0' && character <= '9') ||
                            character == '_';
                if (!safe) return false;
            }
            return true;
        }

        public static bool IsIntegerOrAbsent(string value)
        {
            if (string.Equals(value, Common2DGeometryRecord.Absent, StringComparison.Ordinal))
                return true;
            if (string.IsNullOrEmpty(value) || value.Length > 20) return false;
            for (int i = 0; i < value.Length; i++)
            {
                if (i == 0 && value[i] == '-' && value.Length > 1) continue;
                if (value[i] < '0' || value[i] > '9') return false;
            }
            return true;
        }

        public static bool IsBooleanOrAbsent(string value)
        {
            return string.Equals(value, "true", StringComparison.Ordinal) ||
                   string.Equals(value, "false", StringComparison.Ordinal) ||
                   string.Equals(value, Common2DGeometryRecord.Absent,
                                 StringComparison.Ordinal);
        }

        static bool Contains(string[] approved, string value)
        {
            if (value == null) return false;
            for (int i = 0; i < approved.Length; i++)
                if (string.Equals(approved[i], value, StringComparison.Ordinal))
                    return true;
            return false;
        }
    }

    /// <summary>
    /// The editor-only collection point.
    ///
    /// Disabled by default and owned: tracing is active only while a live developer
    /// harness instance holds the arm. <see cref="ShouldObserve"/> is the guard every
    /// ordinary calculation call site evaluates FIRST, before any argument is copied, so
    /// a disabled or unowned buffer costs no diagnostic allocation at all. Releasing the
    /// owner - on disable, destroy or cancel - drops any uncommitted trace, so ordinary
    /// Practice can never ride on a stale armed flag, and re-activation requires the
    /// explicit opt-in again.
    ///
    /// Every recording entry point is a Safe* wrapper that converts a diagnostic-only
    /// failure into a closed typed status. A diagnostic fault therefore cannot reach the
    /// capture path, the classification or the terminal commit. Real calculation errors
    /// are never caught here: these wrappers only surround copying and bookkeeping.
    /// </summary>
    public static class Common2DDiagnosticTraceBuffer
    {
        static object owner;
        static Common2DDiagnosticTrace active;
        static readonly HashSet<string> committedSessions = new HashSet<string>();

        // Once a stage of one capture has failed, the capture stays failed. A later
        // guarded stage may not build a fresh partial trace that hides the first cause.
        static bool captureFailed;
        static Common2DTraceFailureReason firstFailureReason =
            Common2DTraceFailureReason.None;
        static Common2DCollectionFaultInjection firstFailureStage =
            Common2DCollectionFaultInjection.None;

        /// <summary>Editor-only observation counters, used by the focused tests and the
        /// offline gate to show that a disabled buffer performs no diagnostic work.</summary>
        public static int DiagnosticAllocationCount { get; private set; }
        public static int DiagnosticArrayCopyCount { get; private set; }
        public static int GuardRejectionCount { get; private set; }

        public static Common2DTraceFailureReason LastCollectionFailure { get; private set; }
        public static int CollectionFailureCount { get; private set; }

        /// <summary>The first closed reason recorded for the current capture.</summary>
        public static Common2DTraceFailureReason FirstCollectionFailure
        {
            get { return firstFailureReason; }
        }

        /// <summary>The stage that first failed, as a closed enum value.</summary>
        public static Common2DCollectionFaultInjection FirstCollectionFailureStage
        {
            get { return firstFailureStage; }
        }

        /// <summary>True while the current capture is latched failed.</summary>
        public static bool CaptureFailed { get { return captureFailed; } }

        /// <summary>Test-only fault seam for collection. Default None.</summary>
        public static Common2DCollectionFaultInjection CollectionFaultInjection
        {
            get; set;
        }

        public static bool IsArmed { get { return owner != null; } }
        public static object Owner { get { return owner; } }
        public static Common2DDiagnosticTrace Active { get { return active; } }

        public static Common2DTraceStatus Status
        {
            get
            {
                if (owner == null) return Common2DTraceStatus.Disabled;
                if (active == null) return Common2DTraceStatus.Armed;
                if (active.TerminalAccepted) return Common2DTraceStatus.Complete;
                if (active.AttemptComplete) return Common2DTraceStatus.Complete;
                return Common2DTraceStatus.Collecting;
            }
        }

        /// <summary>
        /// The guard every ordinary call site evaluates before it builds any argument.
        /// It allocates nothing and it is false unless a live owner armed the buffer.
        /// </summary>
        public static bool ShouldObserve()
        {
            if (owner != null) return true;
            GuardRejectionCount++;
            return false;
        }

        /// <summary>
        /// Explicit opt-in. Only the Editor developer harness calls this. It never
        /// silently replaces a different live owner: a second owner is refused with a
        /// closed result and the first owner and its trace are left exactly as they are.
        /// </summary>
        public static Common2DArmResult TryArm(object requestingOwner)
        {
            if (requestingOwner == null) return Common2DArmResult.RejectedNullOwner;
            if (owner != null && !ReferenceEquals(owner, requestingOwner))
                return Common2DArmResult.RejectedOtherOwnerLive;
            if (ReferenceEquals(owner, requestingOwner))
                return Common2DArmResult.AlreadyOwned;
            owner = requestingOwner;
            active = null;
            ClearFailureLatch();
            return Common2DArmResult.Armed;
        }

        /// <summary>Backward-compatible arm that ignores its closed result.</summary>
        public static void Arm(object requestingOwner)
        {
            TryArm(requestingOwner);
        }

        static void ClearFailureLatch()
        {
            captureFailed = false;
            firstFailureReason = Common2DTraceFailureReason.None;
            firstFailureStage = Common2DCollectionFaultInjection.None;
        }

        /// <summary>
        /// Cancels the incomplete trace of the current capture without touching
        /// ownership. The live controller calls this from cancel and reset.
        /// </summary>
        public static void CancelIncompleteTrace()
        {
            if (active != null && !active.TerminalAccepted) active = null;
            ClearFailureLatch();
        }

        /// <summary>
        /// Releases the arm held by this owner and drops any uncommitted trace. Called
        /// from the harness on disable, destroy and cancel.
        /// </summary>
        public static bool ReleaseOwner(object releasingOwner)
        {
            if (releasingOwner == null || !ReferenceEquals(owner, releasingOwner))
                return false;
            owner = null;
            active = null;
            ClearFailureLatch();
            return true;
        }

        /// <summary>Unconditional disarm; used by lifecycle teardown and tests.</summary>
        public static void Disarm()
        {
            owner = null;
            active = null;
            ClearFailureLatch();
        }

        /// <summary>Clears before any new observation on initialize, begin and reset.</summary>
        public static void Clear()
        {
            active = null;
            ClearFailureLatch();
        }

        /// <summary>Forgets ownership, trace, counters and commit history.</summary>
        public static void ResetAll()
        {
            owner = null;
            active = null;
            ClearFailureLatch();
            LastCollectionFailure = Common2DTraceFailureReason.None;
            committedSessions.Clear();
            DiagnosticAllocationCount = 0;
            DiagnosticArrayCopyCount = 0;
            GuardRejectionCount = 0;
            CollectionFailureCount = 0;
            LastCollectionFailure = Common2DTraceFailureReason.None;
            CollectionFaultInjection = Common2DCollectionFaultInjection.None;
        }

        public static void ResetCounters()
        {
            DiagnosticAllocationCount = 0;
            DiagnosticArrayCopyCount = 0;
            GuardRejectionCount = 0;
        }

        public static bool HasCommitted(string captureSessionId)
        {
            return captureSessionId != null && committedSessions.Contains(captureSessionId);
        }

        internal static void MarkCommitted(string captureSessionId)
        {
            if (!string.IsNullOrEmpty(captureSessionId))
                committedSessions.Add(captureSessionId);
        }

        static void RecordCollectionFailure(
            Common2DTraceFailureReason reason, Common2DCollectionFaultInjection stage)
        {
            CollectionFailureCount++;
            LastCollectionFailure = reason;
            if (!captureFailed)
            {
                // Latch the first cause. Later guarded stages of the same capture are
                // refused, so nothing can overwrite or hide why this capture failed.
                captureFailed = true;
                firstFailureReason = reason;
                firstFailureStage = stage;
            }
            // The trace is abandoned rather than published half-built. The capture,
            // the classification and the terminal commit continue untouched.
            active = null;
        }

        /// <summary>
        /// Records a diagnostic failure raised outside this class, such as the
        /// classifier's sink allocation. It is a closed typed status: no exception text
        /// is captured and nothing about the capture or the classification changes.
        /// </summary>
        public static void RecordExternalCollectionFailure(
            Common2DCollectionFaultInjection stage)
        {
            if (owner == null) return;
            RecordCollectionFailure(Common2DTraceFailureReason.CollectionFailed, stage);
        }

        /// <summary>
        /// The classifier calls this before allocating its diagnostic sinks so a sink
        /// allocation fault is injectable at the real allocation site.
        /// </summary>
        public static void ThrowIfSinkAllocationFaultInjected()
        {
            ThrowIfInjected(Common2DCollectionFaultInjection.SinkAllocationThrows);
        }

        static void ThrowIfInjected(Common2DCollectionFaultInjection stage)
        {
            if (CollectionFaultInjection == stage)
                throw new InvalidOperationException("injected diagnostic collection fault");
        }

        static Common2DDiagnosticTrace Require()
        {
            if (captureFailed)
            {
                // Refuse to rebuild. The first cause stays visible.
                return null;
            }
            if (active == null)
            {
                active = new Common2DDiagnosticTrace();
                active.Owner = owner;
                DiagnosticAllocationCount++;
            }
            return active;
        }

        /// <summary>
        /// Binds the trace to the capture identity. Called by the live controller at
        /// finalize, before the common-2D build, so a later capture can never inherit it.
        /// </summary>
        public static bool SafeBeginSession(
            string captureSessionId, string requestedTarget, int sourceFrameCount)
        {
            if (!ShouldObserve()) return false;
            try
            {
                ThrowIfInjected(Common2DCollectionFaultInjection.BeginSessionThrows);
                var trace = new Common2DDiagnosticTrace();
                DiagnosticAllocationCount++;
                trace.Owner = owner;
                trace.CaptureSessionId = captureSessionId ?? string.Empty;
                trace.RequestedTarget = requestedTarget ?? string.Empty;
                trace.SourceFrameCount = sourceFrameCount;
                active = trace;
                return true;
            }
            catch (Exception)
            {
                RecordCollectionFailure(Common2DTraceFailureReason.CollectionFailed,
                                        Common2DCollectionFaultInjection.BeginSessionThrows);
                return false;
            }
        }

        /// <summary>
        /// Records the exact finalized observation sequence AdaptCommon2D was handed,
        /// after the authorized trailing trim and before any normalization.
        /// </summary>
        public static bool SafeRecordRawObservations(
            IReadOnlyList<RawMediaPipeObservation> observations)
        {
            if (!ShouldObserve() || observations == null) return false;
            try
            {
                ThrowIfInjected(Common2DCollectionFaultInjection.RawObservationsThrow);
                Common2DDiagnosticTrace trace = Require();
                if (trace == null) return false;
                trace.RawObservations.Clear();
                for (int i = 0; i < observations.Count; i++)
                {
                    RawMediaPipeObservation observation = observations[i];
                    if (observation == null) continue;
                    trace.RawObservations.Add(new Common2DRawObservationRecord(observation));
                    DiagnosticAllocationCount++;
                    DiagnosticArrayCopyCount += 12;
                }
                if (trace.SourceFrameCount == 0)
                    trace.SourceFrameCount = observations.Count;
                return true;
            }
            catch (Exception)
            {
                RecordCollectionFailure(Common2DTraceFailureReason.CollectionFailed,
                                        Common2DCollectionFaultInjection.RawObservationsThrow);
                return false;
            }
        }

        /// <summary>
        /// Records the normalized common-2D capture. The two defensive copies are taken
        /// INSIDE this method, after the guard, so a disabled buffer never causes them.
        /// </summary>
        public static bool SafeRecordNormalizedCapture(CommonTwoDimensionalCapture capture)
        {
            if (!ShouldObserve() || capture == null) return false;
            try
            {
                ThrowIfInjected(Common2DCollectionFaultInjection.NormalizedCaptureThrows);
                Common2DDiagnosticTrace trace = Require();
                if (trace == null) return false;
                trace.NormalizedFrameCount = capture.FrameCount;
                trace.NormalizedCoordinates = capture.CopyCoordinates();
                trace.InputMask = capture.CopyValidMask();
                DiagnosticArrayCopyCount += 2;
                trace.DomainContract = capture.DomainContract ?? string.Empty;
                return true;
            }
            catch (Exception)
            {
                RecordCollectionFailure(Common2DTraceFailureReason.CollectionFailed,
                                        Common2DCollectionFaultInjection.NormalizedCaptureThrows);
                return false;
            }
        }

        /// <summary>Records the resampled sequence and mask the feature builder produced.</summary>
        public static bool SafeRecordFeaturePath(
            double[] resampled, bool[] resampledMask, double[] features)
        {
            if (!ShouldObserve()) return false;
            try
            {
                ThrowIfInjected(Common2DCollectionFaultInjection.FeaturePathThrows);
                Common2DDiagnosticTrace trace = Require();
                if (trace == null) return false;
                trace.ResampledCoordinates = resampled;
                trace.ResampledMask = resampledMask;
                trace.FeatureVector = features;
                DiagnosticArrayCopyCount += 3;
                return true;
            }
            catch (Exception)
            {
                RecordCollectionFailure(Common2DTraceFailureReason.CollectionFailed,
                                        Common2DCollectionFaultInjection.FeaturePathThrows);
                return false;
            }
        }

        /// <summary>
        /// Records the frozen model decomposition. Called only after the classification
        /// object has already been constructed by the real path.
        /// </summary>
        public static bool SafeRecordModelPath(
            string artifactVersion, string artifactSha256,
            double[] standardized, double[] projection, double[] contributions,
            double intercept, double decision,
            double greetingProbability, double thankYouProbability)
        {
            if (!ShouldObserve()) return false;
            try
            {
                ThrowIfInjected(Common2DCollectionFaultInjection.ModelPathThrows);
                Common2DDiagnosticTrace trace = Require();
                if (trace == null) return false;
                trace.ArtifactVersion = artifactVersion ?? string.Empty;
                trace.ArtifactSha256 = artifactSha256 ?? string.Empty;
                trace.StandardizedFeatureVector = standardized;
                trace.PcaProjection = projection;
                trace.LrComponentContributions = contributions;
                DiagnosticArrayCopyCount += 3;
                trace.LrIntercept = intercept;
                trace.DecisionValue = decision;
                trace.GreetingProbability = greetingProbability;
                trace.ThankYouProbability = thankYouProbability;
                trace.ModelPathRecorded = true;
                return true;
            }
            catch (Exception)
            {
                RecordCollectionFailure(Common2DTraceFailureReason.CollectionFailed,
                                        Common2DCollectionFaultInjection.ModelPathThrows);
                return false;
            }
        }

        /// <summary>
        /// Seals the trace once the attempt result is fully constructed. Nothing is
        /// exported here; the Editor harness asks for that after coordinator finalize.
        /// </summary>
        public static bool SafeMarkAttemptComplete(string requestedTarget)
        {
            if (!ShouldObserve() || active == null) return false;
            try
            {
                ThrowIfInjected(Common2DCollectionFaultInjection.SealThrows);
                if (string.IsNullOrEmpty(active.RequestedTarget))
                    active.RequestedTarget = requestedTarget ?? string.Empty;
                // The seal target is recorded separately so a completed trace cannot be
                // relabelled to another target before its first export.
                active.SealTarget = requestedTarget ?? string.Empty;
                active.AttemptComplete = true;
                return true;
            }
            catch (Exception)
            {
                RecordCollectionFailure(Common2DTraceFailureReason.CollectionFailed,
                                        Common2DCollectionFaultInjection.SealThrows);
                return false;
            }
        }

        /// <summary>
        /// Marks the coordinator's terminal take accepted. Only the owning developer
        /// harness may call it, and only after coordinator.Finalize has returned, so a
        /// trace can never be exported before the accepted terminal state is committed.
        /// </summary>
        public static bool MarkTerminalAccepted(object requestingOwner)
        {
            if (requestingOwner == null || !ReferenceEquals(owner, requestingOwner))
                return false;
            if (active == null || !active.AttemptComplete) return false;
            if (!ReferenceEquals(active.Owner, requestingOwner)) return false;
            active.TerminalAccepted = true;
            return true;
        }
    }

    /// <summary>
    /// The closed result of an ownership acquisition attempt. A second live owner is
    /// refused rather than silently replacing the first.
    /// </summary>
    public enum Common2DArmResult
    {
        Armed,
        AlreadyOwned,
        RejectedNullOwner,
        RejectedOtherOwnerLive,
    }

    /// <summary>
    /// Test-only collection fault stages, one per guarded diagnostic entry point plus
    /// the classifier's sink allocation. Default None.
    /// </summary>
    public enum Common2DCollectionFaultInjection
    {
        None,
        SinkAllocationThrows,
        BeginSessionThrows,
        RawObservationsThrow,
        NormalizedCaptureThrows,
        FeaturePathThrows,
        ModelPathThrows,
        SealThrows,
    }
    /// <summary>
    /// The ordinary log marker.
    ///
    /// Deterministic field order, closed vocabulary, and nothing outside the seventeen
    /// whitelisted fields. Values are validated against approved formats rather than
    /// sanitized: a session id, target, artifact version or digest that does not match
    /// its approved form is refused as a typed failure, so no filename, username or free
    /// text can survive as a "cleaned" token.
    ///
    /// A successful marker carries every array shape and every full 64-character SHA-256
    /// in a fixed member order, and exactly four PCA values and four LR contributions.
    /// Nothing is truncated: if a composed marker were to exceed the pre-fixed bound it
    /// is reported as MarkerTooLong instead of being emitted as a shortened success.
    /// </summary>
    public static class Common2DDiagnosticLogMarker
    {
        public const string Prefix = "WIS_COMMON2D_TRACE_V1";

        public static readonly string[] AllowedFields =
        {
            "captureSessionId",
            "requestedTarget",
            "artifactVersion",
            "artifactSha256",
            "sourceFrameCount",
            "exportId",
            "payloadSha256",
            "arrayShapes",
            "arraySha256s",
            "pca4",
            "lrContributions4",
            "lrIntercept",
            "decisionValue",
            "greetingProbability",
            "thankYouProbability",
            "status",
            "typedFailureReason",
        };

        /// <summary>
        /// Pre-fixed to accommodate a complete marker for the full member set: the
        /// twenty-four member shapes and their twenty-four full digests, plus the eight
        /// scalars and the bindings. It is a validation bound, never a truncation point.
        /// </summary>
        public const int MaxLength = 8192;

        public const int RequiredComponentCount = 4;
        public const string Unavailable = "none";

        /// <summary>
        /// Builds the marker for a successful export. Returns false with a typed reason
        /// if any value is outside its approved form, if the component arrays are not
        /// exactly four long, or if the composed line would exceed the fixed bound.
        /// </summary>
        public static bool TryBuildSuccess(
            Common2DDiagnosticTrace trace,
            string exportId,
            string payloadSha256,
            string arrayShapes,
            string arraySha256s,
            out string marker,
            out Common2DTraceFailureReason reason)
        {
            marker = string.Empty;
            reason = Common2DTraceFailureReason.None;
            if (trace == null)
            {
                reason = Common2DTraceFailureReason.NotArmed;
                return false;
            }
            if (!Common2DClosedVocabulary.IsApprovedSessionId(trace.CaptureSessionId) ||
                !Common2DClosedVocabulary.IsApprovedTarget(trace.RequestedTarget) ||
                !Common2DClosedVocabulary.IsApprovedArtifactVersion(trace.ArtifactVersion) ||
                !Common2DClosedVocabulary.IsLowerHex64(trace.ArtifactSha256))
            {
                reason = Common2DTraceFailureReason.InvalidBinding;
                return false;
            }
            if (!Common2DClosedVocabulary.IsLowerHex64(exportId) ||
                !Common2DClosedVocabulary.IsLowerHex64(payloadSha256))
            {
                reason = Common2DTraceFailureReason.InvalidBinding;
                return false;
            }
            if (!IsShapeList(arrayShapes) || !IsFullDigestList(arraySha256s))
            {
                reason = Common2DTraceFailureReason.InvalidPayloadShape;
                return false;
            }
            if (trace.PcaProjection == null ||
                trace.PcaProjection.Length != RequiredComponentCount ||
                trace.LrComponentContributions == null ||
                trace.LrComponentContributions.Length != RequiredComponentCount)
            {
                reason = Common2DTraceFailureReason.InvalidPayloadShape;
                return false;
            }

            string line = Compose(
                trace.CaptureSessionId, trace.RequestedTarget, trace.ArtifactVersion,
                trace.ArtifactSha256, trace.SourceFrameCount, exportId, payloadSha256,
                arrayShapes, arraySha256s,
                Numbers(trace.PcaProjection), Numbers(trace.LrComponentContributions),
                Number(trace.LrIntercept), Number(trace.DecisionValue),
                Number(trace.GreetingProbability), Number(trace.ThankYouProbability),
                Common2DTraceStatus.Committed, Common2DTraceFailureReason.None);
            if (line.Length > MaxLength)
            {
                reason = Common2DTraceFailureReason.MarkerTooLong;
                return false;
            }
            marker = line;
            return true;
        }

        /// <summary>
        /// Builds the bounded failure marker. It carries no payload metadata at all: the
        /// binding fields are emitted only when they are already in approved form, and
        /// every array field is the closed "none" token.
        /// </summary>
        public static string BuildFailure(
            Common2DDiagnosticTrace trace,
            Common2DTraceStatus status,
            Common2DTraceFailureReason reason)
        {
            string session = trace != null &&
                Common2DClosedVocabulary.IsApprovedSessionId(trace.CaptureSessionId)
                    ? trace.CaptureSessionId : Unavailable;
            string target = trace != null &&
                Common2DClosedVocabulary.IsApprovedTarget(trace.RequestedTarget)
                    ? trace.RequestedTarget : Unavailable;
            string version = trace != null &&
                Common2DClosedVocabulary.IsApprovedArtifactVersion(trace.ArtifactVersion)
                    ? trace.ArtifactVersion : Unavailable;
            string artifact = trace != null &&
                Common2DClosedVocabulary.IsLowerHex64(trace.ArtifactSha256)
                    ? trace.ArtifactSha256 : Unavailable;
            int frames = trace == null ? 0 : trace.SourceFrameCount;

            string line = Compose(
                session, target, version, artifact, frames,
                Unavailable, Unavailable, Unavailable, Unavailable,
                Unavailable, Unavailable, Unavailable, Unavailable,
                Unavailable, Unavailable, status, reason);
            // A failure marker is composed only from closed tokens, so it cannot exceed
            // the bound; the check stays as a fail-closed guard rather than a truncation.
            return line.Length <= MaxLength ? line : Compose(
                Unavailable, Unavailable, Unavailable, Unavailable, 0,
                Unavailable, Unavailable, Unavailable, Unavailable,
                Unavailable, Unavailable, Unavailable, Unavailable,
                Unavailable, Unavailable, status,
                Common2DTraceFailureReason.MarkerTooLong);
        }

        static string Compose(
            string captureSessionId, string requestedTarget, string artifactVersion,
            string artifactSha256, int sourceFrameCount, string exportId,
            string payloadSha256, string arrayShapes, string arraySha256s,
            string pca4, string lrContributions4, string lrIntercept,
            string decisionValue, string greetingProbability, string thankYouProbability,
            Common2DTraceStatus status, Common2DTraceFailureReason reason)
        {
            var text = new System.Text.StringBuilder(4096);
            text.Append(Prefix);
            Append(text, "captureSessionId", captureSessionId);
            Append(text, "requestedTarget", requestedTarget);
            Append(text, "artifactVersion", artifactVersion);
            Append(text, "artifactSha256", artifactSha256);
            Append(text, "sourceFrameCount",
                sourceFrameCount.ToString(CultureInfo.InvariantCulture));
            Append(text, "exportId", exportId);
            Append(text, "payloadSha256", payloadSha256);
            Append(text, "arrayShapes", arrayShapes);
            Append(text, "arraySha256s", arraySha256s);
            Append(text, "pca4", pca4);
            Append(text, "lrContributions4", lrContributions4);
            Append(text, "lrIntercept", lrIntercept);
            Append(text, "decisionValue", decisionValue);
            Append(text, "greetingProbability", greetingProbability);
            Append(text, "thankYouProbability", thankYouProbability);
            Append(text, "status", status.ToString());
            Append(text, "typedFailureReason", reason.ToString());
            return text.ToString();
        }

        static void Append(System.Text.StringBuilder text, string field, string value)
        {
            text.Append(' ').Append(field).Append('=').Append(value);
        }

        /// <summary>A fixed-order shape list: "[a,b,c]|[d]|..." with no free text.</summary>
        public static bool IsShapeList(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length > 2048) return false;
            for (int i = 0; i < value.Length; i++)
            {
                char character = value[i];
                bool safe = (character >= '0' && character <= '9') ||
                            character == '[' || character == ']' ||
                            character == ',' || character == '|';
                if (!safe) return false;
            }
            return value[0] == '[';
        }

        /// <summary>A fixed-order list of complete 64-character digests.</summary>
        public static bool IsFullDigestList(string value)
        {
            if (string.IsNullOrEmpty(value)) return false;
            string[] parts = value.Split('|');
            for (int i = 0; i < parts.Length; i++)
                if (!Common2DClosedVocabulary.IsLowerHex64(parts[i])) return false;
            return true;
        }

        static string Number(double value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }

        static string Numbers(double[] values)
        {
            var text = new System.Text.StringBuilder(96);
            for (int i = 0; i < values.Length; i++)
            {
                if (i > 0) text.Append(',');
                text.Append(Number(values[i]));
            }
            return text.ToString();
        }
    }
}
#endif
