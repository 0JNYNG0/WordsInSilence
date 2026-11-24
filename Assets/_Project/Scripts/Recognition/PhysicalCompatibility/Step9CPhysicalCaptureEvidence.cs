using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;

namespace WordsInSilence.Recognition.PhysicalCompatibility
{
    public static class Step9CPhysicalCompatibilityConstants
    {
        public const string SchemaVersion = "words_in_silence.step9c.physical_capture.v1";
        public const string RunId = "8d17ef49-89ff-4d40-ae59-0386cc406e2e";
        public const string MediaPipePackageVersion = "0.16.3";
        public const double FrozenThreshold = 0.2908679008489196;
        public const string ThresholdRule = "inclusive_less_than_or_equal";

        public static readonly string[] FixedWords = { "GREETING", "LOVE", "THANK_YOU" };
    }

    public sealed class PhysicalTakePlan
    {
        public string RunId { get; }
        public string CaptureId { get; }
        public string ExpectedWord { get; }
        public int TakeNumber { get; }

        internal PhysicalTakePlan(string runId, string captureId, string expectedWord, int takeNumber)
        {
            RunId = runId;
            CaptureId = captureId;
            ExpectedWord = expectedWord;
            TakeNumber = takeNumber;
        }
    }

    /// <summary>Enforces fixed order and permits a retake only when capture validity or hash integrity fails.</summary>
    public sealed class FixedPhysicalCaptureProtocol
    {
        readonly string runId;
        int completedTakeCount;
        PhysicalTakePlan pending;

        public int CompletedTakeCount => completedTakeCount;
        public bool IsComplete => completedTakeCount == Step9CPhysicalCompatibilityConstants.FixedWords.Length;
        public string CurrentExpectedWord => IsComplete ? string.Empty : Step9CPhysicalCompatibilityConstants.FixedWords[completedTakeCount];
        public int CurrentTakeNumber => IsComplete ? 0 : completedTakeCount + 1;
        public PhysicalTakePlan Pending => pending;

        public FixedPhysicalCaptureProtocol(string runId)
        {
            if (!Guid.TryParseExact(runId, "D", out _)) throw new ArgumentException("run ID must be a canonical UUID", nameof(runId));
            this.runId = runId;
        }

        public PhysicalTakePlan PrepareCurrent()
        {
            if (IsComplete) throw new InvalidOperationException("all three fixed takes are complete");
            if (pending != null) throw new InvalidOperationException("the current take is already prepared");
            string word = CurrentExpectedWord;
            string captureId = $"step9c-take-{CurrentTakeNumber:D2}-{word.ToLowerInvariant().Replace('_', '-')}-{Guid.NewGuid():D}";
            pending = new PhysicalTakePlan(runId, captureId, word, CurrentTakeNumber);
            return pending;
        }

        public void RestoreVerifiedExport(string captureId, int takeNumber, string expectedWord)
        {
            if (pending != null) throw new InvalidOperationException("cannot restore evidence while a take is prepared");
            if (IsComplete) throw new InvalidOperationException("all three fixed takes are already complete");
            if (takeNumber != CurrentTakeNumber || !string.Equals(expectedWord, CurrentExpectedWord, StringComparison.Ordinal))
                throw new InvalidDataException("verified evidence does not continue the fixed take order");
            if (string.IsNullOrWhiteSpace(captureId)) throw new ArgumentException("capture ID is required", nameof(captureId));
            completedTakeCount++;
        }

        public void CommitExport(string captureId, bool objectivelyValid, bool evidenceHashVerified)
        {
            if (pending == null) throw new InvalidOperationException("no fixed take is prepared");
            if (!string.Equals(captureId, pending.CaptureId, StringComparison.Ordinal))
                throw new ArgumentException("capture ID does not match the prepared take", nameof(captureId));

            // Recognition outcome is deliberately not an input. A valid wrong/rejected result still consumes the take.
            if (objectivelyValid && evidenceHashVerified) completedTakeCount++;
            pending = null;
        }
    }

    public sealed class PhysicalDeviceSnapshot
    {
        public string DeviceName { get; }
        public int RequestedWidth { get; }
        public int RequestedHeight { get; }
        public double RequestedFrameRate { get; }
        public int ActualWidth { get; }
        public int ActualHeight { get; }
        public bool DeviceOpened { get; }
        public bool PreviewObserved { get; }
        public bool ProviderBound { get; }
        public bool ProviderReadinessObserved { get; }
        public int BridgeObservationCount { get; }
        public int ProviderFrameCount { get; }

        public PhysicalDeviceSnapshot(
            string deviceName,
            int requestedWidth,
            int requestedHeight,
            double requestedFrameRate,
            int actualWidth,
            int actualHeight,
            bool deviceOpened,
            bool previewObserved,
            bool providerBound,
            bool providerReadinessObserved,
            int bridgeObservationCount,
            int providerFrameCount = -1)
        {
            DeviceName = deviceName ?? string.Empty;
            RequestedWidth = requestedWidth;
            RequestedHeight = requestedHeight;
            RequestedFrameRate = requestedFrameRate;
            ActualWidth = actualWidth;
            ActualHeight = actualHeight;
            DeviceOpened = deviceOpened;
            PreviewObserved = previewObserved;
            ProviderBound = providerBound;
            ProviderReadinessObserved = providerReadinessObserved;
            BridgeObservationCount = bridgeObservationCount;
            ProviderFrameCount = providerFrameCount < 0 ? bridgeObservationCount : providerFrameCount;
        }
    }

    public sealed class PhysicalCaptureContext
    {
        public PhysicalTakePlan Plan { get; }
        public string CaptureTimeUtc { get; }
        public string CaptureTimeLocal { get; }
        public string UnityVersion { get; }
        public string MediaPipePackageVersion { get; }
        public string SceneName { get; }
        public PhysicalDeviceSnapshot Device { get; }
        public PhysicalCaptureWindowSnapshot CaptureWindow { get; }

        public PhysicalCaptureContext(
            PhysicalTakePlan plan,
            DateTimeOffset captureTimeUtc,
            DateTimeOffset captureTimeLocal,
            string unityVersion,
            string mediaPipePackageVersion,
            string sceneName,
            PhysicalDeviceSnapshot device,
            PhysicalCaptureWindowSnapshot captureWindow = null)
        {
            Plan = plan ?? throw new ArgumentNullException(nameof(plan));
            CaptureTimeUtc = captureTimeUtc.ToUniversalTime().ToString("O");
            CaptureTimeLocal = captureTimeLocal.ToString("O");
            UnityVersion = Require(unityVersion, nameof(unityVersion));
            MediaPipePackageVersion = Require(mediaPipePackageVersion, nameof(mediaPipePackageVersion));
            SceneName = Require(sceneName, nameof(sceneName));
            Device = device ?? throw new ArgumentNullException(nameof(device));
            CaptureWindow = captureWindow;
        }

        static string Require(string value, string parameter)
        {
            if (string.IsNullOrWhiteSpace(value)) throw new ArgumentException("value is required", parameter);
            return value;
        }
    }

    public sealed class PhysicalFrameEvidence
    {
        readonly bool[] normalizedMask;
        readonly double[] normalizedCoordinates;

        public double TimestampSeconds { get; }
        public bool PosePresent { get; }
        public bool LeftHandPresent { get; }
        public bool RightHandPresent { get; }
        public int PoseValidLandmarks { get; }
        public int LeftHandValidLandmarks { get; }
        public int RightHandValidLandmarks { get; }
        public bool BodyTransformAvailable { get; }
        public double ShoulderWidthRaw { get; }
        public bool HasNormalizedFrame { get; }

        public PhysicalFrameEvidence(
            double timestampSeconds,
            bool posePresent,
            bool leftHandPresent,
            bool rightHandPresent,
            int poseValidLandmarks,
            int leftHandValidLandmarks,
            int rightHandValidLandmarks,
            bool bodyTransformAvailable,
            double shoulderWidthRaw,
            bool hasNormalizedFrame,
            bool[] normalizedMask,
            double[] normalizedCoordinates)
        {
            TimestampSeconds = timestampSeconds;
            PosePresent = posePresent;
            LeftHandPresent = leftHandPresent;
            RightHandPresent = rightHandPresent;
            PoseValidLandmarks = poseValidLandmarks;
            LeftHandValidLandmarks = leftHandValidLandmarks;
            RightHandValidLandmarks = rightHandValidLandmarks;
            BodyTransformAvailable = bodyTransformAvailable;
            ShoulderWidthRaw = shoulderWidthRaw;
            HasNormalizedFrame = hasNormalizedFrame;
            if (normalizedMask == null || normalizedMask.Length != 16) throw new ArgumentException("normalized mask must have 16 entries", nameof(normalizedMask));
            if (normalizedCoordinates == null || normalizedCoordinates.Length != 48) throw new ArgumentException("normalized coordinates must have 48 entries", nameof(normalizedCoordinates));
            this.normalizedMask = (bool[])normalizedMask.Clone();
            this.normalizedCoordinates = (double[])normalizedCoordinates.Clone();
        }

        public bool[] CopyNormalizedMask() => (bool[])normalizedMask.Clone();
        public double[] CopyNormalizedCoordinates() => (double[])normalizedCoordinates.Clone();
    }

    public sealed class PhysicalTemplateDistanceEvidence
    {
        public string PrototypeId { get; }
        public string PrototypeWord { get; }
        public string PrototypeSpeed { get; }
        public double Distance { get; }
        public int PathLength { get; }

        public PhysicalTemplateDistanceEvidence(string prototypeId, string prototypeWord, string prototypeSpeed, double distance, int pathLength)
        {
            PrototypeId = prototypeId;
            PrototypeWord = prototypeWord;
            PrototypeSpeed = prototypeSpeed;
            Distance = distance;
            PathLength = pathLength;
        }
    }

    public sealed class PhysicalRecognitionEvidence
    {
        readonly ReadOnlyCollection<PhysicalTemplateDistanceEvidence> templateDistances;

        public string PredictedWord { get; }
        public string WinningPrototypeId { get; }
        public double GreetingDistance { get; }
        public double LoveDistance { get; }
        public double ThankYouDistance { get; }
        public double BestDistance { get; }
        public double Threshold { get; }
        public bool Accepted { get; }
        public bool ExpectedWordMatch { get; }
        public bool AcceptedAndCorrect { get; }
        public IReadOnlyList<PhysicalTemplateDistanceEvidence> TemplateDistances => templateDistances;

        public PhysicalRecognitionEvidence(
            string expectedWord,
            string predictedWord,
            string winningPrototypeId,
            double greetingDistance,
            double loveDistance,
            double thankYouDistance,
            double bestDistance,
            double threshold,
            bool accepted,
            IEnumerable<PhysicalTemplateDistanceEvidence> templateDistances)
        {
            PredictedWord = predictedWord;
            WinningPrototypeId = winningPrototypeId;
            GreetingDistance = greetingDistance;
            LoveDistance = loveDistance;
            ThankYouDistance = thankYouDistance;
            BestDistance = bestDistance;
            Threshold = threshold;
            Accepted = accepted;
            ExpectedWordMatch = string.Equals(expectedWord, predictedWord, StringComparison.Ordinal);
            AcceptedAndCorrect = accepted && ExpectedWordMatch;
            if (templateDistances == null) throw new ArgumentNullException(nameof(templateDistances));
            this.templateDistances = new ReadOnlyCollection<PhysicalTemplateDistanceEvidence>(new List<PhysicalTemplateDistanceEvidence>(templateDistances).ToArray());
        }
    }

    public sealed class PhysicalFailureEvidence
    {
        public string ExceptionType { get; }
        public string Message { get; }

        public PhysicalFailureEvidence(string exceptionType, string message)
        {
            ExceptionType = string.IsNullOrWhiteSpace(exceptionType) ? "UnknownFailure" : exceptionType;
            Message = message ?? string.Empty;
        }
    }

    public enum Step9CHandIdentityOutcome
    {
        ResolvedByPoseWrist,
        Ambiguous,
    }

    /// <summary>Immutable provider-adjacent evidence for one duplicate handedness frame.</summary>
    public sealed class Step9CHandIdentityDiagnostic
    {
        public int ProviderFrameIndex { get; }
        public double ProviderTimestampSeconds { get; }
        public double HandFrameTimestampSeconds { get; }
        public int SourceHandCount { get; }
        public string SourceCategory0 { get; }
        public double SourceScore0 { get; }
        public string SourceCategory1 { get; }
        public double SourceScore1 { get; }
        public double Hand0WristX { get; }
        public double Hand0WristY { get; }
        public double Hand1WristX { get; }
        public double Hand1WristY { get; }
        public double PoseLeftWristX { get; }
        public double PoseLeftWristY { get; }
        public double PoseRightWristX { get; }
        public double PoseRightWristY { get; }
        public double PoseLeftWristVisibility { get; }
        public double PoseRightWristVisibility { get; }
        public double DirectAssignmentCost { get; }
        public double SwappedAssignmentCost { get; }
        public double BestAssignmentMaxDistance { get; }
        public double AssignmentMargin { get; }
        public double PoseWristSeparation { get; }
        public string SelectedHand0Identity { get; }
        public Step9CHandIdentityOutcome Outcome { get; }
        public string Reason { get; }

        public Step9CHandIdentityDiagnostic(
            int providerFrameIndex,
            double providerTimestampSeconds,
            double handFrameTimestampSeconds,
            int sourceHandCount,
            string sourceCategory0,
            double sourceScore0,
            string sourceCategory1,
            double sourceScore1,
            double hand0WristX,
            double hand0WristY,
            double hand1WristX,
            double hand1WristY,
            double poseLeftWristX,
            double poseLeftWristY,
            double poseRightWristX,
            double poseRightWristY,
            double poseLeftWristVisibility,
            double poseRightWristVisibility,
            double directAssignmentCost,
            double swappedAssignmentCost,
            double bestAssignmentMaxDistance,
            double assignmentMargin,
            double poseWristSeparation,
            string selectedHand0Identity,
            Step9CHandIdentityOutcome outcome,
            string reason)
        {
            ProviderFrameIndex = providerFrameIndex;
            ProviderTimestampSeconds = providerTimestampSeconds;
            HandFrameTimestampSeconds = handFrameTimestampSeconds;
            SourceHandCount = sourceHandCount;
            SourceCategory0 = sourceCategory0 ?? string.Empty;
            SourceScore0 = sourceScore0;
            SourceCategory1 = sourceCategory1 ?? string.Empty;
            SourceScore1 = sourceScore1;
            Hand0WristX = hand0WristX;
            Hand0WristY = hand0WristY;
            Hand1WristX = hand1WristX;
            Hand1WristY = hand1WristY;
            PoseLeftWristX = poseLeftWristX;
            PoseLeftWristY = poseLeftWristY;
            PoseRightWristX = poseRightWristX;
            PoseRightWristY = poseRightWristY;
            PoseLeftWristVisibility = poseLeftWristVisibility;
            PoseRightWristVisibility = poseRightWristVisibility;
            DirectAssignmentCost = directAssignmentCost;
            SwappedAssignmentCost = swappedAssignmentCost;
            BestAssignmentMaxDistance = bestAssignmentMaxDistance;
            AssignmentMargin = assignmentMargin;
            PoseWristSeparation = poseWristSeparation;
            SelectedHand0Identity = selectedHand0Identity ?? string.Empty;
            Outcome = outcome;
            Reason = reason ?? string.Empty;
        }
    }

    public sealed class PhysicalCaptureEvidence
    {
        readonly ReadOnlyCollection<PhysicalFrameEvidence> frames;
        readonly ReadOnlyCollection<Step9CHandIdentityDiagnostic> handIdentityDiagnostics;
        readonly ReadOnlyCollection<string> objectiveInvalidReasons;

        public string SchemaVersion => Step9CPhysicalCompatibilityConstants.SchemaVersion;
        public string RunId { get; }
        public string CaptureId { get; }
        public string ExpectedWord { get; }
        public int TakeNumber { get; }
        public string CaptureTimeUtc { get; }
        public string CaptureTimeLocal { get; }
        public string UnityVersion { get; }
        public string MediaPipePackageVersion { get; }
        public string SceneName { get; }
        public PhysicalDeviceSnapshot Device { get; }
        public PhysicalCaptureWindowSnapshot CaptureWindow { get; }
        public string SourceConventionId { get; }
        public string AxisAndDepth { get; }
        public string Handedness { get; }
        public string Mirror { get; }
        public int RawFrameCount { get; }
        public int NormalizedFrameCount { get; }
        public bool TimestampsStrictlyIncreasing { get; }
        public double TimestampDeltaMin { get; }
        public double TimestampDeltaMedian { get; }
        public double TimestampDeltaMax { get; }
        public int PosePresentFrameCount { get; }
        public int LeftHandPresentFrameCount { get; }
        public int RightHandPresentFrameCount { get; }
        public int RawPoseValidLandmarkCount { get; }
        public int RawLeftHandValidLandmarkCount { get; }
        public int RawRightHandValidLandmarkCount { get; }
        public int NormalizedValidLandmarkCount { get; }
        public int InterpolatedLandmarkCount { get; }
        public int BodyTransformAvailableFrameCount { get; }
        public double ShoulderWidthMin { get; }
        public double ShoulderWidthMedian { get; }
        public double ShoulderWidthMax { get; }
        public int TimestampBoundaryCount { get; }
        public int ReappearanceCount { get; }
        public int StrongDepthBoundaryCount { get; }
        public string FinalizationState { get; }
        public PhysicalRecognitionEvidence Recognition { get; }
        public PhysicalFailureEvidence Failure { get; }
        public IReadOnlyList<Step9CHandIdentityDiagnostic> HandIdentityDiagnostics => handIdentityDiagnostics;
        public int DuplicateHandednessFrameCount { get; }
        public int ResolvedDuplicateHandednessFrameCount { get; }
        public int AmbiguousDuplicateHandednessFrameCount { get; }
        public double DuplicateHandednessFrameRatio { get; }
        public bool PreExportObjectivelyValid { get; }
        public IReadOnlyList<string> ObjectiveInvalidReasons => objectiveInvalidReasons;
        public IReadOnlyList<PhysicalFrameEvidence> Frames => frames;

        internal PhysicalCaptureEvidence(
            PhysicalCaptureContext context,
            string sourceConventionId,
            string axisAndDepth,
            string handedness,
            string mirror,
            IList<PhysicalFrameEvidence> frames,
            int normalizedFrameCount,
            bool timestampsStrictlyIncreasing,
            double timestampDeltaMin,
            double timestampDeltaMedian,
            double timestampDeltaMax,
            int posePresentFrameCount,
            int leftHandPresentFrameCount,
            int rightHandPresentFrameCount,
            int rawPoseValidLandmarkCount,
            int rawLeftHandValidLandmarkCount,
            int rawRightHandValidLandmarkCount,
            int normalizedValidLandmarkCount,
            int interpolatedLandmarkCount,
            int bodyTransformAvailableFrameCount,
            double shoulderWidthMin,
            double shoulderWidthMedian,
            double shoulderWidthMax,
            int timestampBoundaryCount,
            int reappearanceCount,
            int strongDepthBoundaryCount,
            string finalizationState,
            PhysicalRecognitionEvidence recognition,
            PhysicalFailureEvidence failure,
            IList<Step9CHandIdentityDiagnostic> handIdentityDiagnostics,
            IList<string> objectiveInvalidReasons)
        {
            RunId = context.Plan.RunId;
            CaptureId = context.Plan.CaptureId;
            ExpectedWord = context.Plan.ExpectedWord;
            TakeNumber = context.Plan.TakeNumber;
            CaptureTimeUtc = context.CaptureTimeUtc;
            CaptureTimeLocal = context.CaptureTimeLocal;
            UnityVersion = context.UnityVersion;
            MediaPipePackageVersion = context.MediaPipePackageVersion;
            SceneName = context.SceneName;
            Device = context.Device;
            CaptureWindow = context.CaptureWindow;
            SourceConventionId = sourceConventionId;
            AxisAndDepth = axisAndDepth;
            Handedness = handedness;
            Mirror = mirror;
            this.frames = new ReadOnlyCollection<PhysicalFrameEvidence>(new List<PhysicalFrameEvidence>(frames).ToArray());
            RawFrameCount = frames.Count;
            NormalizedFrameCount = normalizedFrameCount;
            TimestampsStrictlyIncreasing = timestampsStrictlyIncreasing;
            TimestampDeltaMin = timestampDeltaMin;
            TimestampDeltaMedian = timestampDeltaMedian;
            TimestampDeltaMax = timestampDeltaMax;
            PosePresentFrameCount = posePresentFrameCount;
            LeftHandPresentFrameCount = leftHandPresentFrameCount;
            RightHandPresentFrameCount = rightHandPresentFrameCount;
            RawPoseValidLandmarkCount = rawPoseValidLandmarkCount;
            RawLeftHandValidLandmarkCount = rawLeftHandValidLandmarkCount;
            RawRightHandValidLandmarkCount = rawRightHandValidLandmarkCount;
            NormalizedValidLandmarkCount = normalizedValidLandmarkCount;
            InterpolatedLandmarkCount = interpolatedLandmarkCount;
            BodyTransformAvailableFrameCount = bodyTransformAvailableFrameCount;
            ShoulderWidthMin = shoulderWidthMin;
            ShoulderWidthMedian = shoulderWidthMedian;
            ShoulderWidthMax = shoulderWidthMax;
            TimestampBoundaryCount = timestampBoundaryCount;
            ReappearanceCount = reappearanceCount;
            StrongDepthBoundaryCount = strongDepthBoundaryCount;
            FinalizationState = finalizationState;
            Recognition = recognition;
            Failure = failure;
            this.handIdentityDiagnostics = new ReadOnlyCollection<Step9CHandIdentityDiagnostic>(new List<Step9CHandIdentityDiagnostic>(handIdentityDiagnostics).ToArray());
            DuplicateHandednessFrameCount = handIdentityDiagnostics.Count;
            int resolved = 0;
            foreach (Step9CHandIdentityDiagnostic diagnostic in handIdentityDiagnostics)
                if (diagnostic.Outcome == Step9CHandIdentityOutcome.ResolvedByPoseWrist) resolved++;
            ResolvedDuplicateHandednessFrameCount = resolved;
            AmbiguousDuplicateHandednessFrameCount = handIdentityDiagnostics.Count - resolved;
            DuplicateHandednessFrameRatio = Device.ProviderFrameCount == 0 ? double.NaN : (double)handIdentityDiagnostics.Count / Device.ProviderFrameCount;
            this.objectiveInvalidReasons = new ReadOnlyCollection<string>(new List<string>(objectiveInvalidReasons).ToArray());
            PreExportObjectivelyValid = objectiveInvalidReasons.Count == 0;
        }
    }

    public enum LiveCoordinateCompatibilityClassification
    {
        ProvisionalPass,
        ObservedFail,
        Inconclusive,
    }

    public sealed class CompatibilityCaptureOutcome
    {
        public string ExpectedWord { get; }
        public string PredictedWord { get; }
        public bool Accepted { get; }
        public bool ObjectivelyValid { get; }
        public bool RecognitionComplete { get; }
        public bool StructurallyCompatible { get; }

        public CompatibilityCaptureOutcome(string expectedWord, string predictedWord, bool accepted, bool objectivelyValid, bool recognitionComplete, bool structurallyCompatible)
        {
            ExpectedWord = expectedWord;
            PredictedWord = predictedWord;
            Accepted = accepted;
            ObjectivelyValid = objectivelyValid;
            RecognitionComplete = recognitionComplete;
            StructurallyCompatible = structurallyCompatible;
        }
    }

    public static class PhysicalCompatibilityClassifier
    {
        public static LiveCoordinateCompatibilityClassification Classify(IReadOnlyList<CompatibilityCaptureOutcome> captures)
        {
            if (captures == null || captures.Count != Step9CPhysicalCompatibilityConstants.FixedWords.Length)
                return LiveCoordinateCompatibilityClassification.Inconclusive;
            for (int index = 0; index < captures.Count; index++)
            {
                CompatibilityCaptureOutcome capture = captures[index];
                if (capture == null || !string.Equals(capture.ExpectedWord, Step9CPhysicalCompatibilityConstants.FixedWords[index], StringComparison.Ordinal))
                    return LiveCoordinateCompatibilityClassification.Inconclusive;
                if (!capture.ObjectivelyValid || !capture.RecognitionComplete)
                    return LiveCoordinateCompatibilityClassification.Inconclusive;
            }
            foreach (CompatibilityCaptureOutcome capture in captures)
                if (!capture.StructurallyCompatible || !capture.Accepted || !string.Equals(capture.ExpectedWord, capture.PredictedWord, StringComparison.Ordinal))
                    return LiveCoordinateCompatibilityClassification.ObservedFail;
            return LiveCoordinateCompatibilityClassification.ProvisionalPass;
        }
    }
}
