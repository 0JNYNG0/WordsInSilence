using System;
using System.Collections.Generic;
using System.Linq;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Recognition.PhysicalCompatibility
{
    /// <summary>Builds immutable, image-free evidence from Step 9B public observations and results.</summary>
    public sealed class PhysicalCaptureEvidenceRecorder
    {
        const double MinimumShoulderWidth = 1e-6;

        readonly RuntimeRecognitionService service;
        readonly ApprovedLivePreprocessingAdapter adapter;

        public PhysicalCaptureEvidenceRecorder(RuntimeRecognitionService service, ApprovedLivePreprocessingAdapter adapter)
        {
            this.service = service ?? throw new ArgumentNullException(nameof(service));
            this.adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));
        }

        public PhysicalCaptureEvidence Create(
            PhysicalCaptureContext context,
            IReadOnlyList<RawMediaPipeObservation> observations,
            ManualDiagnosticCaptureState finalState,
            DiagnosticRecognitionResult step9BResult,
            string failureType,
            string failureMessage,
            IReadOnlyList<Step9CHandIdentityDiagnostic> handIdentityDiagnostics = null)
        {
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (observations == null) throw new ArgumentNullException(nameof(observations));
            RawMediaPipeObservation[] source = observations.ToArray();
            Step9CHandIdentityDiagnostic[] identityDiagnostics = (handIdentityDiagnostics ?? Array.Empty<Step9CHandIdentityDiagnostic>()).ToArray();
            RawObservationConvention convention = source.Length == 0 ? RawObservationConvention.LiveMediaPipe : source[0].Convention;

            bool timestampsStrict = true;
            var deltas = new List<double>();
            int timestampBoundaryCount = 0;
            for (int frame = 0; frame < source.Length; frame++)
            {
                if (source[frame] == null) throw new ArgumentException($"observation {frame} is null", nameof(observations));
                if (!convention.Equals(source[frame].Convention)) throw new ArgumentException("source convention changed within capture", nameof(observations));
                if (frame == 0) continue;
                double delta = source[frame].TimestampSeconds - source[frame - 1].TimestampSeconds;
                deltas.Add(delta);
                if (!IsFinite(delta) || delta <= 0.0) timestampsStrict = false;
                if (!IsFinite(delta) || delta <= 0.0 || delta > ApprovedLivePreprocessingAdapter.TimestampDiscontinuitySeconds)
                    timestampBoundaryCount++;
            }

            double deltaMin = Summary(deltas, SummaryKind.Min);
            double deltaMedian = Summary(deltas, SummaryKind.Median);
            double deltaMax = Summary(deltas, SummaryKind.Max);

            ApprovedPreprocessingResult preprocessing = null;
            PhysicalRecognitionEvidence recognition = null;
            PhysicalFailureEvidence failure = string.IsNullOrWhiteSpace(failureType) && string.IsNullOrWhiteSpace(failureMessage)
                ? null
                : new PhysicalFailureEvidence(failureType, failureMessage);

            if (source.Length > 0 && timestampsStrict)
            {
                try
                {
                    preprocessing = adapter.Adapt(source);
                    if (step9BResult != null)
                    {
                        RuntimeRecognitionResult fullResult = Recognize(context.Plan.CaptureId, preprocessing);
                        RequireSameResult(step9BResult, fullResult);
                        recognition = ToEvidence(context.Plan.ExpectedWord, fullResult);
                    }
                }
                catch (Exception exception)
                {
                    if (failure == null) failure = new PhysicalFailureEvidence(exception.GetType().FullName, exception.Message);
                    preprocessing = null;
                    recognition = null;
                }
            }

            if (finalState == ManualDiagnosticCaptureState.Completed && step9BResult == null && failure == null)
                failure = new PhysicalFailureEvidence(typeof(InvalidOperationException).FullName, "completed capture did not expose a recognition result");

            int posePresentFrames = 0;
            int leftPresentFrames = 0;
            int rightPresentFrames = 0;
            int rawPoseValid = 0;
            int rawLeftValid = 0;
            int rawRightValid = 0;
            int bodyTransformFrames = 0;
            int reappearanceCount = 0;
            int strongDepthCount = 0;
            var shoulderWidths = new List<double>();
            var frames = new List<PhysicalFrameEvidence>(source.Length);

            for (int frame = 0; frame < source.Length; frame++)
            {
                RawMediaPipeObservation observation = source[frame];
                int poseValid = CountValid(observation.Pose);
                int leftValid = CountValid(observation.LeftHand);
                int rightValid = CountValid(observation.RightHand);
                bool posePresent = poseValid > 0;
                bool leftPresent = leftValid > 0;
                bool rightPresent = rightValid > 0;
                if (posePresent) posePresentFrames++;
                if (leftPresent) leftPresentFrames++;
                if (rightPresent) rightPresentFrames++;
                rawPoseValid += poseValid;
                rawLeftValid += leftValid;
                rawRightValid += rightValid;
                CountBoundaries(observation.Pose, ref reappearanceCount, ref strongDepthCount);
                CountBoundaries(observation.LeftHand, ref reappearanceCount, ref strongDepthCount);
                CountBoundaries(observation.RightHand, ref reappearanceCount, ref strongDepthCount);

                double shoulderWidth = ShoulderWidth(observation.Pose);
                bool bodyTransform = DirectBodyTransformAvailable(observation.Pose, shoulderWidth);
                if (IsFinite(shoulderWidth)) shoulderWidths.Add(shoulderWidth);
                if (bodyTransform) bodyTransformFrames++;

                bool hasNormalized = preprocessing != null && frame < preprocessing.Frames.Count;
                bool[] normalizedMask = hasNormalized ? preprocessing.Frames[frame].CopyValidMask() : new bool[16];
                double[] normalizedCoordinates = hasNormalized ? preprocessing.Frames[frame].CopyCoordinates() : NaNCoordinates();
                frames.Add(new PhysicalFrameEvidence(
                    observation.TimestampSeconds,
                    posePresent,
                    leftPresent,
                    rightPresent,
                    poseValid,
                    leftValid,
                    rightValid,
                    bodyTransform,
                    shoulderWidth,
                    hasNormalized,
                    normalizedMask,
                    normalizedCoordinates));
            }

            int normalizedFrameCount = preprocessing?.NormalizedFrameCount ?? 0;
            int normalizedValidCount = preprocessing?.ValidLandmarkCount ?? 0;
            int interpolatedCount = preprocessing?.InterpolatedLandmarkCount ?? 0;
            var invalidReasons = new List<string>();
            if (!context.Device.DeviceOpened) invalidReasons.Add("webcam_device_not_open");
            if (!context.Device.PreviewObserved) invalidReasons.Add("preview_not_observed");
            if (!context.Device.ProviderBound) invalidReasons.Add("provider_bridge_not_bound");
            if (!context.Device.ProviderReadinessObserved) invalidReasons.Add("provider_readiness_not_observed");
            if (source.Length == 0) invalidReasons.Add("zero_frames");
            if (!timestampsStrict) invalidReasons.Add("timestamps_not_strictly_increasing");
            if (bodyTransformFrames == 0) invalidReasons.Add("shoulders_or_body_transform_never_valid");
            if (leftPresentFrames == 0 && rightPresentFrames == 0) invalidReasons.Add("no_hand_observed");
            if (finalState != ManualDiagnosticCaptureState.Completed) invalidReasons.Add("capture_not_completed");
            if (recognition == null) invalidReasons.Add("recognition_result_incomplete");
            if (failure != null) invalidReasons.Add("typed_failure_observed");
            foreach (string reason in Step9CHandIdentityCapturePolicy.InvalidReasons(context.Device.ProviderFrameCount, identityDiagnostics))
                if (!invalidReasons.Contains(reason)) invalidReasons.Add(reason);

            return new PhysicalCaptureEvidence(
                context,
                convention.Id,
                convention.AxisAndDepth,
                convention.Handedness,
                convention.Mirror,
                frames,
                normalizedFrameCount,
                timestampsStrict,
                deltaMin,
                deltaMedian,
                deltaMax,
                posePresentFrames,
                leftPresentFrames,
                rightPresentFrames,
                rawPoseValid,
                rawLeftValid,
                rawRightValid,
                normalizedValidCount,
                interpolatedCount,
                bodyTransformFrames,
                Summary(shoulderWidths, SummaryKind.Min),
                Summary(shoulderWidths, SummaryKind.Median),
                Summary(shoulderWidths, SummaryKind.Max),
                timestampBoundaryCount,
                reappearanceCount,
                strongDepthCount,
                finalState.ToString(),
                recognition,
                failure,
                identityDiagnostics,
                invalidReasons);
        }

        RuntimeRecognitionResult Recognize(string captureId, ApprovedPreprocessingResult preprocessing)
        {
            var input = new RecognitionSequenceInputSession();
            input.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled(captureId));
            foreach (NormalizedRecognitionFrame frame in preprocessing.Frames) input.AddFrame(captureId, frame);
            return service.Recognize(captureId, input.FinalizeSequence(captureId));
        }

        static PhysicalRecognitionEvidence ToEvidence(string expectedWord, RuntimeRecognitionResult result)
        {
            var distances = result.TemplateDistances.Select(item => new PhysicalTemplateDistanceEvidence(
                item.PrototypeId, item.PrototypeWord, item.PrototypeSpeed, item.Distance, item.PathLength));
            return new PhysicalRecognitionEvidence(
                expectedWord,
                result.PredictedWord,
                result.WinningPrototypeId,
                result.GreetingDistance,
                result.LoveDistance,
                result.ThankYouDistance,
                result.BestDistance,
                result.Threshold,
                result.Accepted,
                distances);
        }

        static void RequireSameResult(DiagnosticRecognitionResult step9B, RuntimeRecognitionResult full)
        {
            if (!string.Equals(step9B.PredictedWord, full.PredictedWord, StringComparison.Ordinal) ||
                !string.Equals(step9B.WinningPrototypeId, full.WinningPrototypeId, StringComparison.Ordinal) ||
                step9B.GreetingDistance != full.GreetingDistance ||
                step9B.LoveDistance != full.LoveDistance ||
                step9B.ThankYouDistance != full.ThankYouDistance ||
                step9B.BestDistance != full.BestDistance ||
                step9B.Threshold != full.Threshold ||
                step9B.Accepted != full.Accepted)
                throw new InvalidOperationException("Step 9B diagnostic result differs from full Step 9A replay");
            if (full.Threshold != Step9CPhysicalCompatibilityConstants.FrozenThreshold)
                throw new InvalidOperationException("frozen threshold changed");
            if (full.TemplateDistances.Count != 9) throw new InvalidOperationException("exactly nine DTW distances are required");
        }

        static int CountValid(RawLandmarkObservationSet set)
        {
            int count = 0;
            for (int landmark = 0; landmark < set.LandmarkCount; landmark++) if (set.IsValid(landmark)) count++;
            return count;
        }

        static void CountBoundaries(RawLandmarkObservationSet set, ref int reappearance, ref int strong)
        {
            for (int landmark = 0; landmark < set.LandmarkCount; landmark++)
            {
                if (set.IsReappearance(landmark)) reappearance++;
                if (set.IsStrongDepthBoundary(landmark)) strong++;
            }
        }

        static double ShoulderWidth(RawLandmarkObservationSet pose)
        {
            const int left = 11, right = 12;
            if (!pose.IsValid(left) || !pose.IsValid(right)) return double.NaN;
            double x = pose.Coordinate(right, 0) - pose.Coordinate(left, 0);
            double y = pose.Coordinate(right, 1) - pose.Coordinate(left, 1);
            double z = pose.Coordinate(right, 2) - pose.Coordinate(left, 2);
            return Math.Sqrt(x * x + y * y + z * z);
        }

        static bool DirectBodyTransformAvailable(RawLandmarkObservationSet pose, double shoulderWidth)
        {
            if (!IsFinite(shoulderWidth) || shoulderWidth < MinimumShoulderWidth) return false;
            double[] right = Subtract(pose, 12, 11);
            Scale(right, 1.0 / shoulderWidth);
            double[] origin = Midpoint(pose, 11, 12);
            bool leftHip = pose.IsValid(23);
            bool rightHip = pose.IsValid(24);
            double[] up;
            if (leftHip && rightHip)
            {
                double[] hips = Midpoint(pose, 23, 24);
                up = new[] { origin[0] - hips[0], origin[1] - hips[1], origin[2] - hips[2] };
            }
            else if (leftHip) up = Subtract(origin, Point(pose, 23));
            else if (rightHip) up = Subtract(origin, Point(pose, 24));
            else up = new[] { 0.0, -1.0, 0.0 };
            double upNorm = Norm(up);
            if (upNorm < MinimumShoulderWidth) up = new[] { 0.0, -1.0, 0.0 };
            else Scale(up, 1.0 / upNorm);
            return Norm(Cross(right, up)) >= MinimumShoulderWidth;
        }

        enum SummaryKind { Min, Median, Max }

        static double Summary(IList<double> values, SummaryKind kind)
        {
            double[] finite = values.Where(IsFinite).OrderBy(value => value).ToArray();
            if (finite.Length == 0) return double.NaN;
            if (kind == SummaryKind.Min) return finite[0];
            if (kind == SummaryKind.Max) return finite[finite.Length - 1];
            int middle = finite.Length / 2;
            return finite.Length % 2 == 1 ? finite[middle] : (finite[middle - 1] + finite[middle]) / 2.0;
        }

        static double[] NaNCoordinates()
        {
            var result = new double[48];
            for (int index = 0; index < result.Length; index++) result[index] = double.NaN;
            return result;
        }

        static double[] Point(RawLandmarkObservationSet set, int landmark) => new[]
        {
            set.Coordinate(landmark, 0), set.Coordinate(landmark, 1), set.Coordinate(landmark, 2),
        };
        static double[] Midpoint(RawLandmarkObservationSet set, int a, int b)
        {
            double[] left = Point(set, a), right = Point(set, b);
            return new[] { (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0, (left[2] + right[2]) / 2.0 };
        }
        static double[] Subtract(RawLandmarkObservationSet set, int a, int b) => Subtract(Point(set, a), Point(set, b));
        static double[] Subtract(double[] a, double[] b) => new[] { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
        static double[] Cross(double[] a, double[] b) => new[] { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
        static double Norm(double[] value) => Math.Sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        static void Scale(double[] value, double scale) { value[0] *= scale; value[1] *= scale; value[2] *= scale; }
        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
