using System;

namespace WordsInSilence.Recognition.Common
{
    /// <summary>
    /// Auxiliary common-2D classification seam shared by Practice and World.
    /// Implementations must be side-effect free: no writes, no training, no
    /// acceptance decision.
    /// </summary>
    public interface ICommonSignClassifier
    {
        string ArtifactVersion { get; }
        string ArtifactSha256 { get; }

        /// <summary>
        /// Classifies one already-normalized [T,16,3] / [T,16] sequence against the
        /// target word the game already knows. Never throws for bad input: an
        /// ineligible attempt comes back as a failed classification.
        /// </summary>
        CommonSignClassification Classify(
            double[] coordinates, bool[] validMask, int frameCount, string requestedTarget);
    }

    /// <summary>
    /// The frozen Step 10O v1.6 common-2D classifier. One versioned implementation over
    /// one artifact identity, used by both the Practice route and the World route.
    /// </summary>
    public sealed class FrozenCommonSignClassifier : ICommonSignClassifier
    {
        readonly FrozenCommonSignModel model;
        readonly CommonSignFailureReason loadFailure;
        readonly string loadDetail;

        public string ArtifactVersion { get { return FrozenCommonSignModel.ArtifactVersion; } }
        public string ArtifactSha256
        {
            get { return model == null ? string.Empty : model.ArtifactSha256; }
        }
        public bool ModelAvailable { get { return model != null; } }
        public CommonSignFailureReason LoadFailure { get { return loadFailure; } }
        public string LoadDetail { get { return loadDetail ?? string.Empty; } }

        FrozenCommonSignClassifier(
            FrozenCommonSignModel model, CommonSignFailureReason failure, string detail)
        {
            this.model = model;
            loadFailure = failure;
            loadDetail = detail;
        }

        /// <summary>
        /// Loads from a directory. A failed load yields a classifier that reports the
        /// failure on every call instead of throwing, so callers stay fail-closed.
        /// </summary>
        public static FrozenCommonSignClassifier Load(string modelDirectory)
        {
            CommonSignFailureReason failure;
            string detail;
            FrozenCommonSignModel loaded =
                FrozenCommonSignModel.TryLoad(modelDirectory, out failure, out detail);
            return new FrozenCommonSignClassifier(loaded, failure, detail);
        }

        public CommonSignClassification Classify(
            double[] coordinates, bool[] validMask, int frameCount, string requestedTarget)
        {
            double[] resampledSink = null;
            bool[] resampledMaskSink = null;
            double[] featureSink = null;
            double[] standardizedSink = null;
            double[] projectionSink = null;
            double[] contributionSink = null;
#if UNITY_EDITOR
            // Editor-only observation. When the developer trace is not armed every sink
            // stays null, nothing is allocated, and ClassifyCore runs exactly as it does
            // in a player build.
            // The ownership guard is evaluated before any sink is allocated.
            bool traceArmed = WordsInSilence.Recognition.Diagnostics
                .Common2DDiagnosticTraceBuffer.ShouldObserve();
            if (traceArmed)
            {
              try
              {
                // The sink allocation is itself a fault-injectable diagnostic stage.
                WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                    .ThrowIfSinkAllocationFaultInjected();
                resampledSink = new double[
                    FrozenCommonSignModel.ResampleT * FrozenCommonSignModel.LandmarkCount * 2];
                resampledMaskSink = new bool[
                    FrozenCommonSignModel.ResampleT * FrozenCommonSignModel.LandmarkCount];
                featureSink = new double[FrozenCommonSignModel.FeatureCount];
                standardizedSink = new double[FrozenCommonSignModel.FeatureCount];
                projectionSink = new double[FrozenCommonSignModel.PcaComponents];
                contributionSink = new double[FrozenCommonSignModel.PcaComponents];
              }
              catch (Exception)
              {
                // A diagnostic allocation failure is recorded as a closed typed status
                // and latched for this capture, so the cause survives instead of the
                // observation being silently switched off. It never reaches the
                // classification below.
                WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                    .RecordExternalCollectionFailure(
                        WordsInSilence.Recognition.Diagnostics
                            .Common2DCollectionFaultInjection.SinkAllocationThrows);
                traceArmed = false;
                resampledSink = null;
                resampledMaskSink = null;
                featureSink = null;
                standardizedSink = null;
                projectionSink = null;
                contributionSink = null;
              }
            }
#endif
            double intercept = 0.0;
            CommonSignClassification classification = ClassifyCore(
                coordinates, validMask, frameCount, requestedTarget,
                resampledSink, resampledMaskSink, featureSink,
                standardizedSink, projectionSink, contributionSink, out intercept);
#if UNITY_EDITOR
            // The classification object is already complete before anything is observed.
            if (traceArmed && classification != null && classification.Eligible)
            {
                // Both calls are self-isolating: a diagnostic failure becomes a typed
                // status and the already-constructed classification is returned as is.
                WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                    .SafeRecordFeaturePath(resampledSink, resampledMaskSink, featureSink);
                WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                    .SafeRecordModelPath(
                        FrozenCommonSignModel.ArtifactVersion, ArtifactSha256,
                        standardizedSink, projectionSink, contributionSink,
                        intercept, classification.DecisionValue,
                        classification.GreetingProbability,
                        classification.ThankYouProbability);
            }
#endif
            return classification;
        }

        /// <summary>
        /// The single classification path. The six optional sinks, when non-null, receive
        /// copies of values this path already produced. They are caller-owned buffers, so
        /// no live classifier input array and no frozen model array is ever exposed, and
        /// the arithmetic is identical whether or not they are supplied.
        /// </summary>
        CommonSignClassification ClassifyCore(
            double[] coordinates, bool[] validMask, int frameCount, string requestedTarget,
            double[] resampledSink, bool[] resampledMaskSink, double[] featureSink,
            double[] standardizedSink, double[] projectionSink, double[] contributionSink,
            out double lrIntercept)
        {
            lrIntercept = 0.0;
            string version = FrozenCommonSignModel.ArtifactVersion;
            string sha = ArtifactSha256;

            if (model == null)
            {
                return CommonSignClassification.Failed(
                    version, sha, requestedTarget,
                    loadFailure == CommonSignFailureReason.None
                        ? CommonSignFailureReason.ModelUnavailable
                        : loadFailure,
                    loadDetail, frameCount);
            }
            if (!FrozenCommonSignModel.IsSupportedTarget(requestedTarget))
            {
                return CommonSignClassification.Failed(
                    version, sha, requestedTarget, CommonSignFailureReason.UnsupportedTarget,
                    "target is outside the frozen two-class contract: " +
                    (string.IsNullOrEmpty(requestedTarget) ? "<empty>" : requestedTarget),
                    frameCount);
            }

            double[] features;
            CommonSignFailureReason failure;
            string detail;
#if UNITY_EDITOR
            bool built = CommonSignFeatureBuilder.TryBuildWithTrace(
                coordinates, validMask, frameCount, out features, out failure, out detail,
                resampledSink, resampledMaskSink);
#else
            bool built = CommonSignFeatureBuilder.TryBuild(
                coordinates, validMask, frameCount, out features, out failure, out detail);
#endif
            if (!built)
            {
                return CommonSignClassification.Failed(
                    version, sha, requestedTarget, failure, detail, frameCount);
            }
            if (featureSink != null)
                Array.Copy(features, 0, featureSink, 0, features.Length);

#if UNITY_EDITOR
            double decision = model.DecideWithTrace(
                features, standardizedSink, projectionSink, contributionSink,
                out lrIntercept);
#else
            double decision = model.Decide(features);
#endif
            if (double.IsNaN(decision) || double.IsInfinity(decision))
            {
                return CommonSignClassification.Failed(
                    version, sha, requestedTarget, CommonSignFailureReason.NonFiniteInput,
                    "decision value is non-finite", frameCount);
            }

            double greeting, thankYou;
            FrozenCommonSignModel.ToProbabilities(decision, out greeting, out thankYou);
            return CommonSignClassification.Success(
                version, sha, requestedTarget, greeting, thankYou, decision, frameCount);
        }

        /// <summary>Exposed for parity evidence only.</summary>
        public FrozenCommonSignModel Model { get { return model; } }
    }
}
