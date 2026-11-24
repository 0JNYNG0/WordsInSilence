// F21 - the one seam that connects the already finalized, already trailing-trimmed live
// capture to the F20 full-hand pipeline.
//
// This file adds NO geometry, NO classifier, NO threshold and NO number. It reuses the
// approved adapter (AdaptFullHandCommon2D), the approved binary/hash writer for the source
// fingerprint, and calls the F20 pipeline exactly once per capture. Nothing here is a
// gameplay decision: the returned outcome is research evidence carried alongside the legacy
// result, never in place of it. There is no enrollment, no persistence, no camera, no file
// access and no path discovery in this file.
using System;
using System.Collections.Generic;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    /// <summary>Safe, non-secret failure codes for the game seam. No paths, no identities.</summary>
    public static class FullHandBridgeCodes
    {
        public const string None = "";
        public const string CaptureIdentifierInvalid = "BRIDGE_CAPTURE_IDENTIFIER_INVALID";
        public const string EmptyCapture = "BRIDGE_EMPTY_CAPTURE";
        public const string AdaptRefused = "BRIDGE_ADAPT_REFUSED";
        public const string FrameCountDisagreement = "BRIDGE_FRAME_COUNT_DISAGREEMENT";
        public const string ParticipantUnavailable = "BRIDGE_PARTICIPANT_UNAVAILABLE";
        public const string ParticipantChanged = "BRIDGE_PARTICIPANT_CHANGED";
        public const string PipelineRefused = "BRIDGE_PIPELINE_REFUSED";
        public const string UnexpectedFailure = "BRIDGE_UNEXPECTED_FAILURE";

        // configuration-time codes; an ENABLED but invalid configuration surfaces one of
        // these on every capture instead of silently behaving like "disabled"
        public const string ConfigurationParticipantMissing = "BRIDGE_CONFIG_PARTICIPANT_MISSING";
        public const string ConfigurationPathMissing = "BRIDGE_CONFIG_PATH_MISSING";
        public const string ConfigurationGateUnreadable = "BRIDGE_CONFIG_GATE_UNREADABLE";
        public const string ConfigurationGateInvalid = "BRIDGE_CONFIG_GATE_INVALID";
        public const string ConfigurationModelUnreadable = "BRIDGE_CONFIG_MODEL_UNREADABLE";
        public const string ConfigurationModelHashMismatch = "BRIDGE_CONFIG_MODEL_HASH_MISMATCH";
        public const string ConfigurationRefused = "BRIDGE_CONFIG_REFUSED";
    }

    /// <summary>
    /// Immutable typed evidence for ONE capture. <see cref="Succeeded"/> means the pipeline
    /// ran and produced an outcome; it does NOT mean the sign was accepted. A valid F20
    /// abstention and an unsupported declared target are successful bridge results whose
    /// per-method decisions carry that verdict - they are not bridge failures.
    /// </summary>
    public sealed class FullHandGameCaptureResult
    {
        public bool Succeeded { get; private set; }
        public string CaptureSessionId { get; private set; }
        public int FrameCount { get; private set; }
        /// <summary>Empty when <see cref="Succeeded"/>.</summary>
        public string FailureCode { get; private set; }
        public string FailureDetail { get; private set; }
        /// <summary>Non-null only on success.</summary>
        public FullHandRecognitionOutcome Outcome { get; private set; }

        /// <summary>
        /// OPTIONAL retention, off unless the configured trace route asked for it. These are
        /// the SAME bytes the processor already built and hashed for the pipeline - never a
        /// re-adapted, re-normalized or re-scored capture.
        /// </summary>
        readonly byte[] sourceBinary;

        /// <summary>The source fingerprint handed to the pipeline. Empty when not retained.</summary>
        public string SourceSha256 { get; private set; }
        public bool HasSourceBinary { get { return sourceBinary != null; } }
        public int SourceBinaryLength { get { return sourceBinary == null ? 0 : sourceBinary.Length; } }

        /// <summary>A fresh copy every time; the retained array itself never escapes.</summary>
        public byte[] CopySourceBinary()
        {
            return sourceBinary == null ? null : (byte[])sourceBinary.Clone();
        }

        FullHandGameCaptureResult(bool succeeded, string captureSessionId, int frameCount,
                                  string failureCode, string failureDetail,
                                  FullHandRecognitionOutcome outcome,
                                  string sourceSha256, byte[] sourceBinary)
        {
            Succeeded = succeeded;
            CaptureSessionId = captureSessionId ?? string.Empty;
            FrameCount = frameCount;
            FailureCode = failureCode ?? string.Empty;
            FailureDetail = failureDetail ?? string.Empty;
            Outcome = outcome;
            SourceSha256 = sourceSha256 ?? string.Empty;
            this.sourceBinary = sourceBinary == null || sourceBinary.Length == 0
                ? null : (byte[])sourceBinary.Clone();
        }

        /// <summary>Retains nothing. This is the unchanged default path.</summary>
        public static FullHandGameCaptureResult ForOutcome(string captureSessionId, int frameCount,
                                                           FullHandRecognitionOutcome outcome)
        {
            return ForOutcome(captureSessionId, frameCount, outcome, null, null);
        }

        /// <summary>
        /// Retains the already produced binary and its source hash for the local evidence
        /// route only. Passing null for either behaves exactly like the retention-free form.
        /// </summary>
        public static FullHandGameCaptureResult ForOutcome(string captureSessionId, int frameCount,
                                                           FullHandRecognitionOutcome outcome,
                                                           string sourceSha256, byte[] sourceBinary)
        {
            if (outcome == null) throw new ArgumentNullException("outcome");
            return new FullHandGameCaptureResult(true, captureSessionId, frameCount,
                                                 FullHandBridgeCodes.None, string.Empty, outcome,
                                                 sourceSha256, sourceBinary);
        }

        public static FullHandGameCaptureResult ForRefusal(string captureSessionId, int frameCount,
                                                           string failureCode, string failureDetail)
        {
            // a technical refusal never gains an invented score or an invented binary input
            return new FullHandGameCaptureResult(false, captureSessionId, frameCount,
                                                 failureCode, failureDetail, null, null, null);
        }
    }

    /// <summary>
    /// The injectable seam the capture controller owns. Implementations must not throw; the
    /// controller additionally wraps every call, so a faulty implementation can never alter
    /// the legacy capture outcome.
    /// </summary>
    public interface IFullHandGameCaptureProcessor
    {
        FullHandGameCaptureResult Process(string captureSessionId, string declaredTarget,
                                          IReadOnlyList<RawMediaPipeObservation> observations);
    }

    /// <summary>
    /// The production processor. It is bound to ONE participant id at construction and
    /// re-checks the currently active id on EVERY capture through an injected accessor: a
    /// stale or unavailable identity refuses the capture. There is no profile fallback and
    /// no path discovery here.
    /// </summary>
    public sealed class FullHandGameCaptureProcessor : IFullHandGameCaptureProcessor
    {
        readonly FullHandRecognitionPipeline pipeline;
        readonly Func<string> currentParticipantId;
        readonly bool retainSourceBinary;

        public string BoundParticipantId { get; private set; }
        /// <summary>Observable proof that one capture reaches the pipeline exactly once.</summary>
        public int EvaluationCount { get; private set; }

        /// <summary>Unchanged: retains no capture bytes once scoring is done.</summary>
        public FullHandGameCaptureProcessor(FullHandRecognitionPipeline pipeline,
                                            Func<string> currentParticipantIdAccessor)
            : this(pipeline, currentParticipantIdAccessor, false)
        {
        }

        /// <summary>
        /// <paramref name="retainSourceBinary"/> is true ONLY when the operator configured the
        /// local evidence route. It changes no math, no gate and no decision: it merely keeps
        /// the binary this method already built, so nothing has to be rebuilt or rescored.
        /// </summary>
        public FullHandGameCaptureProcessor(FullHandRecognitionPipeline pipeline,
                                            Func<string> currentParticipantIdAccessor,
                                            bool retainSourceBinary)
        {
            if (pipeline == null) throw new ArgumentNullException("pipeline");
            if (currentParticipantIdAccessor == null)
                throw new ArgumentNullException("currentParticipantIdAccessor");
            this.pipeline = pipeline;
            this.currentParticipantId = currentParticipantIdAccessor;
            this.retainSourceBinary = retainSourceBinary;
            BoundParticipantId = pipeline.ParticipantId;
        }

        public FullHandGameCaptureResult Process(string captureSessionId, string declaredTarget,
                                                 IReadOnlyList<RawMediaPipeObservation> observations)
        {
            int frames = observations == null ? 0 : observations.Count;
            if (!IsUsableIdentifier(captureSessionId))
                return FullHandGameCaptureResult.ForRefusal(string.Empty, frames,
                    FullHandBridgeCodes.CaptureIdentifierInvalid,
                    "the capture identifier is not usable as a sample id");

            string active;
            try { active = currentParticipantId(); }
            catch (Exception) { active = null; }
            if (string.IsNullOrEmpty(active))
                return FullHandGameCaptureResult.ForRefusal(captureSessionId, frames,
                    FullHandBridgeCodes.ParticipantUnavailable,
                    "no active participant identity was available for this capture");
            if (!string.Equals(active, BoundParticipantId, StringComparison.Ordinal))
                return FullHandGameCaptureResult.ForRefusal(captureSessionId, frames,
                    FullHandBridgeCodes.ParticipantChanged,
                    "the active participant is no longer the one this processor was bound to");

            if (observations == null || frames == 0)
                return FullHandGameCaptureResult.ForRefusal(captureSessionId, 0,
                    FullHandBridgeCodes.EmptyCapture, "the capture carried no observation");

            // the ONE approved adaptation of this already finalized, already trimmed segment
            string adaptFailure;
            FullHandCommonCapture capture =
                ApprovedLivePreprocessingAdapter.AdaptFullHandCommon2D(observations,
                                                                       out adaptFailure);
            if (capture == null)
                return FullHandGameCaptureResult.ForRefusal(captureSessionId, frames,
                    FullHandBridgeCodes.AdaptRefused, adaptFailure ?? "the adapter refused");
            if (capture.FrameCount != frames)
                return FullHandGameCaptureResult.ForRefusal(captureSessionId, frames,
                    FullHandBridgeCodes.FrameCountDisagreement,
                    "the adapter returned " + capture.FrameCount + " frames for " + frames +
                    " observations");

            // the true source fingerprint of THIS capture, computed in memory only: the same
            // bytes the approved writer would emit, from the same raw observation timestamps
            var timestamps = new double[frames];
            for (int frame = 0; frame < frames; frame++)
                timestamps[frame] = observations[frame].TimestampSeconds;
            byte[] sourceBinary = FullHandObservationSession.BuildBinary(capture, timestamps);
            string sourceSha256 = FullHandObservationSession.Sha256Hex(sourceBinary);
            // dropped immediately unless the evidence route was configured, so the default
            // path holds no capture bytes past this point
            if (!retainSourceBinary) sourceBinary = null;

            // The declared target is forwarded EXACTLY as given, including a null meaning
            // "undeclared". It is never rewritten or dropped here: F20 validates it itself
            // and turns a malformed declaration into its own typed refusal, so a malformed
            // target can never be silently downgraded to "no target was declared". F20
            // consults a valid target only AFTER scoring and can only report
            // UNSUPPORTED_TARGET or a match; it can never force a class to be correct.
            try
            {
                FullHandRecognitionOutcome outcome =
                    pipeline.Evaluate(capture, captureSessionId, sourceSha256, declaredTarget);
                EvaluationCount++;
                return FullHandGameCaptureResult.ForOutcome(captureSessionId, frames, outcome,
                                                            sourceSha256, sourceBinary);
            }
            catch (FullHandFormatException exception)
            {
                return FullHandGameCaptureResult.ForRefusal(captureSessionId, frames,
                    FullHandBridgeCodes.PipelineRefused, exception.Message);
            }
        }

        static bool IsUsableIdentifier(string value)
        {
            if (string.IsNullOrEmpty(value) || value.Length > 128) return false;
            for (int index = 0; index < value.Length; index++)
                if (char.IsControl(value[index]) || char.IsWhiteSpace(value[index])) return false;
            return true;
        }
    }

    /// <summary>
    /// A processor that refuses every capture with one fixed configuration failure. An
    /// ENABLED but invalid configuration is bound to this, so the invalid state stays
    /// visible in the typed result instead of silently degrading to "feature disabled".
    /// </summary>
    public sealed class FullHandGameCaptureRefusingProcessor : IFullHandGameCaptureProcessor
    {
        readonly string failureCode;
        readonly string failureDetail;

        public FullHandGameCaptureRefusingProcessor(string failureCode, string failureDetail)
        {
            this.failureCode = string.IsNullOrEmpty(failureCode)
                ? FullHandBridgeCodes.ConfigurationRefused : failureCode;
            this.failureDetail = failureDetail ?? string.Empty;
        }

        public FullHandGameCaptureResult Process(string captureSessionId, string declaredTarget,
                                                 IReadOnlyList<RawMediaPipeObservation> observations)
        {
            return FullHandGameCaptureResult.ForRefusal(captureSessionId,
                observations == null ? 0 : observations.Count, failureCode, failureDetail);
        }
    }

    /// <summary>The containment boundary the capture controller calls. It never throws.</summary>
    public static class FullHandGameCaptureBridge
    {
        public static FullHandGameCaptureResult SafeProcess(
            IFullHandGameCaptureProcessor processor, string captureSessionId,
            string declaredTarget, IReadOnlyList<RawMediaPipeObservation> observations)
        {
            if (processor == null) return null;
            int frames = observations == null ? 0 : observations.Count;
            try
            {
                FullHandGameCaptureResult result =
                    processor.Process(captureSessionId, declaredTarget, observations);
                return result ?? FullHandGameCaptureResult.ForRefusal(captureSessionId, frames,
                    FullHandBridgeCodes.UnexpectedFailure,
                    "the processor returned no result");
            }
            catch (Exception exception)
            {
                // an unexpected processor fault is evidence, never a game-path failure
                return FullHandGameCaptureResult.ForRefusal(captureSessionId, frames,
                    FullHandBridgeCodes.UnexpectedFailure, exception.GetType().Name);
            }
        }
    }
}
