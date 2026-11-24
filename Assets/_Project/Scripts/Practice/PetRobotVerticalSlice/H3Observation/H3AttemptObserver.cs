// M3_H3_OBSERVER_R5 - optional, read-only H3 observation of one finalized attempt.
// Prepared in the workspace. NOT installed. Supersedes R4, which is preserved.
//
// WHAT THIS IS. An opt-in diagnostic that applies the FROZEN R2 H3 port to the very same
// finalized, immutable common-2D capture the attempt already produced, and records one
// bounded local comparison line. H3 remains an UNPROMOTED candidate.
//
// WHAT IT NEVER DOES. It never changes the quest result, the success gate, StrictAccepted,
// the outcome, enrollment saving, the candidate repository, the personal profile, input,
// the camera, the DTW references or any threshold, and it writes nothing back into the
// attempt. An observer failure is a diagnostic failure, never a gameplay failure.
//
// R5 reporting correction. The authoritative class counts - scored, reported and omitted -
// now sit in the record HEAD, before the per-class numeric lists, and the fixed claim
// limits are a protected suffix. So if a very large vocabulary ever forces the line to be
// shortened, the truncation removes numeric detail from the MIDDLE while the status, the
// model provenance, the domain labels, the omitted-class count and the NO_PROMOTION
// statement all remain visible. With the current two configured words nothing is
// shortened at all and every numeric pair is present.
//
// SCORING IS NOT BOUNDED. Every configured class is still scored with the unchanged R2
// math; only the REPORT is capped, and the cap is always visible.
//
// DOMAIN HONESTY. The three arms are NOT a controlled identical-feature three-arm accuracy
// trial: personal DTW is the personal pseudo-3D domain, the existing common ML is the
// frozen 1536-feature common-2D model - not the H1/H2 192-feature baseline - and H3 is the
// frozen 36-feature aligned-pair candidate on common-2D.
//
// NormalizedSequence is the personal pseudo-3D domain and is NEVER substituted for a
// missing Common2DCapture. Personal body calibration remains NOT_IMPLEMENTED.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using WordsInSilence.Recognition.H3Runtime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice.H3Observation
{
    /// <summary>Injectable scoring boundary, so a test never needs a real bundle.</summary>
    public interface IH3ObservationScorer
    {
        bool IsReady { get; }
        string NotReadyReason { get; }
        string Provenance { get; }
        H3DecisionResult Score(double[] coordinates, bool[] mask, int frameCount);
    }

    /// <summary>The real scorer: the frozen R2 port over the optional local bundle.</summary>
    public sealed class H3BundleObservationScorer : IH3ObservationScorer
    {
        public bool IsReady { get { return H3ObservationBundle.IsReady; } }
        public string NotReadyReason { get { return H3ObservationBundle.ReasonCode; } }
        public string Provenance { get { return H3ObservationBundle.Provenance; } }

        public H3DecisionResult Score(double[] coordinates, bool[] mask, int frameCount)
        {
            return H3Scorer.Evaluate(H3ObservationBundle.Bundle, H3ObservationBundle.Model,
                                     coordinates, mask, frameCount);
        }
    }

    /// <summary>
    /// Everything the record needs, already extracted from the attempt into private working
    /// copies.
    /// </summary>
    public sealed class H3ObservationInput
    {
        public string AttemptCorrelation = string.Empty;   // opaque technical capture id
        public string ExpectedWord = string.Empty;

        public string DtwPredicted = string.Empty;
        public bool DtwStrictAccepted;
        public bool DtwExpectedClassAccept;

        public bool CommonSignalPresent;
        public string CommonModelVersion = string.Empty;
        public string CommonPredicted = string.Empty;
        public bool CommonEligible;
        public bool CommonExpectedTargetSupport;
        public string CommonFailureReason = string.Empty;

        public bool Common2DAvailable;
        public string Common2DUnavailableReason = string.Empty;
        public int FrameCount;
        public int LandmarkCount;
        public double[] Coordinates;                        // private working copy
        public bool[] Mask;                                 // private working copy
    }

    /// <summary>
    /// Optional observer. Silent unless a bundle resolved AND Install was called.
    /// </summary>
    public static class H3AttemptObserver
    {
        const int MaxRememberedCaptures = 64;

        /// <summary>Where a record goes. The startup integration points this at the log.</summary>
        public static Action<string> Sink;

        /// <summary>Replaceable scoring boundary. Defaults to the real bundle scorer.</summary>
        public static IH3ObservationScorer Scorer = new H3BundleObservationScorer();

        static readonly HashSet<string> observed = new HashSet<string>(StringComparer.Ordinal);
        static readonly Queue<string> observedOrder = new Queue<string>();

        public static int ObservationCount { get; private set; }
        public static string LastRecord { get; private set; }
        public static bool Installed
        {
            get { return CommonSignalAttemptDiagnostics.Observer != null; }
        }

        public static void ResetForTest()
        {
            Sink = null;
            Scorer = new H3BundleObservationScorer();
            observed.Clear();
            observedOrder.Clear();
            ObservationCount = 0;
            LastRecord = string.Empty;
        }

        /// <summary>
        /// Explicit opt-in. Wires the read-only hook on the existing diagnostics seam.
        /// Idempotent. Uninstall restores the previous behaviour exactly.
        /// </summary>
        public static void Install()
        {
            CommonSignalAttemptDiagnostics.Observer = ObserveAttempt;
        }

        public static void Uninstall()
        {
            CommonSignalAttemptDiagnostics.Observer = null;
        }

        /// <summary>
        /// Bounded, at-most-once per capture. Every fault is contained and turned into a
        /// typed reason on the record; nothing propagates to the caller.
        /// </summary>
        public static void ObserveAttempt(PracticeRecognitionAttempt attempt)
        {
            try
            {
                if (attempt == null) return;
                IH3ObservationScorer scorer = Scorer;
                if (scorer == null || !scorer.IsReady) return;   // disabled: record nothing
                if (!TryBeginObservation(attempt.CaptureSessionId)) return;
                Emit(Extract(attempt), scorer);
            }
            catch (Exception)
            {
                // diagnostic-only fail-closed; the capture path must never see this
            }
        }

        /// <summary>Bounded at-most-once admission by opaque capture id.</summary>
        public static bool TryBeginObservation(string captureId)
        {
            if (string.IsNullOrEmpty(captureId)) return false;
            if (observed.Contains(captureId)) return false;
            observed.Add(captureId);
            observedOrder.Enqueue(captureId);
            while (observedOrder.Count > MaxRememberedCaptures)
                observed.Remove(observedOrder.Dequeue());
            return true;
        }

        /// <summary>Reads the attempt into private working values. Never writes back.</summary>
        public static H3ObservationInput Extract(PracticeRecognitionAttempt attempt)
        {
            var input = new H3ObservationInput();
            if (attempt == null)
            {
                input.Common2DUnavailableReason = "no_attempt";
                return input;
            }
            input.AttemptCorrelation = attempt.CaptureSessionId ?? string.Empty;
            input.ExpectedWord = attempt.ExpectedWord ?? string.Empty;
            input.DtwPredicted = attempt.PredictedWord ?? string.Empty;
            input.DtwStrictAccepted = attempt.StrictAccepted;
            input.DtwExpectedClassAccept = attempt.StrictExpectedWordClassAccept;

            var signal = attempt.CommonSignal;
            input.CommonSignalPresent = signal != null;
            if (signal != null)
            {
                input.CommonModelVersion = signal.ArtifactVersion ?? string.Empty;
                input.CommonPredicted = signal.TopClass ?? string.Empty;
                input.CommonEligible = signal.Eligible;
                input.CommonExpectedTargetSupport = signal.ExpectedTargetSupport;
                input.CommonFailureReason = signal.FailureReason.ToString();
            }

            PersonalLiveRecognitionSnapshot snapshot = attempt.Recognition;
            if (snapshot == null)
            {
                input.Common2DUnavailableReason = "no_recognition_snapshot";
                return input;
            }
            CommonTwoDimensionalCapture capture = snapshot.Common2DCapture;
            if (capture == null)
            {
                input.Common2DUnavailableReason =
                    string.IsNullOrEmpty(snapshot.Common2DFailure)
                        ? "common2d_absent" : "common2d_unavailable";
                return input;
            }

            input.FrameCount = capture.FrameCount;
            input.LandmarkCount = capture.LandmarkCount;
            if (capture.LandmarkCount != H3Constants.LandmarkCount)
            {
                input.Common2DUnavailableReason = "landmark_count_unsupported";
                return input;
            }
            if (capture.FrameCount <= 0)
            {
                input.Common2DUnavailableReason = "empty_capture";
                return input;
            }
            input.Coordinates = capture.CopyCoordinates();
            input.Mask = capture.CopyValidMask();
            input.Common2DAvailable = input.Coordinates != null && input.Mask != null;
            if (!input.Common2DAvailable)
                input.Common2DUnavailableReason = "capture_copy_unavailable";
            return input;
        }

        /// <summary>Scores when possible, composes one bounded line, writes it once.</summary>
        public static void Emit(H3ObservationInput input, IH3ObservationScorer scorer)
        {
            if (input == null || scorer == null) return;
            string line;
            try
            {
                line = Compose(input, scorer);
            }
            catch (Exception error)
            {
                line = "[H3Observe] status=OBSERVER_ERROR reason="
                       + H3RecordText.Token(error.GetType().Name, 48)
                       + H3RecordText.ClaimLimits;
            }
            line = H3RecordText.BoundLine(line, H3RecordText.ClaimLimits);
            ObservationCount++;
            LastRecord = line;
            Action<string> sink = Sink;
            if (sink == null) return;
            try { sink(line); }
            catch (Exception)
            {
                // a failing sink is contained; the record still counted above
            }
        }

        static string Compose(H3ObservationInput input, IH3ObservationScorer scorer)
        {
            var text = new StringBuilder(900);
            text.Append("[H3Observe] attempt=")
                .Append(H3RecordText.Token(input.AttemptCorrelation, 48))
                .Append(" expected=").Append(H3RecordText.Token(input.ExpectedWord, 24));
            text.Append(" dtw{domain=personal_pseudo3d predicted=")
                .Append(H3RecordText.Token(input.DtwPredicted, 24))
                .Append(" strict_accepted=").Append(input.DtwStrictAccepted)
                .Append(" expected_class_accept=").Append(input.DtwExpectedClassAccept)
                .Append('}');
            text.Append(" common_ml{domain=common2d_frozen1536 present=")
                .Append(input.CommonSignalPresent)
                .Append(" version=").Append(H3RecordText.Token(input.CommonModelVersion, 40))
                .Append(" predicted=").Append(H3RecordText.Token(input.CommonPredicted, 24))
                .Append(" eligible=").Append(input.CommonEligible)
                .Append(" expected_target_support=").Append(input.CommonExpectedTargetSupport)
                .Append(" failure=").Append(H3RecordText.Token(input.CommonFailureReason, 40))
                .Append('}');

            // MetadataToken, not Token: the provenance is already a structured key=value
            // string built from individually sanitized fields, so keeping '=' and ' ' keeps
            // it parseable while every path separator, drive colon and control character is
            // still removed
            text.Append(" h3{domain=common2d_alignedpair36 ")
                .Append(H3RecordText.MetadataToken(SafeProvenance(scorer), 200));
            if (!input.Common2DAvailable)
            {
                text.Append(" status=NOT_OBSERVED reason=")
                    .Append(H3RecordText.Token(input.Common2DUnavailableReason, 48))
                    .Append(" frames=").Append(input.FrameCount)
                    .Append(" landmarks=").Append(input.LandmarkCount)
                    .Append('}').Append(H3RecordText.ClaimLimits);
                return text.ToString();
            }

            var watch = Stopwatch.StartNew();
            H3DecisionResult decision;
            try
            {
                decision = scorer.Score(input.Coordinates, input.Mask, input.FrameCount);
            }
            catch (H3FormatException error)
            {
                watch.Stop();
                text.Append(" status=TECHNICAL_FAILURE reason=")
                    .Append(H3RecordText.Token(error.Code, 48))
                    .Append('}').Append(H3RecordText.ClaimLimits);
                return text.ToString();
            }
            catch (Exception error)
            {
                watch.Stop();
                text.Append(" status=TECHNICAL_FAILURE reason=")
                    .Append(H3RecordText.Token(error.GetType().Name, 48))
                    .Append('}').Append(H3RecordText.ClaimLimits);
                return text.ToString();
            }
            watch.Stop();

            if (decision == null)
            {
                text.Append(" status=TECHNICAL_FAILURE reason=null_decision}")
                    .Append(H3RecordText.ClaimLimits);
                return text.ToString();
            }

            int scored = decision.Classes == null ? 0 : decision.Classes.Length;
            int reported = scored < H3RecordText.MaxReportedClasses
                ? scored : H3RecordText.MaxReportedClasses;
            text.Append(" status=").Append(H3RecordText.Token(decision.Status, 32))
                .Append(" quality=").Append(H3RecordText.Token(decision.QualityCode, 40))
                .Append(" top1=").Append(H3RecordText.Token(decision.Top1, 24))
                .Append(" predicted=").Append(H3RecordText.Token(decision.Predicted, 24))
                .Append(" best_raw=").Append(Number(decision.BestRawCompatibility))
                .Append(" runner_up_raw=").Append(Number(decision.RunnerUpRawCompatibility))
                .Append(" relative_gate=").Append(Flag(decision.RelativeGatePassed))
                .Append(" absolute_gate=").Append(Flag(decision.AbsoluteGatePassed))
                // authoritative counts live in the HEAD, so a middle truncation cannot hide
                // that the per-class report below was shortened
                .Append(" scored_classes=").Append(scored)
                .Append(" reported_classes=").Append(reported)
                .Append(" omitted_classes=").Append(scored - reported)
                .Append(" thresholds=")
                .Append(Number(H3Decision.RelativeAcceptProbability)).Append('/')
                .Append(Number(H3Decision.RelativeAcceptMargin)).Append('/')
                .Append(Number(H3Decision.AbsoluteCompatibilityFloor));

            AppendPerClass(text, " raw_compatibility", decision.Classes,
                           decision.RawCompatibility);
            AppendPerClass(text, " relative_share_NOT_a_posterior", decision.Classes,
                           decision.RelativeShareNotAPosterior);
            text.Append(" duration_ms=")
                .Append(watch.Elapsed.TotalMilliseconds
                             .ToString("0.###", CultureInfo.InvariantCulture))
                .Append('}').Append(H3RecordText.ClaimLimits);
            return text.ToString();
        }

        static string SafeProvenance(IH3ObservationScorer scorer)
        {
            try { return scorer.Provenance; }
            catch (Exception) { return "provenance_unavailable"; }
        }

        /// <summary>
        /// Reports at most MaxReportedClasses entries with an explicit overflow count. Every
        /// configured class was still SCORED; only this report is capped, and the head
        /// already carries the authoritative scored/reported/omitted counts.
        /// </summary>
        static void AppendPerClass(StringBuilder text, string label, string[] classes,
                                   double[] values)
        {
            text.Append(label).Append('=');
            if (classes == null || values == null || classes.Length != values.Length)
            {
                text.Append(H3RecordText.Empty);
                return;
            }
            int reported = 0;
            for (int index = 0; index < classes.Length
                 && reported < H3RecordText.MaxReportedClasses; index++)
            {
                if (reported > 0) text.Append(',');
                text.Append(H3RecordText.Token(classes[index], 24)).Append(':')
                    .Append(Number(values[index]));
                reported++;
            }
            if (classes.Length > reported)
                text.Append('+').Append(classes.Length - reported).Append("_more");
        }

        static string Flag(bool? value)
        {
            return value.HasValue ? (value.Value ? "true" : "false") : "n_a";
        }

        static string Number(double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value)) return "n_a";
            return value.ToString("0.####", CultureInfo.InvariantCulture);
        }
    }
}
