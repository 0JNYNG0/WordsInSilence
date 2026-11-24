// F7R1 - the pure, testable core of the manual full-hand repeat tool.
//
// It drives the EXISTING harness controller only: prepare, begin, finalize, cancel, reset.
// It constructs no practice session, repository, quest, enrollment, save or PlayerPrefs,
// starts no neutral recalibration, and scores nothing. Raw arrays stay exclusively in the
// already reviewed full-hand observer; this writes small attempt metadata beside them.
//
// READINESS POLICY. The harness recreates its controller unless State == Ready, so
// readiness is prepared ONCE before a take and never touched while a capture is active or
// while a terminal outcome is still unread. An explicit reset happens only after the
// outcome has been preserved.
//
// LABELS. The condition is the intended ground truth REQUESTED of the person; it is never
// a confirmation that the sign was performed correctly.
using System;
using System.Globalization;
using System.Text;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Recognition.FullHandRepeat
{
    /// <summary>The harness surface this tool needs. Injectable so tests need no scene.</summary>
    public interface IFullHandRepeatHarnessPort
    {
        bool LiveInputBound { get; }
        bool EnsureControllerReady();
        bool TrySelectTarget(string word);
        string EffectiveTarget { get; }
        PersonalLiveCaptureState State { get; }
        string CaptureSessionId { get; }
        int BufferedRawFrameCount { get; }
        bool StartCapture();
        bool FinalizeCapture();
        bool CancelCapture();
        bool ResetCapture();
    }

    public static class FullHandRepeatContract
    {
        public const string RecordSchema = "wis_full_hand_repeat_attempt_v1";

        /// <summary>
        /// The person starts and finishes the take. The harness finalizes with a trailing
        /// trim of 0 seconds, so this path applies no trailing trim of its own; it is not
        /// the automatic segmentation the world game route uses.
        /// </summary>
        public const string SegmentationMode = "manual_diagnostic_full_window_v1";
        public const string NoValidSign = "no_valid_sign";
        public const string PositiveCondition = "positive";

        /// <summary>The label is what was ASKED for, never a confirmed KSL judgement.</summary>
        public const string LabelStatus = "requested_unconfirmed";

        public static readonly string[] Conditions =
            { "positive", "idle", "wrong_handshape", "other_motion" };
        public static readonly string[] Partitions = { "development", "final" };

        public static bool IsCondition(string value)
        {
            return Array.IndexOf(Conditions, value) >= 0;
        }

        public static bool IsPartition(string value)
        {
            return Array.IndexOf(Partitions, value) >= 0;
        }

        public static bool IsSupportedTarget(string value)
        {
            if (string.IsNullOrEmpty(value)) return false;
            var words = PersonalLiveRecognitionController.SupportedWords;
            for (int index = 0; index < words.Count; index++)
                if (string.Equals(words[index], value, StringComparison.Ordinal)) return true;
            return false;
        }

        public static string IntendedGroundTruth(string requestedTarget, string condition)
        {
            return string.Equals(condition, PositiveCondition, StringComparison.Ordinal)
                ? requestedTarget : NoValidSign;
        }
    }

    public sealed class FullHandRepeatCoordinator
    {
        public const int MaxAttempts = 512;
        public const int MaxFieldLength = 200;

        readonly IFullHandRepeatHarnessPort harness;
        readonly IFullHandObservationSink sink;
        readonly Action<string> log;
        readonly Func<DateTime> utcNow;

        int issuedAttemptNumbers;
        bool prepared;
        string activeTarget = string.Empty;
        string activeCondition = string.Empty;
        string activePartition = string.Empty;
        string activeStartState = string.Empty;
        string activeCaptureId = string.Empty;
        DateTime activeStartUtc;
        int activeNumber;

        public bool CaptureActive { get; private set; }
        public int AttemptCount { get; private set; }
        public int RecordedCount { get; private set; }
        public int RecordingFaultCount { get; private set; }
        public int SuppressedOverflowCount { get; private set; }
        public int PrepareInvocationCount { get; private set; }
        public string LastReason { get; private set; }
        public string LastStatusLine { get; private set; }
        public string ActiveCaptureSessionId { get { return activeCaptureId; } }

        public FullHandRepeatCoordinator(IFullHandRepeatHarnessPort harness,
                                         IFullHandObservationSink sink, Action<string> log,
                                         Func<DateTime> clock)
        {
            if (harness == null) throw new ArgumentNullException("harness");
            if (sink == null) throw new ArgumentNullException("sink");
            this.harness = harness;
            this.sink = sink;
            this.log = log;
            utcNow = clock ?? (() => DateTime.UtcNow);
            LastReason = string.Empty;
            LastStatusLine = string.Empty;
        }

        /// <summary>
        /// True when a take may be started now. It calls the harness's destructive readiness
        /// rebuild AT MOST once per cycle, and never while capturing or while a terminal
        /// outcome is still unread, so a controller, capture id or early failure is never
        /// destroyed by a status tick.
        /// </summary>
        public bool EnsureReadyForNextTake()
        {
            try
            {
                if (CaptureActive) return false;
                PersonalLiveCaptureState state = harness.State;
                if (state == PersonalLiveCaptureState.Capturing ||
                    state == PersonalLiveCaptureState.Processing)
                    return false;
                if (state == PersonalLiveCaptureState.Completed ||
                    state == PersonalLiveCaptureState.Cancelled ||
                    state == PersonalLiveCaptureState.Failed)
                    return false;                  // an explicit reset must come first
                if (prepared && state == PersonalLiveCaptureState.Ready) return true;
                PrepareInvocationCount++;
                prepared = harness.EnsureControllerReady();
                return prepared;
            }
            catch (Exception)
            {
                prepared = false;
                return false;
            }
        }

        public bool ReadyToBegin
        {
            get
            {
                try
                {
                    return !CaptureActive && harness.LiveInputBound &&
                           harness.State == PersonalLiveCaptureState.Ready;
                }
                catch (Exception) { return false; }
            }
        }

        /// <summary>
        /// Validates the configuration and target, RESERVES the bounded sequence number, and
        /// only then asks the harness to start. A refused or throwing start is preserved as
        /// a failed requested take rather than dropped.
        /// </summary>
        public bool TryBegin(string requestedTarget, string condition, string partition,
                             out string reason)
        {
            reason = string.Empty;
            if (CaptureActive) return Refuse("CAPTURE_ALREADY_ACTIVE", out reason);
            if (!FullHandRepeatContract.IsSupportedTarget(requestedTarget))
                return Refuse("TARGET_NOT_SUPPORTED", out reason);
            if (!FullHandRepeatContract.IsCondition(condition))
                return Refuse("CONDITION_INVALID", out reason);
            if (!FullHandRepeatContract.IsPartition(partition))
                return Refuse("PARTITION_INVALID", out reason);
            try
            {
                if (!harness.LiveInputBound) return Refuse("LIVE_INPUT_NOT_BOUND", out reason);
                if (!EnsureReadyForNextTake()) return Refuse("CONTROLLER_NOT_READY", out reason);
                if (!harness.TrySelectTarget(requestedTarget))
                    return Refuse("TARGET_SELECTION_REFUSED", out reason);
                // the harness restores its own word index whenever it rebuilds a controller,
                // so the effective target is verified rather than assumed
                if (!string.Equals(harness.EffectiveTarget, requestedTarget,
                                   StringComparison.Ordinal))
                    return Refuse("TARGET_SELECTION_NOT_EFFECTIVE", out reason);
            }
            catch (Exception error)
            {
                return Refuse("PREPARE_ERROR_" + error.GetType().Name, out reason);
            }
            if (issuedAttemptNumbers >= MaxAttempts)
            {
                SuppressedOverflowCount++;
                return Refuse("ATTEMPT_LIMIT_REACHED", out reason);
            }

            activeNumber = ++issuedAttemptNumbers;      // reserved before the start attempt
            activeTarget = requestedTarget;
            activeCondition = condition;
            activePartition = partition;
            activeStartUtc = Now();
            activeCaptureId = string.Empty;
            string startFault = null;
            bool started = false;
            try
            {
                activeStartState = harness.State.ToString();
                started = harness.StartCapture();
                if (!started) startFault = "START_REFUSED";
            }
            catch (Exception error)
            {
                startFault = "START_ERROR_" + error.GetType().Name;
            }
            if (!started)
            {
                AttemptCount++;
                // a requested take that never started is still recorded, with no capture id
                Persist(activeNumber, null, startFault, SafeState(), -1, false,
                        activeStartUtc, Now());
                Report(startFault, activeNumber, -1);
                reason = startFault;
                LastReason = startFault;
                return false;
            }

            AttemptCount++;
            try { activeCaptureId = harness.CaptureSessionId ?? string.Empty; }
            catch (Exception) { activeCaptureId = string.Empty; }
            CaptureActive = true;
            Report("BEGIN", activeNumber, SafeBuffered());
            return true;
        }

        /// <summary>
        /// Finishes the take. Returns true only when the harness accepted the finalize AND
        /// the metadata was written; a recording fault is never converted into success.
        /// </summary>
        public bool TryFinish(bool autoFinished, out string reason)
        {
            return Close(true, autoFinished, out reason);
        }

        /// <summary>
        /// Cancels the take. Returns true when the cancel completed and its metadata was
        /// written: a deliberate cancel is a successful RECORDING, not a completed capture.
        /// It returns false only when there was nothing to cancel, the harness refused, or
        /// the metadata could not be written.
        /// </summary>
        public bool TryCancel(out string reason)
        {
            return Close(false, false, out reason);
        }

        /// <summary>
        /// Prepares the next take. Allowed only once the current outcome has been preserved.
        /// The controller issues a fresh capture id on the next begin; no earlier file is
        /// deleted, rewritten or reused.
        /// </summary>
        public bool TryPrepareNext(out string reason)
        {
            reason = string.Empty;
            if (CaptureActive) return Refuse("CAPTURE_STILL_ACTIVE", out reason);
            try
            {
                activeCaptureId = string.Empty;
                prepared = false;
                if (!harness.ResetCapture()) return Refuse("RESET_REFUSED", out reason);
                return true;
            }
            catch (Exception error)
            {
                return Refuse("RESET_ERROR_" + error.GetType().Name, out reason);
            }
        }

        bool Close(bool finish, bool autoFinished, out string reason)
        {
            reason = string.Empty;
            if (!CaptureActive) return Refuse("NO_ACTIVE_CAPTURE", out reason);

            int buffered = SafeBuffered();           // read BEFORE any clearing call
            string captureId = activeCaptureId;
            if (string.IsNullOrEmpty(captureId))
            {
                try { captureId = harness.CaptureSessionId; } catch (Exception) { }
            }
            PersonalLiveCaptureState state = SafeStateValue();
            bool harnessOk = false;
            string outcome;
            string endState;
            if (state == PersonalLiveCaptureState.Failed)
            {
                // the source already failed this capture: do not finalize or cancel over it
                // and do not overwrite its primary failure. The buffer was cleared by that
                // failure, so the count is unavailable rather than an invented zero.
                buffered = -1;
                outcome = "already_failed_before_close";
                endState = state.ToString();
            }
            else
            {
                try
                {
                    harnessOk = finish ? harness.FinalizeCapture() : harness.CancelCapture();
                    endState = SafeState();
                }
                catch (Exception error)
                {
                    endState = "HARNESS_ERROR_" + error.GetType().Name;
                }
                outcome = !finish ? (harnessOk ? "cancelled" : "cancel_refused")
                        : harnessOk ? "completed" : "failed";
            }
            CaptureActive = false;

            bool recorded = Persist(activeNumber, captureId, outcome, endState, buffered,
                                    autoFinished, activeStartUtc, Now());
            activeCaptureId = string.Empty;
            prepared = false;
            Report(outcome + (recorded ? string.Empty : "_RECORDING_FAULT"), activeNumber,
                   buffered);
            if (!recorded) { reason = "RECORDING_FAULT"; LastReason = reason; return false; }
            if (state == PersonalLiveCaptureState.Failed) { reason = outcome; LastReason = reason; }
            return harnessOk;
        }

        bool Persist(int number, string captureId, string outcome, string endState,
                     int buffered, bool autoFinished, DateTime startUtc, DateTime endUtc)
        {
            try
            {
                string stamp = number.ToString("0000", CultureInfo.InvariantCulture);
                sink.WriteText("repeat_attempt_" + stamp + ".json",
                               Record(number, captureId, outcome, endState, buffered,
                                      autoFinished, startUtc, endUtc));
                RecordedCount++;
                return true;
            }
            catch (Exception)
            {
                RecordingFaultCount++;
                return false;
            }
        }

        string Record(int number, string captureId, string outcome, string endState,
                      int buffered, bool autoFinished, DateTime startUtc, DateTime endUtc)
        {
            bool positive = string.Equals(activeCondition,
                                          FullHandRepeatContract.PositiveCondition,
                                          StringComparison.Ordinal);
            double elapsed = (endUtc - startUtc).TotalSeconds;
            var text = new StringBuilder(1400);
            text.Append("{\n \"schema\": \"").Append(FullHandRepeatContract.RecordSchema)
                .Append("\",\n");
            text.Append(" \"record_kind\": \"repeat_attempt\",\n");
            text.Append(" \"handling\": \"LOCAL_PRIVATE_ONLY_NOT_REDISTRIBUTED\",\n");
            text.Append(" \"sequence_number\": ").Append(number).Append(",\n");
            text.Append(" \"private_capture_session_id\": ").Append(Optional(captureId))
                .Append(",\n");
            text.Append(" \"capture_session_id_present\": ")
                .Append(string.IsNullOrEmpty(captureId) ? "false" : "true").Append(",\n");
            text.Append(" \"intended_ground_truth\": ")
                .Append(Optional(FullHandRepeatContract.IntendedGroundTruth(activeTarget,
                                                                            activeCondition)))
                .Append(",\n");
            text.Append(" \"label_status\": \"").Append(FullHandRepeatContract.LabelStatus)
                .Append("\",\n");
            text.Append(" \"requested_condition\": ").Append(Optional(activeCondition))
                .Append(",\n");
            text.Append(" \"harness_api_target\": ").Append(Optional(activeTarget)).Append(",\n");
            text.Append(" \"harness_api_target_is_intended_ground_truth\": ")
                .Append(positive ? "true" : "false").Append(",\n");
            text.Append(" \"partition\": ").Append(Optional(activePartition)).Append(",\n");
            text.Append(" \"segmentation_mode\": \"")
                .Append(FullHandRepeatContract.SegmentationMode).Append("\",\n");
            text.Append(" \"start_state\": ").Append(Optional(activeStartState)).Append(",\n");
            text.Append(" \"end_state\": ").Append(Optional(endState)).Append(",\n");
            text.Append(" \"outcome\": ").Append(Optional(outcome)).Append(",\n");
            text.Append(" \"auto_finished_at_duration_bound\": ")
                .Append(autoFinished ? "true" : "false").Append(",\n");
            text.Append(" \"start_utc\": \"").Append(Stamp(startUtc)).Append("\",\n");
            text.Append(" \"end_utc\": \"").Append(Stamp(endUtc)).Append("\",\n");
            text.Append(" \"elapsed_seconds\": ")
                .Append(elapsed.ToString("0.###", CultureInfo.InvariantCulture)).Append(",\n");
            text.Append(" \"buffered_raw_frame_count\": ")
                .Append(buffered < 0 ? "null"
                        : buffered.ToString(CultureInfo.InvariantCulture)).Append(",\n");
            text.Append(" \"buffered_raw_frame_count_available\": ")
                .Append(buffered < 0 ? "false" : "true").Append(",\n");
            text.Append(" \"notes\": [\n");
            text.Append("  \"label_status requested_unconfirmed: the condition is what was ")
                .Append("ASKED of the person, not a confirmation that the sign was correct\",\n");
            text.Append("  \"for a non-positive condition the selected supported word is ")
                .Append("only the harness API input; the intended ground truth is ")
                .Append(FullHandRepeatContract.NoValidSign).Append("\",\n");
            text.Append("  \"manual full-window take: the person started and finished it and ")
                .Append("the harness finalized with a trailing trim of 0 seconds, so this is ")
                .Append("NOT the automatic segmentation of the world game route and must not ")
                .Append("share an accuracy denominator with those takes\",\n");
            text.Append("  \"a countdown cancelled before the start is not a take and is not ")
                .Append("recorded here\",\n");
            text.Append("  \"no model score, acceptance or correctness is decided or stored ")
                .Append("here; any evaluation lives only in the full-hand observation ")
                .Append("records for this capture id\",\n");
            text.Append("  \"an already_failed_before_close outcome means the source failed ")
                .Append("this capture first; its primary failure and buffered count belong ")
                .Append("to the earlier failure record and were not overwritten\"\n");
            text.Append(" ]\n}\n");
            return text.ToString();
        }

        DateTime Now()
        {
            try { return utcNow(); }
            catch (Exception) { return DateTime.UtcNow; }
        }

        int SafeBuffered()
        {
            try { return harness.BufferedRawFrameCount; }
            catch (Exception) { return -1; }
        }

        PersonalLiveCaptureState SafeStateValue()
        {
            try { return harness.State; }
            catch (Exception) { return PersonalLiveCaptureState.Failed; }
        }

        string SafeState()
        {
            try { return harness.State.ToString(); }
            catch (Exception error) { return "HARNESS_ERROR_" + error.GetType().Name; }
        }

        bool Refuse(string code, out string reason)
        {
            reason = code;
            LastReason = code;
            Report("REFUSED_" + code, activeNumber, -1);
            return false;
        }

        void Report(string status, int number, int buffered)
        {
            string line = "[FullHandRepeat] attempt="
                        + number.ToString("0000", CultureInfo.InvariantCulture)
                        + " status=" + Token(status)
                        + " buffered=" + (buffered < 0 ? "unavailable"
                                          : buffered.ToString(CultureInfo.InvariantCulture))
                        + " note=diagnostic_only.label_requested_unconfirmed";
            LastStatusLine = line;
            try { if (log != null) log(line); }
            catch (Exception) { /* a failing log must never reach the capture path */ }
        }

        static string Stamp(DateTime value)
        {
            return value.ToUniversalTime()
                        .ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture);
        }

        static string Optional(string value)
        {
            if (string.IsNullOrEmpty(value)) return "null";
            string bounded = value.Length <= MaxFieldLength
                ? value : value.Substring(0, MaxFieldLength - 1) + "~";
            var text = new StringBuilder(bounded.Length + 2);
            text.Append('"');
            foreach (char character in bounded)
            {
                if (character == '"' || character == '\\') text.Append('\\').Append(character);
                else if (character < ' ') text.Append(' ');
                else text.Append(character);
            }
            return text.Append('"').ToString();
        }

        static string Token(string value)
        {
            if (string.IsNullOrEmpty(value)) return "none";
            int copy = value.Length < 64 ? value.Length : 64;
            var text = new StringBuilder(copy);
            for (int index = 0; index < copy; index++)
            {
                char character = value[index];
                bool allowed = (character >= 'a' && character <= 'z')
                            || (character >= 'A' && character <= 'Z')
                            || (character >= '0' && character <= '9')
                            || character == '.' || character == '_' || character == '-';
                text.Append(allowed ? character : '_');
            }
            return text.ToString();
        }
    }
}
