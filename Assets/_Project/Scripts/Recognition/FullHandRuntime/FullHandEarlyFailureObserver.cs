// F6 - opt-in, OFF-by-default record of a recognition attempt that failed BEFORE any
// snapshot existed, so a future early failure stays diagnosable.
//
// Deliberately independent of the F5 bundle session: an early failure must be recordable
// when no model loaded, no capture finalized and nothing was ever scored. It therefore
// reads only the shared enable flag and the shared output root, never the bundle manifest,
// and it reuses the F5 sink so the fully qualified, not-under-Assets and CreateNew rules
// are exactly the ones already reviewed.
//
// It scores nothing, decides nothing and grants no authority. It records no exception
// message, no frame, no profile content and no path. It does not explain any earlier
// preserved failure; it only makes the next one describable.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    public static class FullHandEarlyFailureObserver
    {
        public const string RecordSchema = "wis_full_hand_early_failure_v1";
        public const string AbsentSessionToken = "(absent)";

        /// <summary>Hard ceiling on records per process. Duplicates are suppressed first.</summary>
        public const int MaxRecords = 512;

        /// <summary>
        /// Per-field character bound for recorded strings. The controller supplies typed
        /// enum names and its own capture id, but this entry point is public, so no caller
        /// can grow a record without limit.
        /// </summary>
        public const int MaxFieldLength = 200;

        /// <summary>Test-only environment source. Production uses the real environment.</summary>
        public static Func<string, string> EnvironmentForTest;

        /// <summary>Test-only sink. When set, no configuration is resolved at all.</summary>
        public static IFullHandObservationSink SinkForTest;

        /// <summary>Test-only console sink. Production uses the Unity log.</summary>
        public static Action<string> LogForTest;

        static bool resolved;
        static FullHandObserverState state = FullHandObserverState.NotAttempted;
        static string reasonCode = string.Empty;
        static IFullHandObservationSink sink;
        // monotonic and never reused, so a failed write cannot be overwritten by the next
        static int issuedRecordNumbers;
        // at most MaxRecords keys: a new key past the cap is counted but never stored, and
        // a key that IS stored is never evicted for the life of this process
        static readonly HashSet<string> recorded = new HashSet<string>(StringComparer.Ordinal);

        public static FullHandObserverState State { get { return state; } }
        public static string ReasonCode { get { return reasonCode; } }
        /// <summary>Stored dedup keys. Bounded by MaxRecords and never evicted.</summary>
        public static int RememberedKeyCount { get { return recorded.Count; } }
        public static int RecordCount { get; private set; }
        public static int WriteFailureCount { get; private set; }
        public static int SuppressedDuplicateCount { get; private set; }
        public static int SuppressedOverflowCount { get; private set; }
        public static string LastStatusLine { get; private set; }

        public static void ResetForTest()
        {
            resolved = false;
            state = FullHandObserverState.NotAttempted;
            reasonCode = string.Empty;
            sink = null;
            issuedRecordNumbers = 0;
            recorded.Clear();
            EnvironmentForTest = null;
            SinkForTest = null;
            LogForTest = null;
            RecordCount = 0;
            WriteFailureCount = 0;
            SuppressedDuplicateCount = 0;
            SuppressedOverflowCount = 0;
            LastStatusLine = string.Empty;
        }

        /// <summary>
        /// The controller's only entry point. Inert unless deliberately enabled, and every
        /// fault - configuration, sink or log - is contained here, so the caller's failure
        /// outcome is exactly what it would have been without this call.
        /// </summary>
        public static void SafeReportFailure(string captureSessionId, string declaredTarget,
                                             string previousState, string failureCode,
                                             string exceptionTypeName, int observationCount)
        {
            try
            {
                IFullHandObservationSink active = Resolve();
                if (active == null) return;

                bool sessionPresent = !string.IsNullOrEmpty(captureSessionId);
                string key = (sessionPresent ? captureSessionId : AbsentSessionToken)
                           + "|" + (previousState ?? string.Empty)
                           + "|" + (failureCode ?? string.Empty)
                           + "|" + (exceptionTypeName ?? string.Empty);
                // membership is tested WITHOUT storing, so an unbounded stream of distinct
                // keys past the cap cannot grow the history. A known key stays a duplicate.
                if (recorded.Contains(key)) { SuppressedDuplicateCount++; return; }
                if (issuedRecordNumbers >= MaxRecords)
                {
                    SuppressedOverflowCount++;
                    return;                       // counted, and deliberately not stored
                }
                recorded.Add(key);

                int number = ++issuedRecordNumbers;
                string stamp = number.ToString("0000", CultureInfo.InvariantCulture);
                string name = "early_failure_" + stamp + ".json";
                try
                {
                    active.WriteText(name, Record(number, sessionPresent, captureSessionId,
                                                  declaredTarget, previousState, failureCode,
                                                  exceptionTypeName, observationCount));
                    RecordCount++;
                    Write("[FullHandEarlyFailure] record=" + stamp
                          + " stage=" + Token(previousState)
                          + " code=" + Token(failureCode)
                          + " exception=" + (string.IsNullOrEmpty(exceptionTypeName)
                                             ? "none" : Token(exceptionTypeName))
                          + " buffered=" + (observationCount < 0 ? "unavailable"
                                            : observationCount.ToString(
                                                CultureInfo.InvariantCulture))
                          + " note=read_only_diagnostic.observer_did_not_score.no_promotion");
                }
                catch (Exception error)
                {
                    // the number is consumed on purpose: nothing is retried or overwritten
                    WriteFailureCount++;
                    Write("[FullHandEarlyFailure] record=" + stamp
                          + " status=WRITE_FAILED reason=" + Token(error.GetType().Name));
                }
            }
            catch (Exception)
            {
                // the capture path must never see a diagnostic fault
            }
        }

        static IFullHandObservationSink Resolve()
        {
            if (SinkForTest != null) return SinkForTest;
            if (resolved) return sink;
            resolved = true;                       // exactly one resolution per process
            Func<string, string> read =
                EnvironmentForTest ?? Environment.GetEnvironmentVariable;
            string enabled;
            try { enabled = read(FullHandAttemptObserver.EnableVariable); }
            catch (Exception) { Fault("ENVIRONMENT_UNREADABLE"); return null; }
            if (!string.Equals(enabled, FullHandAttemptObserver.EnableValue,
                               StringComparison.Ordinal))
            {
                // OFF: no root is read, no directory is touched and nothing is logged
                state = FullHandObserverState.Disabled;
                reasonCode = "NOT_ENABLED";
                return null;
            }
            string root;
            try { root = read(FullHandAttemptObserver.OutputVariable); }
            catch (Exception) { Fault("ENVIRONMENT_UNREADABLE"); return null; }
            // the raw value must already name its own root; no fallback is ever substituted
            if (!FullHandFileObservationSink.IsFullyQualified(root))
            { Fault("OUTPUT_ROOT_NOT_ABSOLUTE"); return null; }
            try
            {
                sink = new FullHandFileObservationSink(root);
                state = FullHandObserverState.Ready;
                reasonCode = string.Empty;
                return sink;
            }
            catch (FullHandFormatException error) { Fault(error.Code); }
            catch (Exception error) { Fault("SETUP_" + error.GetType().Name); }
            return null;
        }

        static void Fault(string code)
        {
            state = FullHandObserverState.Faulted;
            reasonCode = code;
            sink = null;
            Write("[FullHandEarlyFailure] setup=UNAVAILABLE reason=" + Token(code));
        }

        static string Record(int number, bool sessionPresent, string captureSessionId,
                             string declaredTarget, string previousState, string failureCode,
                             string exceptionTypeName, int observationCount)
        {
            var text = new StringBuilder(1024);
            text.Append("{\n \"schema\": \"").Append(RecordSchema).Append("\",\n");
            text.Append(" \"record_kind\": \"early_failure\",\n");
            text.Append(" \"handling\": \"LOCAL_PRIVATE_ONLY_NOT_REDISTRIBUTED\",\n");
            text.Append(" \"record_number\": ").Append(number).Append(",\n");
            // absent metadata stays JSON null: an empty string would look like a real value
            text.Append(" \"stage\": ").Append(Optional(previousState)).Append(",\n");
            text.Append(" \"capture_session_id_present\": ")
                .Append(sessionPresent ? "true" : "false").Append(",\n");
            text.Append(" \"private_capture_session_id\": ")
                .Append(Optional(captureSessionId)).Append(",\n");
            text.Append(" \"private_declared_target\": ")
                .Append(Optional(declaredTarget)).Append(",\n");
            text.Append(" \"declared_target_use\": \"audit metadata only; no branch depends ")
                .Append("on it and this diagnostic scores nothing\",\n");
            text.Append(" \"failure_code\": ").Append(Optional(failureCode)).Append(",\n");
            // TYPE name only: an exception message could carry a path or free text
            text.Append(" \"exception_type\": ").Append(Optional(exceptionTypeName))
                .Append(",\n");
            // a negative count cannot be a real buffer size, so it is reported unavailable
            // rather than fabricated into a valid value
            text.Append(" \"buffered_observation_count\": ")
                .Append(observationCount < 0 ? "null"
                        : observationCount.ToString(CultureInfo.InvariantCulture))
                .Append(",\n");
            text.Append(" \"buffered_observation_count_available\": ")
                .Append(observationCount < 0 ? "false" : "true").Append(",\n");
            // precisely scoped to THIS diagnostic. The attempt may already have been
            // observed or scored elsewhere, for example when a Processing-stage failure
            // follows a completed full-hand observation of the same capture.
            text.Append(" \"observer_loaded_model\": false,\n");
            text.Append(" \"observer_scored\": false,\n");
            text.Append(" \"attempt_model_and_snapshot_state\": ")
                .Append("\"unobserved_by_this_diagnostic\",\n");
            text.Append(" \"note\": \"this diagnostic neither loaded a model nor scored ")
                .Append("anything. Whether the attempt itself had already been observed or ")
                .Append("scored elsewhere is UNOBSERVED here, not false. This record does ")
                .Append("not explain any earlier preserved failure.\"\n}\n");
            return text.ToString();
        }

        static void Write(string line)
        {
            LastStatusLine = line;
            Action<string> console = LogForTest;
            try
            {
                if (console != null) console(line);
                else UnityEngine.Debug.Log(line);
            }
            catch (Exception)
            {
                // a failing log must never reach the capture path
            }
        }

        /// <summary>A JSON string, or null when the value is genuinely absent.</summary>
        static string Optional(string value)
        {
            return string.IsNullOrEmpty(value) ? "null" : Quote(Bounded(value));
        }

        /// <summary>Bounded to MaxFieldLength, with the truncation made visible.</summary>
        static string Bounded(string value)
        {
            if (value == null || value.Length <= MaxFieldLength) return value;
            return value.Substring(0, MaxFieldLength - 1) + "~";
        }

        /// <summary>Console-safe: letters, digits, '.', '_' and '-' only, bounded.</summary>
        static string Token(string value)
        {
            if (string.IsNullOrEmpty(value)) return "none";
            int copy = value.Length < 48 ? value.Length : 48;
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

        static string Quote(string value)
        {
            string source = value ?? string.Empty;
            var text = new StringBuilder(source.Length + 2);
            text.Append('"');
            foreach (char character in source)
            {
                if (character == '"' || character == '\\') text.Append('\\').Append(character);
                else if (character < ' ') text.Append(' ');
                else text.Append(character);
            }
            return text.Append('"').ToString();
        }
    }
}
