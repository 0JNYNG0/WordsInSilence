using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>
    /// Pure, dependency-free composer for the Step 10P v1.0c common-2D observability
    /// record.
    ///
    /// This type is DIAGNOSTIC ONLY. It reads values that the approved recognition and
    /// fusion seams already produced and turns them into one single-line JSON record.
    /// It performs no recognition, no fusion, no persistence and no decision, so
    /// deleting every call to it leaves the recognition result, the guided outcome and
    /// every write policy byte-for-byte unchanged.
    ///
    /// It has no UnityEngine dependency on purpose: the exact record text can be
    /// compiled and executed offline, without the Editor, as reproducible evidence.
    /// </summary>
    public static class CommonSignalDiagnosticRecord
    {
        /// <summary>Stable line prefix a log reader can grep for.</summary>
        public const string Prefix = "WIS_COMMON2D_V1C ";

        /// <summary>
        /// Schema id carried inside every record. Bumped in Step 10W v1.0 because the
        /// record gained the four World-outcome-authority fields. The grep prefix is
        /// deliberately unchanged, and "outcomeKind" keeps its original guided meaning,
        /// so an existing reader of the old fields still reads them correctly.
        /// </summary>
        public const string Schema = "step10w_v1_0_common2d_observability_v2";

        /// <summary>Free-text detail is truncated so a record stays one readable line.</summary>
        public const int MaxDetailLength = 400;

        /// <summary>Marker that replaces any filesystem location found in free text.</summary>
        public const string RedactedPathMarker = "<redacted-path>";

        /// <summary>Marker appended when the text had to be shortened.</summary>
        public const string TruncationMarker = "...";

        // Upstream failure detail is free text produced by the model loader and by the
        // geometry contract, and it can legitimately contain a filesystem location, for
        // example "model directory not found: <directory>".
        //
        // Step 10P v1.0c2: the previous patterns stopped at the first space, so a path
        // such as C:\Users\local-user Smith\secret left "Smith\secret" - a user name and a
        // path fragment - in the text, and a root or single-component path like /secret
        // or ~/secret was not matched at all. Free text gives no reliable way to find
        // where a path ENDS, so the redaction is now FAIL-CLOSED: the earliest
        // path-like start is located and everything from there to the end of the field
        // is replaced by one marker. Nothing after a path survives, which is the point.
        //
        // Only a genuine location start is matched, so ordinary text keeps its numbers
        // and its wording: a slash inside a token such as 0/32 or and/or is not a path
        // start, and the typed failure reason lives in its own field and is untouched.
        static readonly Regex[] PathStartPatterns =
        {
            new Regex(@"file:", RegexOptions.IgnoreCase),
            new Regex(@"[A-Za-z]:[\\/]"),
            new Regex(@"(?<![^\s\(\[:=,;'""])[\\/]{2}[^\s\\/]"),
            new Regex(@"(?<![^\s\(\[:=,;'""])~[\\/]"),
            new Regex(@"(?<![^\s\(\[:=,;'""])[\\/][^\s\\/]"),
        };

        /// <summary>
        /// Deterministically removes filesystem locations and control characters from
        /// free text and bounds the result. The same helper feeds the log record and the
        /// on-screen text, so the two can never disagree.
        ///
        /// The returned value is at most <see cref="MaxDetailLength"/> characters,
        /// including the truncation marker, and is always a single line.
        /// </summary>
        public static string SanitizeDetail(string value, out bool redacted)
        {
            redacted = false;
            if (string.IsNullOrEmpty(value)) return string.Empty;

            var flattened = new StringBuilder(value.Length);
            foreach (char c in value)
            {
                if (c == '\r' || c == '\n' || c == '\t' || c < ' ')
                {
                    flattened.Append(' ');
                    redacted = true;
                    continue;
                }
                flattened.Append(c);
            }
            string text = flattened.ToString();

            int cut = -1;
            foreach (Regex pattern in PathStartPatterns)
            {
                Match match = pattern.Match(text);
                if (match.Success && (cut < 0 || match.Index < cut)) cut = match.Index;
            }
            if (cut >= 0)
            {
                text = text.Substring(0, cut) + RedactedPathMarker;
                redacted = true;
            }

            if (text.Length > MaxDetailLength)
            {
                text = text.Substring(0, MaxDetailLength - TruncationMarker.Length) +
                    TruncationMarker;
                redacted = true;
            }
            return text;
        }

        /// <summary>
        /// Composes one record. Every argument is a value the approved objects already
        /// hold; nothing is recomputed here.
        /// </summary>
        public static string Compose(
            string captureSessionId,
            string expectedWord,
            bool commonSignalPresent,
            string artifactVersion,
            string artifactSha256,
            string requestedTarget,
            bool eligible,
            string failureReason,
            string failureDetail,
            double greetingProbability,
            double thankYouProbability,
            string topClass,
            double topProbability,
            double probabilityMargin,
            bool expectedTargetSupport,
            double decisionValue,
            int sourceFrameCount,
            string commonEvidenceState,
            string outcomeKind,
            string predictedWord,
            string captureState,
            bool strictAccepted,
            bool strictExpectedWordClassAccept,
            bool strictQuestSuccess,
            bool guidedProgressAllowed,
            bool evidenceGrantsSuccess,
            bool evidenceGrantsLearnerError,
            bool evidenceGrantsVeto,
            string questAuthoritativeOutcome,
            string worldOutcomeAuthority,
            string legacyHybridOutcomeKind,
            string outcomeAgreement)
        {
            bool detailRedacted;
            string safeDetail = SanitizeDetail(failureDetail, out detailRedacted);
            var nonFinite = new List<string>();
            var text = new StringBuilder(512);
            text.Append(Prefix).Append('{');
            Str(text, "schema", Schema, true);
            Str(text, "captureSessionId", captureSessionId, false);
            Str(text, "expectedWord", expectedWord, false);
            Bool(text, "commonSignalPresent", commonSignalPresent);

            Str(text, "artifactVersion", artifactVersion, false);
            Str(text, "artifactSha256", artifactSha256, false);
            Str(text, "requestedTarget", requestedTarget, false);
            Bool(text, "eligible", eligible);
            Str(text, "failureReason", failureReason, false);
            Str(text, "failureDetail", safeDetail, false);
            Bool(text, "failureDetailRedacted", detailRedacted);
            Num(text, "greetingProbability", greetingProbability, nonFinite);
            Num(text, "thankYouProbability", thankYouProbability, nonFinite);
            Str(text, "topClass", topClass, false);
            Num(text, "topProbability", topProbability, nonFinite);
            Num(text, "probabilityMargin", probabilityMargin, nonFinite);
            Bool(text, "expectedTargetSupport", expectedTargetSupport);
            Num(text, "decisionValue", decisionValue, nonFinite);
            Int(text, "sourceFrameCount", sourceFrameCount);
            Str(text, "commonEvidenceState", commonEvidenceState, false);

            Str(text, "outcomeKind", outcomeKind, false);
            Str(text, "predictedWord", predictedWord, false);
            Str(text, "captureState", captureState, false);
            Bool(text, "strictAccepted", strictAccepted);
            Bool(text, "strictExpectedWordClassAccept", strictExpectedWordClassAccept);
            Bool(text, "strictQuestSuccess", strictQuestSuccess);
            Bool(text, "guidedProgressAllowed", guidedProgressAllowed);
            Bool(text, "evidenceGrantsSuccess", evidenceGrantsSuccess);
            Bool(text, "evidenceGrantsLearnerError", evidenceGrantsLearnerError);
            Bool(text, "evidenceGrantsVeto", evidenceGrantsVeto);

            // Step 10W v1.0. Step 10V showed that one record carrying a single
            // "outcomeKind" is ambiguous: that field is the GUIDED layer's outcome, and a
            // reader reasonably but wrongly read it as the World branch. The four fields
            // below make the two layers impossible to confuse. "outcomeKind" keeps its
            // original guided meaning for backward compatibility and is not redefined.
            Str(text, "questAuthoritativeOutcome", questAuthoritativeOutcome, false);
            Str(text, "worldOutcomeAuthority", worldOutcomeAuthority, false);
            Str(text, "legacyHybridOutcomeKind", legacyHybridOutcomeKind, false);
            Str(text, "outcomeAgreement", outcomeAgreement, false);

            text.Append(",\"nonFinite\":[");
            for (int index = 0; index < nonFinite.Count; index++)
            {
                if (index > 0) text.Append(',');
                text.Append('"').Append(nonFinite[index]).Append('"');
            }
            text.Append("]}");
            return text.ToString();
        }

        /// <summary>
        /// Minimal, always-valid record used when composing the real one failed. It
        /// carries the exception type name only, never its message, because an
        /// exception message can itself contain a filesystem location.
        /// </summary>
        public static string ComposeDiagnosticFailure(string stage, string exceptionTypeName)
        {
            var text = new StringBuilder(160);
            text.Append(Prefix).Append('{');
            Str(text, "schema", Schema, true);
            Str(text, "diagnosticError", stage, false);
            Str(text, "exceptionType", exceptionTypeName, false);
            text.Append(",\"nonFinite\":[]}");
            return text.ToString();
        }

        static void Str(StringBuilder text, string key, string value, bool first)
        {
            if (!first) text.Append(',');
            text.Append('"').Append(key).Append("\":\"");
            Escape(text, value ?? string.Empty);
            text.Append('"');
        }

        static void Bool(StringBuilder text, string key, bool value)
        {
            text.Append(",\"").Append(key).Append("\":").Append(value ? "true" : "false");
        }

        static void Int(StringBuilder text, string key, int value)
        {
            text.Append(",\"").Append(key).Append("\":")
                .Append(value.ToString(CultureInfo.InvariantCulture));
        }

        static void Num(StringBuilder text, string key, double value, List<string> nonFinite)
        {
            text.Append(",\"").Append(key).Append("\":");
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                // JSON has no NaN or Infinity literal, so the value becomes null and the
                // field name is listed under "nonFinite". No information is lost.
                text.Append("null");
                nonFinite.Add(key);
                return;
            }
            text.Append(value.ToString("R", CultureInfo.InvariantCulture));
        }

        /// <summary>
        /// JSON string escaping that also collapses newlines, so one record is always
        /// exactly one log line.
        /// </summary>
        static void Escape(StringBuilder text, string value)
        {
            for (int index = 0; index < value.Length; index++)
            {
                char c = value[index];
                switch (c)
                {
                    case '"': text.Append("\\\""); break;
                    case '\\': text.Append("\\\\"); break;
                    case '\n': text.Append("\\n"); break;
                    case '\r': text.Append("\\r"); break;
                    case '\t': text.Append("\\t"); break;
                    default:
                        if (c < ' ')
                            text.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else
                            text.Append(c);
                        break;
                }
            }
        }
    }
}
