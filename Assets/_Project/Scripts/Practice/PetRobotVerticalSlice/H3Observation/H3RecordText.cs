// M3_H3_OBSERVER_R5 - the single sanitizer and bound for everything a record may carry.
// Prepared in the workspace. NOT installed. Supersedes R4, which is preserved.
//
// R4 declared hard bounds but exceeded them: Token took maxLength characters and THEN
// appended a marker, so it returned maxLength + 1; the "none" placeholder could exceed a
// small bound outright; and BoundLine took MaxRecordLength characters and then appended
// nine more. R5 makes every declared bound inclusive of its own truncation marker.
//
// BoundLine also takes a required suffix. The claim limits and the NO_PROMOTION statement
// live at the tail of a record, and blindly cutting the tail would remove exactly the part
// that must never disappear, so a truncation now removes from the MIDDLE and re-attaches
// the suffix. The authoritative reported/omitted class counts are additionally carried in
// the record HEAD, so they survive even a middle truncation.
//
// The filter is a WHITELIST, which is why a path can never survive it: only ASCII letters,
// digits, '.', '_' and '-' pass. Every separator, quote, brace, space, newline, tab and
// control character becomes '_'.
using System.Text;

namespace WordsInSilence.Practice.PetRobotVerticalSlice.H3Observation
{
    public static class H3RecordText
    {
        /// <summary>Reported classes per list. Scoring still covers EVERY configured class.</summary>
        public const int MaxReportedClasses = 8;

        /// <summary>Hard bound on one whole record line, INCLUDING any truncation marker.</summary>
        public const int MaxRecordLength = 1200;

        public const string Empty = "none";
        public const char TruncationMark = '~';

        /// <summary>
        /// The fixed tail every scored record ends with, and the suffix BoundLine protects.
        /// </summary>
        public const string ClaimLimits =
            " limits=unpromoted_diagnostic.different_domains_not_a_controlled_"
            + "identical_feature_trial.shares_not_calibrated_posteriors."
            + "no_quest_or_gate_or_enrollment_or_threshold_change.NO_PROMOTION";

        /// <summary>
        /// Sanitizes and bounds one value. The result is NEVER longer than maxLength, the
        /// truncation marker included, and never null or empty.
        /// </summary>
        public static string Token(string value, int maxLength)
        {
            if (maxLength < 1) maxLength = 1;
            if (string.IsNullOrEmpty(value))
                return Empty.Length <= maxLength ? Empty : Empty.Substring(0, maxLength);

            int copy = value.Length <= maxLength ? value.Length : maxLength;
            var text = new StringBuilder(maxLength);
            for (int index = 0; index < copy; index++)
            {
                char character = value[index];
                bool allowed =
                    (character >= 'a' && character <= 'z') ||
                    (character >= 'A' && character <= 'Z') ||
                    (character >= '0' && character <= '9') ||
                    character == '.' || character == '_' || character == '-';
                text.Append(allowed ? character : '_');
            }
            // the marker replaces the last kept character, so the bound is never exceeded
            if (value.Length > maxLength) text[maxLength - 1] = TruncationMark;
            return text.ToString();
        }

        /// <summary>
        /// Sanitizes and bounds an already-structured metadata string such as the bundle
        /// provenance. It keeps the characters that make "key=value key=value" parseable -
        /// '=', ' ', '|' and '+' - while still removing every path separator, drive colon,
        /// quote, brace, newline, tab and control character, so a path shape can never
        /// survive. Use Token for a single value; use this only where structure matters.
        /// </summary>
        public static string MetadataToken(string value, int maxLength)
        {
            if (maxLength < 1) maxLength = 1;
            if (string.IsNullOrEmpty(value))
                return Empty.Length <= maxLength ? Empty : Empty.Substring(0, maxLength);

            int copy = value.Length <= maxLength ? value.Length : maxLength;
            var text = new StringBuilder(maxLength);
            for (int index = 0; index < copy; index++)
            {
                char character = value[index];
                bool allowed =
                    (character >= 'a' && character <= 'z') ||
                    (character >= 'A' && character <= 'Z') ||
                    (character >= '0' && character <= '9') ||
                    character == '.' || character == '_' || character == '-' ||
                    character == '=' || character == ' ' || character == '|' ||
                    character == '+';
                text.Append(allowed ? character : '_');
            }
            if (value.Length > maxLength) text[maxLength - 1] = TruncationMark;
            return text.ToString();
        }

        /// <summary>
        /// Sanitizes and bounds a label list, appending an explicit overflow count when more
        /// entries exist than are reported. The authoritative counts are also in the record
        /// head, so a middle truncation cannot hide that the report was shortened.
        /// </summary>
        public static string TokenList(string[] values, int maxEntries, int maxLength)
        {
            if (values == null || values.Length == 0) return Empty;
            var text = new StringBuilder();
            int reported = 0;
            for (int index = 0; index < values.Length && reported < maxEntries; index++)
            {
                if (reported > 0) text.Append('|');
                text.Append(Token(values[index], maxLength));
                reported++;
            }
            if (values.Length > reported)
                text.Append('+').Append(values.Length - reported).Append("_more");
            return text.Length == 0 ? Empty : text.ToString();
        }

        /// <summary>
        /// Final guard: one line, never longer than MaxRecordLength including the marker,
        /// and always still ending with the required suffix when one is supplied.
        /// </summary>
        public static string BoundLine(string line, string requiredSuffix)
        {
            string single = string.IsNullOrEmpty(line) ? Empty : line;
            if (single.IndexOf('\n') >= 0 || single.IndexOf('\r') >= 0)
                single = single.Replace('\n', '_').Replace('\r', '_');
            if (single.Length <= MaxRecordLength) return single;

            string suffix = requiredSuffix ?? string.Empty;
            if (suffix.Length + 1 > MaxRecordLength)
            {
                // a suffix that cannot fit is itself bounded rather than silently dropped
                return suffix.Substring(0, MaxRecordLength - 1) + TruncationMark;
            }
            int keep = MaxRecordLength - suffix.Length - 1;
            return single.Substring(0, keep) + TruncationMark + suffix;
        }
    }
}
