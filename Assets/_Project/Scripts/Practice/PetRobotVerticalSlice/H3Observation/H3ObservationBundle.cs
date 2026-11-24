// M3_H3_OBSERVER_R5 - opt-in bundle source for the frozen H3 diagnostic observer.
// Prepared in the workspace. NOT installed. Supersedes R4, which is preserved.
//
// The runtime model comes ONLY from an explicit local command-line flag:
//
//     -wisH3BundleRoot <fully qualified local directory, e.g. D:\private\wis_h3_bundle>
//
// Absent flag means completely disabled: nothing is loaded, scored, installed or logged.
// No bundled model, no reference bank in the project, no StreamingAssets or Resources
// fallback, no working-directory default, no learner-path default, no training, export,
// subprocess or network.
//
// R5 path correction. R4 used Path.IsPathRooted, which on Windows also accepts forms that
// are NOT fully qualified or NOT local: a drive-relative "C:bundle" resolves against that
// drive's current directory, a root-relative "\bundle" or "/bundle" resolves against the
// process drive, and "\\server\share", "\\?\...", "\\.\..." and "file://..." reach UNC,
// device or URI namespaces. All of those contradict the explicit LOCAL-only contract, so
// R5 validates a fully qualified ordinary local drive path itself and refuses everything
// else BEFORE any existence check or loader call. No drive is enumerated, no permission is
// inspected and the flag value is still never logged.
using System;
using System.IO;
using WordsInSilence.Recognition.H3Runtime;

namespace WordsInSilence.Practice.PetRobotVerticalSlice.H3Observation
{
    public enum H3BundleState
    {
        NotAttempted,
        Ready,
        FlagAbsent,
        FlagDuplicated,
        FlagValueMissing,
        ArgumentListTooLong,
        BundleRootNotFullyQualifiedLocal,
        BundleRootMissing,
        BundleInvalid,
        BundleLoadFailed,
    }

    /// <summary>
    /// One-time, fail-closed resolution of the optional H3 bundle. Nothing here throws to a
    /// caller: every fault becomes a state plus a short non-personal reason code.
    /// </summary>
    public static class H3ObservationBundle
    {
        public const string FlagName = "-wisH3BundleRoot";
        public const int MaxScannedArguments = 256;

        static bool resolved;
        static H3BundleState state = H3BundleState.NotAttempted;
        static string reasonCode = string.Empty;
        static H3RuntimeBundle bundle;
        static H3PairModel model;
        static string provenance = string.Empty;

        /// <summary>How many times a loader was actually invoked. Diagnostics and tests.</summary>
        public static int LoadAttempts { get; private set; }

        public static H3BundleState State { get { Resolve(); return state; } }
        public static string ReasonCode { get { Resolve(); return reasonCode; } }
        public static bool IsReady { get { Resolve(); return state == H3BundleState.Ready; } }
        public static H3RuntimeBundle Bundle { get { Resolve(); return bundle; } }
        public static H3PairModel Model { get { Resolve(); return model; } }
        public static string Provenance { get { Resolve(); return provenance; } }

        /// <summary>Test-only argument source. Production uses the real command line.</summary>
        public static Func<string[]> CommandLineSourceForTest;

        /// <summary>Test-only existence check. Production uses Directory.Exists.</summary>
        public static Func<string, bool> DirectoryExistsForTest;

        /// <summary>Test-only loader. Production uses H3RuntimeFormat.Load.</summary>
        public static Func<string, H3RuntimeBundle> LoaderForTest;

        public static void ResetForTest()
        {
            resolved = false;
            state = H3BundleState.NotAttempted;
            reasonCode = string.Empty;
            bundle = null;
            model = null;
            provenance = string.Empty;
            LoadAttempts = 0;
            CommandLineSourceForTest = null;
            DirectoryExistsForTest = null;
            LoaderForTest = null;
        }

        static void Fail(H3BundleState failure, string code)
        {
            state = failure;
            reasonCode = code;
            bundle = null;
            model = null;
            provenance = string.Empty;
        }

        static void Resolve()
        {
            if (resolved) return;
            resolved = true;                       // exactly one resolution per process
            string root;
            H3BundleState flagState = TryReadFlag(out root);
            if (flagState != H3BundleState.Ready)
            {
                Fail(flagState, flagState.ToString());
                return;
            }
            try
            {
                Func<string, bool> exists = DirectoryExistsForTest ?? Directory.Exists;
                if (!exists(root))
                {
                    Fail(H3BundleState.BundleRootMissing, "bundle_root_missing");
                    return;
                }
                LoadAttempts++;
                bundle = LoaderForTest != null
                    ? LoaderForTest(root)
                    : H3RuntimeFormat.Load(root);
                if (bundle == null)
                {
                    Fail(H3BundleState.BundleLoadFailed, "bundle_null");
                    return;
                }
                model = H3PairModel.FromBundle(bundle);
                provenance = BuildProvenance(bundle);
                state = H3BundleState.Ready;
                reasonCode = string.Empty;
            }
            catch (H3FormatException error)
            {
                Fail(H3BundleState.BundleInvalid,
                     "bundle_invalid_" + H3RecordText.Token(error.Code, 40));
            }
            catch (Exception error)
            {
                Fail(H3BundleState.BundleLoadFailed,
                     "bundle_load_failed_" + H3RecordText.Token(error.GetType().Name, 40));
            }
        }

        /// <summary>
        /// Bounded, explicit flag parse. Exactly one occurrence, exactly one following
        /// value, and the value must be a fully qualified ordinary local drive path.
        /// </summary>
        static H3BundleState TryReadFlag(out string value)
        {
            value = null;
            string[] arguments;
            try
            {
                arguments = CommandLineSourceForTest != null
                    ? CommandLineSourceForTest()
                    : Environment.GetCommandLineArgs();
            }
            catch (Exception)
            {
                return H3BundleState.FlagAbsent;
            }
            if (arguments == null) return H3BundleState.FlagAbsent;
            if (arguments.Length > MaxScannedArguments)
            {
                // a truncated scan could not prove there is no second occurrence further
                // along, so this refuses rather than accepting an unverified single match
                return H3BundleState.ArgumentListTooLong;
            }

            int found = 0;
            string candidate = null;
            for (int index = 0; index < arguments.Length; index++)
            {
                if (!string.Equals(arguments[index], FlagName, StringComparison.Ordinal))
                    continue;
                found++;
                if (found > 1) return H3BundleState.FlagDuplicated;
                candidate = index + 1 < arguments.Length ? arguments[index + 1] : null;
            }
            if (found == 0) return H3BundleState.FlagAbsent;
            if (string.IsNullOrEmpty(candidate)) return H3BundleState.FlagValueMissing;
            string trimmed = candidate.Trim();
            if (trimmed.Length == 0) return H3BundleState.FlagValueMissing;
            if (trimmed[0] == '-') return H3BundleState.FlagValueMissing;
            if (!IsFullyQualifiedLocalDirectory(trimmed))
                return H3BundleState.BundleRootNotFullyQualifiedLocal;
            value = trimmed;
            return H3BundleState.Ready;
        }

        /// <summary>
        /// True only for a fully qualified ORDINARY LOCAL drive path on this Windows Player:
        /// a drive letter, ':', then a separator, with no control or invalid characters.
        ///
        /// Deliberately refused, because none of them is a fully qualified local directory:
        ///   C:bundle        drive-relative, resolves against that drive's current directory
        ///   \bundle  /b     root-relative, resolves against the process drive
        ///   \\host\share    UNC, which is a network location
        ///   \\?\  \\.\      extended-length and device namespaces
        ///   file://  http:  URI forms
        /// No drive is enumerated and no permission is inspected; this is pure string shape.
        /// </summary>
        public static bool IsFullyQualifiedLocalDirectory(string value)
        {
            if (string.IsNullOrEmpty(value)) return false;
            if (value.Length < 3) return false;
            if (value.IndexOf("://", StringComparison.Ordinal) >= 0) return false;

            char drive = value[0];
            bool letter = (drive >= 'a' && drive <= 'z') || (drive >= 'A' && drive <= 'Z');
            if (!letter) return false;                       // also refuses \\ and / and \
            if (value[1] != ':') return false;               // refuses "file:..." and names
            char separator = value[2];
            if (separator != '\\' && separator != '/') return false;   // refuses "C:bundle"

            for (int index = 0; index < value.Length; index++)
            {
                char character = value[index];
                if (character < ' ' || character == '\u007f') return false;
            }
            try
            {
                char[] invalid = Path.GetInvalidPathChars();
                for (int index = 0; index < invalid.Length; index++)
                    if (value.IndexOf(invalid[index]) >= 0) return false;
            }
            catch (Exception)
            {
                return false;
            }
            return true;
        }

        /// <summary>
        /// Sanitized and bounded provenance. Every field passes the shared token filter and
        /// the class list is capped with an explicit overflow count.
        /// </summary>
        static string BuildProvenance(H3RuntimeBundle loaded)
        {
            string digest = H3RecordText.Token(loaded.PayloadSha256, 12);
            string modelVersion = H3RecordText.Token(Version(loaded, "model"), 32);
            string features = H3RecordText.Token(Version(loaded, "pair_features"), 32);
            string alignment = H3RecordText.Token(Version(loaded, "alignment"), 32);
            string classes = H3RecordText.TokenList(loaded.Vocabulary,
                                                    H3RecordText.MaxReportedClasses, 24);
            return "model=" + modelVersion + " pair_features=" + features
                   + " alignment=" + alignment + " classes=" + classes
                   + " payload=" + digest;
        }

        static string Version(H3RuntimeBundle loaded, string key)
        {
            string value;
            if (loaded.Versions != null && loaded.Versions.TryGetValue(key, out value))
                return value ?? string.Empty;
            return string.Empty;
        }
    }
}
