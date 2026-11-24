// M3_RUNTIME_PATH_A4 - draft, NOT installed.
//
// One narrow concern: on Windows the classic MAX_PATH limit makes File/Directory calls
// fail with DirectoryNotFoundException once a fully qualified path reaches 260
// characters. The built profile's longest prototype path is exactly 260, which is why
// the same built DLL reads the shorter source profile and the extended-form built
// profile but not the built profile at its normal path.
//
// This helper is applied ONLY at file-read boundaries, and only AFTER the existing
// containment and relative-path validation has produced a canonical absolute path. It
// never participates in a containment decision, never normalizes away a traversal, and
// changes no size, SHA-256, schema, threshold or recognition value.
using System;
using System.IO;

namespace WordsInSilence.Recognition.Runtime
{
    public static class RuntimeLongPath
    {
        const string Prefix = @"\\?\";
        const string UncPrefix = @"\\?\UNC\";

        /// <summary>
        /// Returns a form of <paramref name="path"/> that Windows file APIs can open
        /// regardless of length. Non-Windows, already-prefixed and unusable inputs are
        /// returned unchanged, so a caller can always pass the result straight to File
        /// or Directory without a second code path.
        /// </summary>
        public static string ForIo(string path)
        {
            if (string.IsNullOrEmpty(path)) return path;
            if (Path.DirectorySeparatorChar != '\\') return path;   // not Windows
            if (path.StartsWith(Prefix, StringComparison.Ordinal)) return path;

            string full;
            try { full = Path.GetFullPath(path); }
            catch (Exception) { return path; }                      // caller reports it

            // \\?\ disables normalization, so it is only ever applied to an already
            // fully qualified path
            if (full.StartsWith(@"\\", StringComparison.Ordinal))
                return UncPrefix + full.Substring(2);
            return Prefix + full;
        }
    }
}
