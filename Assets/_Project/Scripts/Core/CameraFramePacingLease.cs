// F58 - opt-in bounded frame pacing lease, held ONLY while a camera owner has a starting or
// playing camera. Reversible A/B candidate, NOT a proven camera fix.
//
// Enabled only by WIS_CAMERA_FRAME_PACING=1. It never opens, closes or reopens a camera, never
// runs per frame, never touches QualitySettings.vSyncCount or any other setting, and restores
// the previous Application.targetFrameRate exactly (including -1) when the last owner releases.
//
// Cap rule: an existing lower positive cap (< 60) is left untouched, because the operator or
// platform already asked for less. Uncapped (-1) or a higher cap (> 60) is bounded to 60.
//
// Ownership is the exact same simple refcount pattern as CameraBackgroundLease.
using System;
using UnityEngine;

namespace WordsInSilence.Core
{
    public static class CameraFramePacingLease
    {
        public const string EnvironmentFlagName = "WIS_CAMERA_FRAME_PACING";
        public const int TargetFrameRate = 60;

        static Func<int> read = DefaultRead;
        static Action<int> write = DefaultWrite;
        static Func<int> readVSync = DefaultReadVSync;
        static int active;
        static int previous;
        static bool hasPrevious;

        public static bool Enabled { get; set; } = ReadFlag();
        public static int ActiveLeases { get { return active; } }
        public static bool HasCapturedPrevious { get { return hasPrevious; } }
        public static int CapturedPreviousValue { get { return previous; } }

        static bool ReadFlag()
        {
            try
            {
                return string.Equals(Environment.GetEnvironmentVariable(EnvironmentFlagName),
                                     "1", StringComparison.Ordinal);
            }
            catch (Exception) { return false; }
        }

        static int DefaultRead() { return Application.targetFrameRate; }
        static void DefaultWrite(int value) { Application.targetFrameRate = value; }
        // read only: vSyncCount is reported for verification and is never written here
        static int DefaultReadVSync() { return QualitySettings.vSyncCount; }

        public static int CurrentValue { get { return read(); } }
        public static int CurrentVSyncCount { get { return readVSync(); } }

        /// <summary>
        /// The value this lease would apply over <paramref name="current"/>. An existing lower
        /// positive cap wins; uncapped or a higher cap is bounded to 60.
        /// </summary>
        public static int Bound(int current)
        {
            return current > 0 && current < TargetFrameRate ? current : TargetFrameRate;
        }

        /// <summary>
        /// Acquires for one holder. <paramref name="held"/> is the holder's own flag, so a
        /// second acquire by the same owner is a no-op and the count stays honest.
        /// </summary>
        public static void Acquire(ref bool held)
        {
            if (!Enabled || held) return;
            held = true;
            active++;
            if (active != 1) return;              // an overlapping owner already holds it
            previous = read();
            hasPrevious = true;
            int bounded = Bound(previous);
            if (bounded != previous) write(bounded);
        }

        /// <summary>
        /// Releases one holder. Idempotent, and a departing owner cannot clear a lease that a
        /// newly started owner is still holding: only the last release restores.
        /// </summary>
        public static void Release(ref bool held)
        {
            if (!held) return;
            held = false;
            if (active > 0) active--;
            if (active != 0 || !hasPrevious) return;
            hasPrevious = false;
            if (read() != previous) write(previous);
        }

        /// <summary>Test seam only: replaces the accessors and clears all lease state.</summary>
        public static void ConfigureForTests(bool enabled, Func<int> reader, Action<int> writer,
                                             Func<int> vSyncReader)
        {
            Enabled = enabled;
            read = reader ?? DefaultRead;
            write = writer ?? DefaultWrite;
            readVSync = vSyncReader ?? DefaultReadVSync;
            active = 0;
            hasPrevious = false;
            previous = 0;
        }

        public static void ResetToProductionDefaults()
        {
            ConfigureForTests(ReadFlag(), null, null, null);
        }
    }
}
