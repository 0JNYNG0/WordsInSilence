// F56 - opt-in runInBackground lease, held ONLY while a camera owner has a starting or
// playing camera. Robustness candidate, not a proven camera fix.
//
// Enabled only by WIS_CAMERA_CONTINUITY=1. It never opens, closes or reopens a camera, never
// runs per frame, and restores the previous value when the last owner releases.
using System;
using UnityEngine;

namespace WordsInSilence.Core
{
    public static class CameraBackgroundLease
    {
        public const string EnvironmentFlagName = "WIS_CAMERA_CONTINUITY";

        static Func<bool> read = DefaultRead;
        static Action<bool> write = DefaultWrite;
        static int active;
        static bool previous;
        static bool hasPrevious;

        public static bool Enabled { get; set; } = ReadFlag();
        public static int ActiveLeases { get { return active; } }
        public static bool HasCapturedPrevious { get { return hasPrevious; } }
        public static bool CapturedPreviousValue { get { return previous; } }

        static bool ReadFlag()
        {
            try
            {
                return string.Equals(Environment.GetEnvironmentVariable(EnvironmentFlagName),
                                     "1", StringComparison.Ordinal);
            }
            catch (Exception) { return false; }
        }

        static bool DefaultRead() { return Application.runInBackground; }
        static void DefaultWrite(bool value) { Application.runInBackground = value; }

        public static bool CurrentValue { get { return read(); } }

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
            write(true);
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
            write(previous);
        }

        /// <summary>Test seam only: replaces the accessors and clears all lease state.</summary>
        public static void ConfigureForTests(bool enabled, Func<bool> reader, Action<bool> writer)
        {
            Enabled = enabled;
            read = reader ?? DefaultRead;
            write = writer ?? DefaultWrite;
            active = 0;
            hasPrevious = false;
            previous = false;
        }

        public static void ResetToProductionDefaults()
        {
            ConfigureForTests(ReadFlag(), null, null);
        }
    }
}
