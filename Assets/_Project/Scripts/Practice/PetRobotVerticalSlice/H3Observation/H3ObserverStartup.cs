// M3_H3_OBSERVER_R5 - the activation. Prepared in the workspace. NOT installed.
//
// One runtime initialization method, no scene surgery, no polling, no gameplay-flow change,
// no MonoBehaviour and no object created.
//
// Behaviour, exactly three cases:
//   * flag absent            -> ENTIRELY INERT. Nothing loads, nothing installs, and not a
//                               single line is logged, so a normal build is untouched.
//   * flag present, valid    -> the bundle loads once, the sink is pointed at the default
//                               Unity log, the observer is installed, and one short READY
//                               line records the model provenance.
//   * flag present, invalid  -> ONE short typed setup-failure line and nothing else. No
//                               install, no scoring, no inference, and clearly
//                               distinguishable from the disabled case.
//
// Every fault here is diagnostic-only: the method never throws into engine startup.
using System;
using UnityEngine;

namespace WordsInSilence.Practice.PetRobotVerticalSlice.H3Observation
{
    public static class H3ObserverStartup
    {
        /// <summary>True once this run installed the observer. Diagnostics and tests.</summary>
        public static bool Activated { get; private set; }

        /// <summary>The single line this startup emitted, or empty. Tests read it.</summary>
        public static string LastSetupLine { get; private set; }

        /// <summary>
        /// Test-only sink for the SETUP line only. It never replaces the observer's own
        /// record sink, which is the default Unity log that Initialize installs.
        /// </summary>
        public static Action<string> SetupSinkForTest;

        public static void ResetForTest()
        {
            Activated = false;
            LastSetupLine = string.Empty;
            SetupSinkForTest = null;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        public static void Initialize()
        {
            try
            {
                H3BundleState state = H3ObservationBundle.State;

                // absent flag: return before any log, so a normal build sees nothing at all
                if (state == H3BundleState.FlagAbsent) return;

                if (state != H3BundleState.Ready)
                {
                    // one short typed line, carrying no flag value, path, message or stack
                    Report("[H3Observe] setup=UNAVAILABLE reason="
                           + H3RecordText.Token(H3ObservationBundle.ReasonCode, 64));
                    return;
                }

                H3AttemptObserver.Sink = LogSink;
                H3AttemptObserver.Install();
                Activated = true;
                Report("[H3Observe] setup=READY "
                       + H3RecordText.MetadataToken(H3ObservationBundle.Provenance, 200)
                       + " note=opt_in_diagnostic_only.no_quest_or_gate_or_enrollment_change"
                       + ".NO_PROMOTION");
            }
            catch (Exception error)
            {
                try
                {
                    Report("[H3Observe] setup=FAILED reason="
                           + H3RecordText.Token(error.GetType().Name, 48));
                }
                catch (Exception)
                {
                    // startup must never throw because a diagnostic could not report
                }
            }
        }

        static void Report(string line)
        {
            string bounded = H3RecordText.BoundLine(line, string.Empty);
            LastSetupLine = bounded;
            Action<string> sink = SetupSinkForTest;
            if (sink != null) { sink(bounded); return; }
            Debug.Log(bounded);
        }

        static void LogSink(string line)
        {
            Debug.Log(line);
        }
    }
}
