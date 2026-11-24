using System;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>
    /// Explicit live profile selection. The serialized zero value preserves the approved
    /// Step 9J/9K single-medoid route for every existing harness and prefab.
    /// </summary>
    public enum PersonalLiveProfileMode
    {
        LegacySingleMedoid = 0,
        MultiPrototypeCandidate = 1,
    }

    /// <summary>
    /// Step 9M candidate-only factory. It reuses the approved global loader and Step 9L
    /// session activation; it does not duplicate preprocessing, DTW, or threshold logic.
    /// </summary>
    public sealed class ApprovedPersonalMultiPrototypeLiveRuntimeFactory : IPersonalLiveRuntimeFactory
    {
        public const string ProfileDirectoryName = "step9l_current_user_multi_prototype_v1_0";

        readonly ApprovedPersonalLiveRuntimeFactory approvedLegacyFactory =
            new ApprovedPersonalLiveRuntimeFactory();

        public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths)
        {
            return approvedLegacyFactory.LoadGlobal(paths);
        }

        public IPersonalLiveRecognitionRuntime CreateAndActivate(
            IPersonalLiveGlobalRuntime globalRuntime,
            PersonalLiveRuntimePaths paths)
        {
            if (!(globalRuntime is ApprovedGlobalRuntimeHandle approved))
                throw new ArgumentException("approved global runtime handle is required", nameof(globalRuntime));
            if (paths == null) throw new ArgumentNullException(nameof(paths));

            var session = new PersonalRuntimeOptInSession(approved.Service);
            session.ActivatePersonalMultiPrototypeProfile(paths.PersonalProfileDirectory);
            return new ApprovedPersonalLiveRecognitionRuntime(session);
        }
    }
}
