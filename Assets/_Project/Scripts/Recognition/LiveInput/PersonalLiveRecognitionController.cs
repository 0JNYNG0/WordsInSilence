using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using WordsInSilence.Core;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.PhysicalCompatibility;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Recognition.LiveInput
{
    public enum PersonalLiveCaptureState
    {
        Idle,
        Ready,
        Capturing,
        Processing,
        Completed,
        Cancelled,
        Failed,
    }

    public enum PersonalLiveFailureCode
    {
        None,
        RuntimeInitializationFailure,
        PersonalProfileActivationFailure,
        BridgeMissing,
        BridgeNotBound,
        InvalidExpectedWord,
        InvalidStateTransition,
        EmptyCapture,
        ObservationRejected,
        NoValidNormalizedLandmarks,
        BridgeFailure,
        RecognitionFailure,
    }

    public sealed class PersonalLiveFailure
    {
        public PersonalLiveFailureCode Code { get; }
        public string Message { get; }
        public string ExceptionType { get; }

        internal PersonalLiveFailure(PersonalLiveFailureCode code, string message, Exception exception = null)
        {
            Code = code;
            Message = message ?? string.Empty;
            ExceptionType = exception == null ? string.Empty : exception.GetType().FullName;
        }
    }

    /// <summary>
    /// Immutable first-failure diagnostic captured before failure cleanup mutates live state.
    /// The snapshot remains pinned until a Reset completes successfully.
    /// </summary>
    public sealed class PersonalLivePrimaryFailureSnapshot
    {
        public PersonalLiveFailureCode Code { get; }
        public string Message { get; }
        public string ExceptionType { get; }
        public PersonalLiveCaptureState StateBeforeFailure { get; }
        public string CaptureSessionId { get; }
        public int BufferedRawFrameCount { get; }

        internal PersonalLivePrimaryFailureSnapshot(
            PersonalLiveFailureCode code,
            string message,
            Exception exception,
            PersonalLiveCaptureState stateBeforeFailure,
            string captureSessionId,
            int bufferedRawFrameCount)
        {
            Code = code;
            Message = message ?? string.Empty;
            ExceptionType = exception == null ? string.Empty : exception.GetType().FullName;
            StateBeforeFailure = stateBeforeFailure;
            CaptureSessionId = captureSessionId ?? string.Empty;
            BufferedRawFrameCount = bufferedRawFrameCount;
        }
    }

    /// <summary>
    /// Immutable diagnostic for a candidate-only, nonfatal ambiguous hand-identity drop.
    /// No observation is synthesized and no hand identity is inferred from this event.
    /// </summary>
    public sealed class PersonalLiveAmbiguityDiagnostic
    {
        public string Message { get; }
        public string ExceptionType { get; }
        public PersonalLiveCaptureState StateWhenRecorded { get; }
        public string CaptureSessionId { get; }
        public int BufferedRawFrameCount { get; }
        public bool DroppedDuringCapture { get; }

        internal PersonalLiveAmbiguityDiagnostic(
            Step9CAmbiguousHandIdentityException exception,
            PersonalLiveCaptureState stateWhenRecorded,
            string captureSessionId,
            int bufferedRawFrameCount,
            bool droppedDuringCapture)
        {
            Message = exception?.Message ?? string.Empty;
            ExceptionType = exception == null ? string.Empty : exception.GetType().FullName;
            StateWhenRecorded = stateWhenRecorded;
            CaptureSessionId = captureSessionId ?? string.Empty;
            BufferedRawFrameCount = bufferedRawFrameCount;
            DroppedDuringCapture = droppedDuringCapture;
        }
    }

    public sealed class PersonalLiveRuntimePaths
    {
        public string GlobalTemplateDirectory { get; }
        public string GlobalThresholdConfigPath { get; }
        public string PersonalProfileDirectory { get; }

        /// <summary>True only when the caller explicitly declared that this route has no personal
        /// profile directory at all, instead of inventing a fake model location.</summary>
        public bool AllowsNoPersonalProfileDirectory { get; }

        public PersonalLiveRuntimePaths(
            string globalTemplateDirectory,
            string globalThresholdConfigPath,
            string personalProfileDirectory,
            bool allowNoPersonalProfileDirectory = false)
        {
            if (string.IsNullOrWhiteSpace(globalTemplateDirectory))
                throw new ArgumentException("global template directory is required", nameof(globalTemplateDirectory));
            if (string.IsNullOrWhiteSpace(globalThresholdConfigPath))
                throw new ArgumentException("global threshold config path is required", nameof(globalThresholdConfigPath));
            if (!allowNoPersonalProfileDirectory && string.IsNullOrWhiteSpace(personalProfileDirectory))
                throw new ArgumentException("personal profile directory is required", nameof(personalProfileDirectory));
            GlobalTemplateDirectory = globalTemplateDirectory;
            GlobalThresholdConfigPath = globalThresholdConfigPath;
            PersonalProfileDirectory = personalProfileDirectory;
            AllowsNoPersonalProfileDirectory = allowNoPersonalProfileDirectory;
        }
    }

    public interface IPersonalLiveObservationSource
    {
        bool IsBound { get; }
        event Action<RawMediaPipeObservation> ObservationArrived;
        event Action<Exception> SourceFailed;
    }

    /// <summary>
    /// Explicit adapter over the approved Step 9C bridge. It never searches for a bridge,
    /// starts a provider, stops a provider, or owns a webcam.
    /// </summary>
    public sealed class Step9CObservationSourceAdapter : IPersonalLiveObservationSource, IDisposable
    {
        readonly Step9CHandIdentityObservationBridge bridge;
        Action<RawMediaPipeObservation> observationHandlers;
        Action<Exception> failureHandlers;
        bool attached;
        bool disposed;

        public bool IsBound => !disposed && bridge != null && bridge.IsBound;
        public int SubscriberCount =>
            (observationHandlers == null ? 0 : observationHandlers.GetInvocationList().Length) +
            (failureHandlers == null ? 0 : failureHandlers.GetInvocationList().Length);

        public Step9CObservationSourceAdapter(Step9CHandIdentityObservationBridge bridge)
        {
            this.bridge = bridge ?? throw new ArgumentNullException(nameof(bridge));
        }

        public event Action<RawMediaPipeObservation> ObservationArrived
        {
            add
            {
                ThrowIfDisposed();
                observationHandlers += value;
                UpdateAttachment();
            }
            remove
            {
                observationHandlers -= value;
                UpdateAttachment();
            }
        }

        public event Action<Exception> SourceFailed
        {
            add
            {
                ThrowIfDisposed();
                failureHandlers += value;
                UpdateAttachment();
            }
            remove
            {
                failureHandlers -= value;
                UpdateAttachment();
            }
        }

        public void Dispose()
        {
            if (disposed) return;
            observationHandlers = null;
            failureHandlers = null;
            Detach();
            disposed = true;
        }

        void UpdateAttachment()
        {
            if (disposed) return;
            bool needsAttachment = observationHandlers != null || failureHandlers != null;
            if (needsAttachment && !attached)
            {
                bridge.ObservationArrived += ForwardObservation;
                bridge.BridgeFailed += ForwardFailure;
                attached = true;
            }
            else if (!needsAttachment && attached)
            {
                Detach();
            }
        }

        void Detach()
        {
            if (!attached) return;
            bridge.ObservationArrived -= ForwardObservation;
            bridge.BridgeFailed -= ForwardFailure;
            attached = false;
        }

        void ForwardObservation(RawMediaPipeObservation observation) => observationHandlers?.Invoke(observation);
        void ForwardFailure(Exception exception) => failureHandlers?.Invoke(exception);

        void ThrowIfDisposed()
        {
            if (disposed) throw new ObjectDisposedException(nameof(Step9CObservationSourceAdapter));
        }
    }

    public interface IPersonalLiveGlobalRuntime
    {
    }

    public interface IPersonalLiveRecognitionRuntime
    {
        bool IsPersonalActive { get; }
        string ActiveProfileId { get; }
        PersonalLiveRoutingResult Recognize(string sessionId, RecognitionSequence sequence, string expectedWord);
    }

    public interface IPersonalLiveRuntimeFactory
    {
        IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths);
        IPersonalLiveRecognitionRuntime CreateAndActivate(
            IPersonalLiveGlobalRuntime globalRuntime,
            PersonalLiveRuntimePaths paths);
    }

    sealed class ApprovedGlobalRuntimeHandle : IPersonalLiveGlobalRuntime
    {
        public RuntimeRecognitionService Service { get; }

        public ApprovedGlobalRuntimeHandle(RuntimeRecognitionService service)
        {
            Service = service ?? throw new ArgumentNullException(nameof(service));
        }
    }

    /// <summary>Production factory retaining the approved global loader and explicit Step 9I opt-in.</summary>
    public sealed class ApprovedPersonalLiveRuntimeFactory : IPersonalLiveRuntimeFactory
    {
        public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths)
        {
            if (paths == null) throw new ArgumentNullException(nameof(paths));
            return new ApprovedGlobalRuntimeHandle(RuntimeRecognitionService.Load(
                paths.GlobalTemplateDirectory,
                paths.GlobalThresholdConfigPath));
        }

        public IPersonalLiveRecognitionRuntime CreateAndActivate(
            IPersonalLiveGlobalRuntime globalRuntime,
            PersonalLiveRuntimePaths paths)
        {
            if (!(globalRuntime is ApprovedGlobalRuntimeHandle approved))
                throw new ArgumentException("approved global runtime handle is required", nameof(globalRuntime));
            if (paths == null) throw new ArgumentNullException(nameof(paths));
            var session = new PersonalRuntimeOptInSession(approved.Service);
            session.ActivatePersonalProfile(paths.PersonalProfileDirectory);
            return new ApprovedPersonalLiveRecognitionRuntime(session);
        }
    }

    /// <summary>Recognition routing seam; the production implementation delegates only to Step 9I.</summary>
    public sealed class ApprovedPersonalLiveRecognitionRuntime : IPersonalLiveRecognitionRuntime
    {
        readonly PersonalRuntimeOptInSession session;

        public bool IsPersonalActive => session.IsPersonalActive;
        public string ActiveProfileId => session.ActiveProfileId;

        public ApprovedPersonalLiveRecognitionRuntime(PersonalRuntimeOptInSession session)
        {
            this.session = session ?? throw new ArgumentNullException(nameof(session));
            if (!session.IsPersonalActive || session.Mode != RuntimeOptInMode.Personal)
                throw new ArgumentException("personal runtime session must be explicitly active", nameof(session));
        }

        public PersonalLiveRoutingResult Recognize(
            string sessionId,
            RecognitionSequence sequence,
            string expectedWord)
        {
            RuntimeOptInRecognitionResult routed = session.Recognize(sessionId, sequence, expectedWord);
            if (routed.Mode != RuntimeOptInMode.Personal || routed.PersonalResult == null)
                throw new InvalidOperationException("Step 9I personal mode was not used");
            PersonalRuntimeRecognitionResult result = routed.PersonalResult;
            return new PersonalLiveRoutingResult(
                result.PredictedWord,
                result.Accepted,
                result.ExpectedWordClassAccept,
                result.QuestSuccess,
                result.Scores,
                result.PerClassAccept,
                result.BestDistance,
                result.WinningPrototypeId,
                result.Warnings);
        }
    }

    public sealed class PersonalLiveRoutingResult
    {
        readonly IReadOnlyDictionary<string, double> scores;
        readonly IReadOnlyDictionary<string, bool> perClassAccept;

        public string PredictedWord { get; }
        public bool Accepted { get; }
        public bool ExpectedWordClassAccept { get; }
        public bool QuestSuccess { get; }
        public IReadOnlyDictionary<string, double> Scores => scores;
        public IReadOnlyDictionary<string, bool> PerClassAccept => perClassAccept;
        public double BestDistance { get; }
        public string WinningPrototypeId { get; }
        public IReadOnlyList<string> Warnings { get; }

        public PersonalLiveRoutingResult(
            string predictedWord,
            bool accepted,
            bool expectedWordClassAccept,
            bool questSuccess,
            IReadOnlyDictionary<string, double> scores,
            IReadOnlyDictionary<string, bool> perClassAccept,
            double bestDistance,
            string winningPrototypeId,
            IEnumerable<string> warnings)
        {
            if (!PersonalLiveRecognitionController.IsSupportedWord(predictedWord))
                throw new ArgumentException("predicted word is unsupported", nameof(predictedWord));
            if (scores == null) throw new ArgumentNullException(nameof(scores));
            if (perClassAccept == null) throw new ArgumentNullException(nameof(perClassAccept));
            if (scores.Count != 3 || perClassAccept.Count != 3 ||
                PersonalLiveRecognitionController.SupportedWords.Any(word =>
                    !scores.ContainsKey(word) || !perClassAccept.ContainsKey(word)))
                throw new ArgumentException("personal routing result must contain exactly three word decisions");
            if (scores.Values.Any(value => double.IsNaN(value) || double.IsInfinity(value) || value < 0.0))
                throw new ArgumentException("personal routing score is invalid", nameof(scores));
            if (double.IsNaN(bestDistance) || double.IsInfinity(bestDistance) || bestDistance < 0.0)
                throw new ArgumentOutOfRangeException(nameof(bestDistance));
            if (string.IsNullOrWhiteSpace(winningPrototypeId))
                throw new ArgumentException("winning prototype ID is required", nameof(winningPrototypeId));
            PredictedWord = predictedWord;
            Accepted = accepted;
            ExpectedWordClassAccept = expectedWordClassAccept;
            QuestSuccess = questSuccess;
            this.scores = new ReadOnlyDictionary<string, double>(
                new Dictionary<string, double>(scores, StringComparer.Ordinal));
            this.perClassAccept = new ReadOnlyDictionary<string, bool>(
                new Dictionary<string, bool>(perClassAccept, StringComparer.Ordinal));
            BestDistance = bestDistance;
            WinningPrototypeId = winningPrototypeId;
            Warnings = Array.AsReadOnly((warnings ?? Array.Empty<string>()).ToArray());
        }
    }

    /// <summary>Immutable terminal snapshot displayed by the Step 9J technical harness.</summary>
    public sealed class PersonalLiveRecognitionSnapshot
    {
        public string CaptureSessionId { get; }
        public PersonalLiveCaptureState CaptureState => PersonalLiveCaptureState.Completed;
        public string ExpectedWord { get; }
        public string PredictedWord { get; }
        public bool Accepted { get; }
        public bool ExpectedWordClassAccept { get; }
        public bool QuestSuccess { get; }
        public double GreetingDistance { get; }
        public double LoveDistance { get; }
        public double ThankYouDistance { get; }
        public double BestDistance { get; }
        public string WinningPrototypeId { get; }
        public int RawFrameCount { get; }
        public int NormalizedFrameCount { get; }
        public int ValidLandmarkCount { get; }
        public int InterpolatedLandmarkCount { get; }
        public string PersonalProfileId { get; }
        public bool PersonalModeActive { get; }
        public bool GreetingMarginNarrow { get; }
        public RecognitionSequence NormalizedSequence { get; }

        /// <summary>
        /// Active common-2D capture for the SAME finalized segment: raw/post-gap x/y
        /// restored to the actual pixel plane with z=0 and passed through the approved
        /// torso_normalize_2d contract. This is the only permitted input for the frozen
        /// common model. <see cref="NormalizedSequence"/> is the pseudo-3D strict/personal
        /// payload and must never be substituted for it. Null when the common-2D domain
        /// could not be built, with the reason in <see cref="Common2DFailure"/>.
        /// </summary>
        public CommonTwoDimensionalCapture Common2DCapture { get; }

        public string Common2DFailure { get; }

        internal PersonalLiveRecognitionSnapshot(
            string captureSessionId,
            string expectedWord,
            ApprovedPreprocessingResult preprocessing,
            RecognitionSequence normalizedSequence,
            PersonalLiveRoutingResult recognition,
            string profileId,
            bool personalModeActive,
            CommonTwoDimensionalCapture common2DCapture = null,
            string common2DFailure = null)
        {
            Common2DCapture = common2DCapture;
            Common2DFailure = common2DFailure ?? string.Empty;
            CaptureSessionId = captureSessionId;
            ExpectedWord = expectedWord;
            PredictedWord = recognition.PredictedWord;
            Accepted = recognition.Accepted;
            ExpectedWordClassAccept = recognition.ExpectedWordClassAccept;
            QuestSuccess = recognition.QuestSuccess;
            GreetingDistance = recognition.Scores["GREETING"];
            LoveDistance = recognition.Scores["LOVE"];
            ThankYouDistance = recognition.Scores["THANK_YOU"];
            BestDistance = recognition.BestDistance;
            WinningPrototypeId = recognition.WinningPrototypeId;
            RawFrameCount = preprocessing.RawFrameCount;
            NormalizedFrameCount = preprocessing.NormalizedFrameCount;
            ValidLandmarkCount = preprocessing.ValidLandmarkCount;
            InterpolatedLandmarkCount = preprocessing.InterpolatedLandmarkCount;
            PersonalProfileId = profileId;
            PersonalModeActive = personalModeActive;
            NormalizedSequence = normalizedSequence ?? throw new ArgumentNullException(nameof(normalizedSequence));
            GreetingMarginNarrow = recognition.Warnings.Contains(
                PersonalRuntimeProfileLoader.GreetingMarginWarning,
                StringComparer.Ordinal);
        }
    }

    /// <summary>
    /// Manual single-take state machine connecting the approved Step 9C observation bridge,
    /// Step 9B preprocessing, sequence-input contract, and Step 9I personal evaluator.
    /// </summary>
    public sealed class PersonalLiveRecognitionController : IDisposable
    {
        static readonly string[] supportedWords = { "GREETING", "LOVE", "THANK_YOU" };

        readonly IPersonalLiveObservationSource observationSource;
        readonly IPersonalLiveRuntimeFactory runtimeFactory;
        readonly ApprovedLivePreprocessingAdapter preprocessing;
        readonly bool candidateTransientAmbiguityToleranceActive;
        readonly bool deferredBridgeBindingAllowed;
        readonly bool allowDemoParticipantRuntime;
        readonly RecognitionSequenceInputSession sequenceInput = new RecognitionSequenceInputSession();
        readonly List<RawMediaPipeObservation> observations = new List<RawMediaPipeObservation>();
        /// <summary>F21 opt-in seam. Null keeps the legacy path exactly as it was.</summary>
        readonly WordsInSilence.Recognition.FullHandRuntime.IFullHandGameCaptureProcessor fullHandProcessor;

        /// <summary>
        /// F27 opt-in, fixed at construction. When true this capture route finalizes on the
        /// full-hand evidence and does NOT run the legacy 16-point preprocessing, sequence
        /// construction, personal recognizer or common-2D build. False keeps every existing
        /// behaviour byte-for-byte.
        /// </summary>
        readonly bool fullHandGameplayEnabled;

        /// <summary>True only between a successful full-hand finalize and the next finalize
        /// attempt, cancel, reset or new capture.</summary>
        bool fullHandGameplayCaptureFinalized;

        IPersonalLiveRecognitionRuntime runtime;
        RawObservationConvention captureConvention;
        double lastTimestamp;
        RawMediaPipeObservation primedTerminalObservation;
        bool suppressPrimedTerminalEcho;
        string captureSessionId = string.Empty;
        bool subscribed;
        bool disposed;

        public static IReadOnlyList<string> SupportedWords => Array.AsReadOnly((string[])supportedWords.Clone());
        public PersonalLiveCaptureState State { get; private set; } = PersonalLiveCaptureState.Idle;
        public string ExpectedWord { get; private set; } = "GREETING";
        public string CaptureSessionId => captureSessionId;
        public int BufferedRawFrameCount => observations.Count;
        public bool IsObservationSubscribed => subscribed;
        public bool IsPersonalModeActive => runtime != null && runtime.IsPersonalActive;
        public string ActivePersonalProfileId => runtime == null ? string.Empty : runtime.ActiveProfileId;

        /// <summary>The demo-participant marker, and only when the caller opted in explicitly.</summary>
        IDemoParticipantLiveRuntime DemoRuntime =>
            allowDemoParticipantRuntime ? runtime as IDemoParticipantLiveRuntime : null;

        public bool IsDemoParticipantRuntime => DemoRuntime != null;

        /// <summary>An explicitly marked, opted-in demo runtime answers for itself (so a stale one
        /// is not ready even if it is personally active); otherwise the legacy strict rule applies.</summary>
        public bool IsRuntimeReady =>
            DemoRuntime != null
                ? DemoRuntime.IsReady
                : runtime != null && runtime.IsPersonalActive && !string.IsNullOrWhiteSpace(runtime.ActiveProfileId);

        /// <summary>Honest provenance of whatever actually produced the decision.</summary>
        public string RuntimeProvenanceId =>
            DemoRuntime != null ? DemoRuntime.RuntimeIdentity : ActivePersonalProfileId;

        public IReadOnlyDictionary<string, double> DemoRuntimeThresholds =>
            DemoRuntime == null ? null : DemoRuntime.Thresholds;
        public PersonalLiveRecognitionSnapshot Result { get; private set; }
        public PersonalLiveFailure Failure { get; private set; }

        /// <summary>
        /// Typed research evidence from the optional F21 full-hand seam for the most recent
        /// capture, or null when no processor is configured or the lifecycle cleared it. It
        /// is never a gameplay decision: the strict result above remains the only authority.
        /// It is deliberately NOT cleared by a failure, so a completed bridge result survives
        /// a later legacy preprocessing or runtime failure of the same capture.
        /// </summary>
        public WordsInSilence.Recognition.FullHandRuntime.FullHandGameCaptureResult
            LastFullHandResult { get; private set; }

        /// <summary>F27. The mode this controller was constructed with. Never re-read.</summary>
        public bool FullHandGameplayEnabled { get { return fullHandGameplayEnabled; } }

        /// <summary>Whether a full-hand processor is actually attached to this controller.</summary>
        public bool HasFullHandGameCaptureProcessor { get { return fullHandProcessor != null; } }

        /// <summary>
        /// F27. True only when THIS capture finalized on the full-hand route. Any later
        /// finalize attempt clears it first, so an invalid finalize can never re-use the
        /// still-cached evidence of the previous capture as a new success.
        /// </summary>
        public bool FullHandGameplayCaptureFinalized
        {
            get { return fullHandGameplayCaptureFinalized; }
        }
        public PersonalLivePrimaryFailureSnapshot PrimaryFailure { get; private set; }
        public bool CandidateTransientAmbiguityToleranceActive =>
            candidateTransientAmbiguityToleranceActive;
        public int ToleratedAmbiguousHandIdentityCount { get; private set; }
        public PersonalLiveAmbiguityDiagnostic LastToleratedAmbiguousHandIdentity { get; private set; }

        public PersonalLiveRecognitionController(
            IPersonalLiveObservationSource observationSource,
            IPersonalLiveRuntimeFactory runtimeFactory = null,
            ApprovedLivePreprocessingAdapter preprocessing = null,
            bool enableCandidateTransientAmbiguityTolerance = false,
            bool allowDeferredBridgeBinding = false,
            bool allowDemoParticipantRuntime = false,
            WordsInSilence.Recognition.FullHandRuntime.IFullHandGameCaptureProcessor fullHandProcessor = null,
            bool fullHandGameplayEnabled = false)
        {
            this.fullHandGameplayEnabled = fullHandGameplayEnabled;
            this.fullHandProcessor = fullHandProcessor;
            this.observationSource = observationSource;
            this.runtimeFactory = runtimeFactory ?? new ApprovedPersonalLiveRuntimeFactory();
            this.preprocessing = preprocessing ?? new ApprovedLivePreprocessingAdapter();
            candidateTransientAmbiguityToleranceActive = enableCandidateTransientAmbiguityTolerance;
            deferredBridgeBindingAllowed = allowDeferredBridgeBinding;
            this.allowDemoParticipantRuntime = allowDemoParticipantRuntime;
        }

        public bool Initialize(PersonalLiveRuntimePaths paths)
        {
            ThrowIfDisposed();
            if (State != PersonalLiveCaptureState.Idle)
                return Fail(PersonalLiveFailureCode.InvalidStateTransition, $"initialize is invalid in state {State}");
            if (observationSource == null)
                return Fail(PersonalLiveFailureCode.BridgeMissing, "Step 9C observation bridge is required");
            if (!observationSource.IsBound && !deferredBridgeBindingAllowed)
                return Fail(PersonalLiveFailureCode.BridgeNotBound, "Step 9C observation bridge is not bound");
            if (paths == null)
                return Fail(PersonalLiveFailureCode.RuntimeInitializationFailure, "runtime paths are required");

#if UNITY_EDITOR
            // Editor-only: initialize starts from an empty trace buffer.
            WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer.Clear();
#endif

            IPersonalLiveGlobalRuntime global;
            try
            {
                global = runtimeFactory.LoadGlobal(paths);
            }
            catch (Exception exception)
            {
                return Fail(PersonalLiveFailureCode.RuntimeInitializationFailure,
                    "global runtime initialization failed: " + exception.Message, exception);
            }

            try
            {
                runtime = runtimeFactory.CreateAndActivate(global, paths);
                if (runtime == null)
                    throw new InvalidOperationException("personal profile was not explicitly activated");
                if (!IsRuntimeReady)
                    throw new InvalidOperationException(allowDemoParticipantRuntime
                        ? "no explicitly marked ready demo participant runtime was returned"
                        : "personal profile was not explicitly activated");
            }
            catch (Exception exception)
            {
                runtime = null;
                return Fail(PersonalLiveFailureCode.PersonalProfileActivationFailure,
                    "personal profile activation failed: " + exception.Message, exception);
            }

            Result = null;
            Failure = null;
            LastFullHandResult = null;
            ResetCaptureAmbiguityTolerance();
            State = PersonalLiveCaptureState.Ready;
            Subscribe();
            return true;
        }

        public bool SetExpectedWord(string expectedWord)
        {
            ThrowIfDisposed();
            if (State != PersonalLiveCaptureState.Ready)
                return Fail(PersonalLiveFailureCode.InvalidStateTransition, $"set-expected-word is invalid in state {State}");
            if (!IsSupportedWord(expectedWord))
                return Fail(PersonalLiveFailureCode.InvalidExpectedWord, "expected word must be GREETING, LOVE, or THANK_YOU");
            ExpectedWord = expectedWord;
            Failure = null;
            return true;
        }

        public bool BeginCapture(string requestedSessionId = null)
        {
            return BeginCaptureCore(requestedSessionId, null);
        }

        public bool BeginCaptureWithPreRoll(
            IReadOnlyList<RawMediaPipeObservation> preRoll,
            string requestedSessionId = null)
        {
            ThrowIfDisposed();
            List<RawMediaPipeObservation> validated;
            try
            {
                validated = ValidatePreRoll(preRoll);
            }
            catch (Exception exception)
            {
                return Fail(PersonalLiveFailureCode.ObservationRejected,
                    "raw pre-roll validation failed: " + exception.Message, exception);
            }
            return BeginCaptureCore(requestedSessionId, validated);
        }

        bool BeginCaptureCore(
            string requestedSessionId,
            IReadOnlyList<RawMediaPipeObservation> validatedPreRoll)
        {
            ThrowIfDisposed();
            if (State != PersonalLiveCaptureState.Ready)
                return Fail(PersonalLiveFailureCode.InvalidStateTransition, $"begin is invalid in state {State}");
            if (observationSource == null)
                return Fail(PersonalLiveFailureCode.BridgeMissing, "Step 9C observation bridge is required");
            if (!observationSource.IsBound)
                return Fail(PersonalLiveFailureCode.BridgeNotBound, "Step 9C observation bridge is not bound");
            if (!IsSupportedWord(ExpectedWord))
                return Fail(PersonalLiveFailureCode.InvalidExpectedWord, "expected word is invalid");
            if (!IsRuntimeReady)
                return Fail(PersonalLiveFailureCode.PersonalProfileActivationFailure, "recognition runtime is not ready");

            captureSessionId = string.IsNullOrWhiteSpace(requestedSessionId)
                ? Guid.NewGuid().ToString("D")
                : requestedSessionId;
            observations.Clear();
            captureConvention = null;
            lastTimestamp = double.NaN;
            ResetPrimedTerminalEcho();
            sequenceInput.Reset();
            ResetCaptureAmbiguityTolerance();
#if UNITY_EDITOR
            // Editor-only: no field of a previous capture may survive into this one.
            WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer.Clear();
#endif
            Result = null;
            Failure = null;
            LastFullHandResult = null;
            Subscribe();
            State = PersonalLiveCaptureState.Capturing;
            if (validatedPreRoll != null)
            {
                foreach (RawMediaPipeObservation observation in validatedPreRoll)
                    AppendValidatedObservation(observation);
                primedTerminalObservation = validatedPreRoll[validatedPreRoll.Count - 1];
                suppressPrimedTerminalEcho = true;
            }
            return true;
        }

        public bool FinalizeCapture()
        {
            return FinalizeCaptureCore(0d);
        }

        public bool FinalizeCaptureWithTrailingTrim(double trailingTrimSeconds)
        {
            return FinalizeCaptureCore(trailingTrimSeconds);
        }

        bool FinalizeCaptureCore(double trailingTrimSeconds)
        {
            ThrowIfDisposed();
            // F27: any finalize attempt invalidates the previous capture's full-hand
            // completion before anything else is checked.
            fullHandGameplayCaptureFinalized = false;
            if (double.IsNaN(trailingTrimSeconds) || double.IsInfinity(trailingTrimSeconds) ||
                trailingTrimSeconds < 0d)
                return Fail(PersonalLiveFailureCode.ObservationRejected,
                    "trailing trim must be finite and non-negative");
            if (State != PersonalLiveCaptureState.Capturing)
                return Fail(PersonalLiveFailureCode.InvalidStateTransition, $"finalize is invalid in state {State}");
            if (observations.Count == 0)
                return Fail(PersonalLiveFailureCode.EmptyCapture, "cannot finalize a capture with zero observations");

            IReadOnlyList<RawMediaPipeObservation> recognitionObservations = observations;
            if (trailingTrimSeconds > 0d)
            {
                double cutoff = observations[observations.Count - 1].TimestampSeconds - trailingTrimSeconds;
                var trimmed = new List<RawMediaPipeObservation>();
                foreach (RawMediaPipeObservation observation in observations)
                    if (observation.TimestampSeconds <= cutoff) trimmed.Add(observation);
                if (trimmed.Count == 0)
                    return Fail(PersonalLiveFailureCode.EmptyCapture,
                        "trailing-neutral trim removed the entire capture");
                recognitionObservations = trimmed;
            }

            State = PersonalLiveCaptureState.Processing;
            Unsubscribe();
            // F5 opt-in, read-only observation of this ALREADY TRIMMED segment, placed before
            // the old preprocessing so an attempt is recorded even when the old path then
            // fails. Off unless deliberately enabled; every observation fault is handled
            // inside the observer, so no state, control flow or result below changes.
            if (fullHandProcessor != null)
            {
                // F21 opt-in: the SAME finalized, already trailing-trimmed segment reaches
                // the shared full-hand pipeline exactly once, at exactly this point. When a
                // processor is configured the older F5 observer does NOT also run for this
                // capture. Every fault is contained inside SafeProcess, so no state, control
                // flow, timing or result below changes, and the evidence is never authority.
                LastFullHandResult = WordsInSilence.Recognition.FullHandRuntime
                    .FullHandGameCaptureBridge.SafeProcess(fullHandProcessor, captureSessionId,
                                                           ExpectedWord, recognitionObservations);
            }
            else
            {
                WordsInSilence.Recognition.FullHandRuntime.FullHandAttemptObserver.SafeObserve(
                    captureSessionId, ExpectedWord, recognitionObservations);
            }

            if (fullHandGameplayEnabled)
            {
                // F27. The full-hand evidence above is the authority for this capture, so
                // the legacy preprocessing, sequence construction, personal recognizer and
                // common-2D build below are deliberately skipped: nothing fabricates a
                // legacy result or a 16-point sequence here, and a legacy failure can no
                // longer veto a completed full-hand capture. When no processor is attached
                // LastFullHandResult simply stays null and the gameplay seam refuses
                // explicitly rather than falling back to the legacy route.
                observations.Clear();
                captureConvention = null;
                lastTimestamp = double.NaN;
                ResetPrimedTerminalEcho();
                Result = null;
                Failure = null;
                fullHandGameplayCaptureFinalized = true;
                State = PersonalLiveCaptureState.Completed;
                return true;
            }

            ApprovedPreprocessingResult adapted;
            try
            {
                adapted = preprocessing.Adapt(recognitionObservations);
            }
            catch (Exception exception)
            {
                return Fail(PersonalLiveFailureCode.ObservationRejected,
                    "approved preprocessing rejected the observation capture: " + exception.Message, exception);
            }
            if (adapted.ValidLandmarkCount == 0)
                return Fail(PersonalLiveFailureCode.NoValidNormalizedLandmarks,
                    "capture contains no valid normalized landmarks");

            RecognitionSequence sequence;
            try
            {
                sequenceInput.Reset();
                sequenceInput.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled(captureSessionId));
                foreach (NormalizedRecognitionFrame frame in adapted.Frames)
                    sequenceInput.AddFrame(captureSessionId, frame);
                sequence = sequenceInput.FinalizeSequence(captureSessionId);
            }
            catch (Exception exception)
            {
                return Fail(PersonalLiveFailureCode.RecognitionFailure,
                    "recognition sequence construction failed: " + exception.Message, exception);
            }

            PersonalLiveRoutingResult recognition;
            try
            {
                recognition = runtime.Recognize(captureSessionId, sequence, ExpectedWord);
            }
            catch (Exception exception)
            {
                return Fail(PersonalLiveFailureCode.RecognitionFailure,
                    "personal recognition failed: " + exception.Message, exception);
            }

            // Active common-2D domain for the SAME finalized segment, built from the
            // retained raw/post-gap observations before they are released. Never derived
            // from the pseudo-3D sequence above. A failure here is recorded as typed
            // evidence and never alters the strict result.
            //
            // Out-of-contract targets (the legacy LOVE route) skip the common-2D build
            // entirely: no capture, no feature vector and no frozen forward pass.
            string common2DFailure = null;
            CommonTwoDimensionalCapture common2DCapture = null;
            if (WordsInSilence.Recognition.Common.FrozenCommonSignModel
                    .IsSupportedTarget(ExpectedWord))
            {
#if UNITY_EDITOR
                // Editor-only: bind the trace to THIS capture before the common-2D build
                // consumes the finalized, already-trimmed observation sequence.
                WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                    .SafeBeginSession(captureSessionId, ExpectedWord,
                                      recognitionObservations.Count);
#endif
                common2DCapture = ApprovedLivePreprocessingAdapter.AdaptCommon2D(
                    recognitionObservations, out common2DFailure);
            }

            Result = new PersonalLiveRecognitionSnapshot(
                captureSessionId,
                ExpectedWord,
                adapted,
                sequence,
                recognition,
                runtime.ActiveProfileId,
                runtime.IsPersonalActive,
                common2DCapture,
                common2DFailure);
            observations.Clear();
            captureConvention = null;
            lastTimestamp = double.NaN;
            ResetPrimedTerminalEcho();
            Failure = null;
            State = PersonalLiveCaptureState.Completed;
            return true;
        }

        public bool CancelCapture()
        {
            ThrowIfDisposed();
#if UNITY_EDITOR
            // Editor-only: cancelling a capture drops its incomplete diagnostic trace.
            // Ownership, subscription and every existing capture semantic are untouched,
            // and this never enables diagnostics.
            WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                .CancelIncompleteTrace();
#endif
            if (State != PersonalLiveCaptureState.Capturing && State != PersonalLiveCaptureState.Processing)
                return Fail(PersonalLiveFailureCode.InvalidStateTransition, $"cancel is invalid in state {State}");
            observations.Clear();
            sequenceInput.Reset();
            captureConvention = null;
            lastTimestamp = double.NaN;
            ResetPrimedTerminalEcho();
            Result = null;
            Failure = null;
            LastFullHandResult = null;
            State = PersonalLiveCaptureState.Cancelled;
            Unsubscribe();
            return true;
        }

        /// <summary>
        /// Establishes the authoritative typed failure for a source event that an owning
        /// coordinator has already classified as fatal during an active capture. This seam
        /// is idempotent so source-subscriber invocation order cannot replace the first
        /// failure snapshot.
        /// </summary>
        public bool FailActiveCaptureFromSource(Exception exception)
        {
            ThrowIfDisposed();
            if (State == PersonalLiveCaptureState.Failed) return true;
            if (State != PersonalLiveCaptureState.Capturing) return false;

            Fail(PersonalLiveFailureCode.BridgeFailure,
                "Step 9C observation bridge failed: " +
                (exception?.Message ?? "unknown bridge failure"),
                exception);
            return State == PersonalLiveCaptureState.Failed;
        }

        public bool Reset()
        {
            ThrowIfDisposed();
            if (State != PersonalLiveCaptureState.Completed &&
                State != PersonalLiveCaptureState.Cancelled &&
                State != PersonalLiveCaptureState.Failed)
                return Fail(PersonalLiveFailureCode.InvalidStateTransition, $"reset is invalid in state {State}");
            if (observationSource == null)
                return Fail(PersonalLiveFailureCode.BridgeMissing, "Step 9C observation bridge is required");
            if (!observationSource.IsBound)
                return Fail(PersonalLiveFailureCode.BridgeNotBound, "Step 9C observation bridge is not bound");
            if (!IsRuntimeReady)
                return Fail(PersonalLiveFailureCode.PersonalProfileActivationFailure, "recognition runtime is not ready");

            observations.Clear();
            sequenceInput.Reset();
            captureConvention = null;
            lastTimestamp = double.NaN;
            ResetPrimedTerminalEcho();
            captureSessionId = string.Empty;
            Result = null;
            Failure = null;
            LastFullHandResult = null;
            PrimaryFailure = null;
            ResetCaptureAmbiguityTolerance();
#if UNITY_EDITOR
            // Editor-only: reset drops any uncommitted trace before a new observation.
            WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer.Clear();
#endif
            State = PersonalLiveCaptureState.Ready;
            Subscribe();
            return true;
        }

        public void Disable()
        {
            if (disposed) return;
#if UNITY_EDITOR
            // Editor-only: disabling the controller drops any incomplete diagnostic
            // trace. Ownership, subscription semantics and every existing capture
            // behaviour are untouched, and this never enables diagnostics.
            WordsInSilence.Recognition.Diagnostics.Common2DDiagnosticTraceBuffer
                .CancelIncompleteTrace();
#endif
            Unsubscribe();
            observations.Clear();
            sequenceInput.Reset();
            captureConvention = null;
            lastTimestamp = double.NaN;
            ResetPrimedTerminalEcho();
            if (State == PersonalLiveCaptureState.Capturing ||
                State == PersonalLiveCaptureState.Processing ||
                State == PersonalLiveCaptureState.Ready)
            {
                Result = null;
                Failure = null;
                LastFullHandResult = null;
                State = PersonalLiveCaptureState.Cancelled;
            }
        }

        public void Dispose()
        {
            if (disposed) return;
            Disable();
            disposed = true;
        }

        public static bool IsSupportedWord(string word) =>
            supportedWords.Contains(word, StringComparer.Ordinal);

        void OnObservation(RawMediaPipeObservation observation)
        {
            if (State != PersonalLiveCaptureState.Capturing) return;
            if (observation == null)
            {
                Fail(PersonalLiveFailureCode.ObservationRejected, "bridge emitted a null observation");
                return;
            }
            if (suppressPrimedTerminalEcho)
            {
                bool isEcho = ReferenceEquals(observation, primedTerminalObservation);
                ResetPrimedTerminalEcho();
                if (isEcho) return;
            }
            string rejection;
            if (!TryAppendObservation(observation, out rejection))
                Fail(PersonalLiveFailureCode.ObservationRejected, rejection);
        }

        /// <summary>
        /// The only observation conventions this capture route accepts.
        ///
        /// Either the unchanged approved live convention, or the F51 upright convention and
        /// ONLY when the observation's own recorded geometry backs it: both provider
        /// geometries available and fully contract compatible, both declaring the applied
        /// pre-inference vertical flip with top-down rows, and both reporting an actual
        /// camera geometry of rotation 0 with no vertical mirror. Unknown rotation or mirror
        /// metadata is refused rather than assumed. No other normalized-image convention is
        /// accepted, and mixing the two inside one capture stays blocked by the existing
        /// capture-convention stability checks below.
        /// </summary>
        // F53: the rule itself now lives in LiveObservationConventionPolicy so the ring
        // buffer and the neutral auto segmenter apply the identical predicate. This method
        // is kept as a thin delegation: the behaviour and the reflection tests are unchanged.
        static bool IsAcceptedCaptureConvention(RawMediaPipeObservation observation)
        {
            return LiveObservationConventionPolicy.IsAccepted(observation);
        }

        static bool IsSupportedActualCameraGeometry(Step9DInferenceFrameGeometry geometry)
        {
            return LiveObservationConventionPolicy.IsSupportedActualCameraGeometry(geometry);
        }

        List<RawMediaPipeObservation> ValidatePreRoll(IReadOnlyList<RawMediaPipeObservation> preRoll)
        {
            if (preRoll == null) throw new ArgumentNullException(nameof(preRoll));
            if (preRoll.Count == 0) throw new ArgumentException("raw pre-roll cannot be empty", nameof(preRoll));
            var copy = new List<RawMediaPipeObservation>(preRoll.Count);
            RawObservationConvention convention = null;
            double timestamp = double.NaN;
            foreach (RawMediaPipeObservation observation in preRoll)
            {
                if (observation == null) throw new ArgumentException("raw pre-roll contains null", nameof(preRoll));
                if (!IsAcceptedCaptureConvention(observation))
                    throw new ArgumentException("raw pre-roll convention is not an accepted live MediaPipe convention", nameof(preRoll));
                if (convention == null) convention = observation.Convention;
                else if (!convention.Equals(observation.Convention))
                    throw new ArgumentException("raw pre-roll convention changed", nameof(preRoll));
                if (!double.IsNaN(timestamp) && observation.TimestampSeconds <= timestamp)
                    throw new ArgumentException("raw pre-roll timestamps must be strictly increasing", nameof(preRoll));
                timestamp = observation.TimestampSeconds;
                copy.Add(observation);
            }
            return copy;
        }

        bool TryAppendObservation(RawMediaPipeObservation observation, out string rejection)
        {
            rejection = string.Empty;
            if (observation == null)
            {
                rejection = "bridge emitted a null observation";
                return false;
            }
            if (!IsAcceptedCaptureConvention(observation))
            {
                rejection = "observation convention is not an accepted live MediaPipe convention, "
                    + "or its upright inference geometry is missing, mixed or unsupported";
                return false;
            }
            if (captureConvention == null) captureConvention = observation.Convention;
            else if (!captureConvention.Equals(observation.Convention))
            {
                rejection = "observation convention changed within the capture";
                return false;
            }
            if (!double.IsNaN(lastTimestamp) && observation.TimestampSeconds <= lastTimestamp)
            {
                rejection = "observation timestamps must be strictly increasing";
                return false;
            }
            AppendValidatedObservation(observation);
            return true;
        }

        void AppendValidatedObservation(RawMediaPipeObservation observation)
        {
            if (captureConvention == null) captureConvention = observation.Convention;
            observations.Add(observation);
            lastTimestamp = observation.TimestampSeconds;
        }

        void ResetPrimedTerminalEcho()
        {
            primedTerminalObservation = null;
            suppressPrimedTerminalEcho = false;
        }

        void OnSourceFailure(Exception exception)
        {
            if (State == PersonalLiveCaptureState.Completed ||
                State == PersonalLiveCaptureState.Cancelled ||
                State == PersonalLiveCaptureState.Failed)
                return;

            var ambiguity = exception as Step9CAmbiguousHandIdentityException;
            if (candidateTransientAmbiguityToleranceActive && ambiguity != null)
            {
                if (State == PersonalLiveCaptureState.Ready)
                {
                    RecordNonfatalAmbiguity(ambiguity, false);
                    return;
                }
                // Step 10P v1.0e: every typed ambiguity in the active capture is a
                // dropped source event, not only the first one. The live run showed why:
                // a second Step9CAmbiguousHandIdentityException exhausted the one-event
                // tolerance and was promoted to a fatal BridgeFailure, which cleared the
                // observations already buffered for that capture. Nothing is synthesized
                // here - no observation is appended, no timestamp or convention moves and
                // no hand identity is inferred - so a capture that ends up with no valid
                // observation still fails EmptyCapture at finalize.
                if (State == PersonalLiveCaptureState.Capturing)
                {
                    if (ToleratedAmbiguousHandIdentityCount < int.MaxValue)
                        ToleratedAmbiguousHandIdentityCount++;
                    RecordNonfatalAmbiguity(ambiguity, true);
                    return;
                }
            }

            Fail(PersonalLiveFailureCode.BridgeFailure,
                "Step 9C observation bridge failed: " + (exception?.Message ?? "unknown bridge failure"),
                exception);
        }

        void RecordNonfatalAmbiguity(
            Step9CAmbiguousHandIdentityException exception,
            bool droppedDuringCapture)
        {
            LastToleratedAmbiguousHandIdentity = new PersonalLiveAmbiguityDiagnostic(
                exception,
                State,
                captureSessionId,
                observations.Count,
                droppedDuringCapture);
        }

        void ResetCaptureAmbiguityTolerance()
        {
            ToleratedAmbiguousHandIdentityCount = 0;
            LastToleratedAmbiguousHandIdentity = null;
        }

        void Subscribe()
        {
            if (subscribed || observationSource == null) return;
            observationSource.ObservationArrived += OnObservation;
            observationSource.SourceFailed += OnSourceFailure;
            subscribed = true;
        }

        void Unsubscribe()
        {
            if (!subscribed || observationSource == null) return;
            observationSource.ObservationArrived -= OnObservation;
            observationSource.SourceFailed -= OnSourceFailure;
            subscribed = false;
        }

        bool Fail(PersonalLiveFailureCode code, string message, Exception exception = null)
        {
            // F6 opt-in, read-only failure note. Off unless deliberately enabled, needs no
            // model, scores nothing, and contains every fault internally, so the outcome
            // below is identical with or without it. It records that THIS diagnostic
            // neither loaded a model nor scored; whether the attempt was already observed
            // or scored elsewhere - a Processing-stage failure can follow a completed
            // observation of the same capture - is unobserved here, not false. The failure
            // message is deliberately NOT passed: only the typed code, the exception TYPE
            // name and the pre-failure counts are.
            WordsInSilence.Recognition.FullHandRuntime.FullHandEarlyFailureObserver
                .SafeReportFailure(captureSessionId, ExpectedWord, State.ToString(),
                                   code.ToString(),
                                   exception == null ? null : exception.GetType().Name,
                                   observations.Count);
            if (PrimaryFailure == null)
            {
                PrimaryFailure = new PersonalLivePrimaryFailureSnapshot(
                    code,
                    message,
                    exception,
                    State,
                    captureSessionId,
                    observations.Count);
            }
            Unsubscribe();
            observations.Clear();
            sequenceInput.Reset();
            captureConvention = null;
            lastTimestamp = double.NaN;
            ResetPrimedTerminalEcho();
            Result = null;
            Failure = new PersonalLiveFailure(code, message, exception);
            State = PersonalLiveCaptureState.Failed;
            return false;
        }

        void ThrowIfDisposed()
        {
            if (disposed) throw new ObjectDisposedException(nameof(PersonalLiveRecognitionController));
        }
    }
}
