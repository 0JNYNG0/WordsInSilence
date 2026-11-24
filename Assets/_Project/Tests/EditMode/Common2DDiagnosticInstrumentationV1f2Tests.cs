using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;
using WordsInSilence.Practice.PetRobotPortfolioQuest;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Step 10P v1.0f2b focused gate for the common-2D diagnostic instrumentation.
    ///
    /// Every test drives real implementation types over the real frozen artifact. One
    /// synthetic raw fixture goes through the real
    /// <see cref="ApprovedLivePreprocessingAdapter.AdaptCommon2D"/>, and the normalized
    /// input handed to the real classifier is the output of exactly those raw
    /// observations. Tests whose names name lifecycle or outcome semantics drive the real
    /// <see cref="PersonalLiveRecognitionController"/> and the real
    /// <see cref="CommonSignalTwoTargetValidationCoordinator"/>; only the observation
    /// source, the runtime factory and a persistence port are spies. Exports are read
    /// back member by member and the frozen scaler, PCA and LR are recomputed from the
    /// exported feature rather than reread.
    ///
    /// The fixture is written for a later authorized Unity run. It is not executed during
    /// Step 10P v1.0f2b.
    /// </summary>
    [TestFixture]
    public sealed class Common2DDiagnosticInstrumentationV1f2Tests
    {
        const int PoseLandmarks = 33;
        const int HandLandmarks = 21;
        const int Common2DLandmarks = 16;

        string scratchRoot;
        object owner;

        /// <summary>
        /// Every real Windows junction this fixture created. A junction must be removed
        /// as a link, without descending into it: a recursive delete walks through the
        /// reparse point and fails on, or worse deletes inside, the target.
        /// </summary>
        readonly List<string> createdJunctionLinks = new List<string>();

        /// <summary>
        /// World dialogue hosts created by the Step 10T policy tests. They are destroyed
        /// in TearDown so no scene object survives a test.
        /// </summary>
        readonly List<GameObject> worldControllerObjects = new List<GameObject>();

        [SetUp]
        public void SetUp()
        {
            Common2DDiagnosticTraceBuffer.ResetAll();
            Common2DDiagnosticTraceExporter.FaultInjection = Common2DExportFaultInjection.None;
            owner = new object();
            createdJunctionLinks.Clear();
            scratchRoot = Path.Combine(
                Path.GetTempPath(), "wis_v1_0f2b_" + Guid.NewGuid().ToString("N"));
        }

        [TearDown]
        public void TearDown()
        {
            Common2DDiagnosticTraceBuffer.ResetAll();
            Common2DDiagnosticTraceExporter.FaultInjection = Common2DExportFaultInjection.None;
            // Links first, each one non-recursively, so the reparse point itself goes and
            // nothing is followed into its target. Nothing here is caught: a cleanup that
            // cannot complete must stay visible as a failure.
            for (int i = createdJunctionLinks.Count - 1; i >= 0; i--)
            {
                string link = createdJunctionLinks[i];
                if (Directory.Exists(link)) Directory.Delete(link, false);
                Assert.That(Directory.Exists(link), Is.False,
                    "the junction link must be gone before the scratch root is deleted: " +
                    link);
            }
            createdJunctionLinks.Clear();
            for (int i = worldControllerObjects.Count - 1; i >= 0; i--)
                if (worldControllerObjects[i] != null)
                    UnityEngine.Object.DestroyImmediate(worldControllerObjects[i]);
            worldControllerObjects.Clear();
            if (Directory.Exists(scratchRoot)) Directory.Delete(scratchRoot, true);
        }

        // ------------------------------------------------------------------ spies
        sealed class ObservationSourceSpy : IPersonalLiveObservationSource
        {
            public bool IsBound { get; set; }
            public int SubscribeCount { get; private set; }
            public int UnsubscribeCount { get; private set; }

            event Action<RawMediaPipeObservation> observationArrived;
            event Action<Exception> sourceFailed;

            public event Action<RawMediaPipeObservation> ObservationArrived
            {
                add { observationArrived += value; SubscribeCount++; }
                remove { observationArrived -= value; UnsubscribeCount++; }
            }

            public event Action<Exception> SourceFailed
            {
                add { sourceFailed += value; }
                remove { sourceFailed -= value; }
            }

            public void Emit(RawMediaPipeObservation observation)
            {
                if (observationArrived != null) observationArrived(observation);
            }
        }

        sealed class GlobalRuntimeSpy : IPersonalLiveGlobalRuntime { }

        /// <summary>
        /// Step 10W v1.0a. The personal routing result the fixture forces.
        ///
        /// StrictSuccess is the historical fixture behaviour. LowConfidenceCorrect
        /// reproduces the exact Step 10V shape the World repair exists for: the expected
        /// word IS the predicted word, so the guided evaluator does not take its
        /// WrongWord branch, but no strict flag is set, so it does not take its
        /// StrictSuccess branch either and falls through to LowConfidenceCorrect.
        /// </summary>
        internal enum PersonalRoutingSpyMode
        {
            StrictSuccess,
            LowConfidenceCorrect,
        }

        sealed class RecognitionRuntimeSpy : IPersonalLiveRecognitionRuntime
        {
            public PersonalRoutingSpyMode Mode = PersonalRoutingSpyMode.StrictSuccess;
            public bool IsPersonalActive { get { return true; } }
            public string ActiveProfileId
            {
                get { return PersonalMultiPrototypeRuntimeProfileLoader.ProfileId; }
            }
            public int RecognizeCalls { get; private set; }

            public PersonalLiveRoutingResult Recognize(
                string sessionId, RecognitionSequence sequence, string expectedWord)
            {
                RecognizeCalls++;
                bool strict = Mode == PersonalRoutingSpyMode.StrictSuccess;
                var scores = new Dictionary<string, double>();
                var accepts = new Dictionary<string, bool>();
                foreach (string word in PersonalLiveRecognitionController.SupportedWords)
                {
                    bool hit = string.Equals(word, expectedWord, StringComparison.Ordinal);
                    scores[word] = hit ? 0.20 : 0.80;
                    accepts[word] = hit && strict;
                }
                // predictedWord stays the expected word in BOTH modes. Only the three
                // strict flags move, which is what separates LowConfidenceCorrect from
                // StrictSuccess in the real guided evaluator.
                return new PersonalLiveRoutingResult(
                    expectedWord, strict, strict, strict, scores, accepts, 0.20,
                    "fixture-prototype", new string[0]);
            }
        }

        sealed class RuntimeFactorySpy : IPersonalLiveRuntimeFactory
        {
            public readonly RecognitionRuntimeSpy Runtime = new RecognitionRuntimeSpy();
            public int LoadGlobalCalls { get; private set; }
            public int CreateAndActivateCalls { get; private set; }

            public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths)
            {
                LoadGlobalCalls++;
                return new GlobalRuntimeSpy();
            }

            public IPersonalLiveRecognitionRuntime CreateAndActivate(
                IPersonalLiveGlobalRuntime globalRuntime, PersonalLiveRuntimePaths paths)
            {
                CreateAndActivateCalls++;
                return Runtime;
            }
        }

        /// <summary>
        /// The coordinator's adapter port, backed by the real controller. Only the call
        /// counts are observed here; the coordinator itself is the real subject.
        /// </summary>
        /// <summary>
        /// The one validation port. It owns no controller and no classifier: every call
        /// is delegated to the SAME real PetRobotSignPracticeSession, and the returned
        /// attempt info is composed only from that session's completed LastAttempt.
        ///
        /// This is what makes the comparison honest. The v1.0f2d port finalized a second
        /// controller and reclassified a copied capture, so the compared common bits came
        /// from a different attempt than the progression and persistence values it was
        /// compared against.
        /// </summary>
        sealed class SessionBackedValidationPort : ICommonSignalValidationAdapterPort
        {
            readonly PetRobotSignPracticeSession practice;
            readonly PersonalLivePracticeRecognitionAdapter practiceAdapter;
            readonly ObservationSourceSpy source;
            readonly List<RawMediaPipeObservation> rawFixture;

            public int SetExpectedWordCalls { get; private set; }
            public int BeginCaptureCalls { get; private set; }
            public int FinalizeCaptureCalls { get; private set; }
            public int ResetCaptureCalls { get; private set; }
            public int EmittedObservationCount { get; private set; }

            /// <summary>The completed attempt this port produced; nothing else.</summary>
            public PracticeRecognitionAttempt LastAttempt { get; private set; }

            public SessionBackedValidationPort(
                PetRobotSignPracticeSession practice,
                PersonalLivePracticeRecognitionAdapter practiceAdapter,
                ObservationSourceSpy source,
                List<RawMediaPipeObservation> rawFixture)
            {
                this.practice = practice;
                this.practiceAdapter = practiceAdapter;
                this.source = source;
                this.rawFixture = rawFixture;
            }

            /// <summary>The exact production adapter this port delegates to.</summary>
            public PersonalLivePracticeRecognitionAdapter DelegateAdapter
            {
                get { return practiceAdapter; }
            }

            /// <summary>How often the real adapter selected the target for this port.</summary>
            public int AdapterSetExpectedWordDelegations { get; private set; }
            public string LastDelegatedTarget { get; private set; }
            public bool LastDelegationResult { get; private set; }

            /// <summary>The one raw list instance this port emits, for identity checks.</summary>
            public List<RawMediaPipeObservation> RawFixture { get { return rawFixture; } }

            public bool SetExpectedWord(string target)
            {
                SetExpectedWordCalls++;
                // The session's current prompt must already be this target, AND the exact
                // production adapter the real session received must actually select it.
                // Returning the equality alone would let the coordinator reach this port
                // without the production adapter ever being asked.
                if (!string.Equals(practice.CurrentWord, target, StringComparison.Ordinal))
                    return false;
                AdapterSetExpectedWordDelegations++;
                LastDelegatedTarget = target;
                LastDelegationResult = practiceAdapter.SetExpectedWord(target);
                return LastDelegationResult;
            }

            public bool BeginCapture()
            {
                BeginCaptureCalls++;
                return practice.StartExpression();
            }

            public bool ResetCapture()
            {
                ResetCaptureCalls++;
                return practice.Retry();
            }

            public ValidationAttemptInfo FinalizeCapture()
            {
                FinalizeCaptureCalls++;
                for (int i = 0; i < rawFixture.Count; i++)
                {
                    source.Emit(rawFixture[i]);
                    EmittedObservationCount++;
                }
                if (!practice.FinishExpression()) return null;
                LastAttempt = practice.LastAttempt;
                if (LastAttempt == null) return null;
                CommonSignClassification common = LastAttempt.CommonSignal;
                string record = common == null
                    ? "common=none"
                    : "common=" + common.TopClass + ";support=" +
                      (common.ExpectedTargetSupport ? "1" : "0");
                return new ValidationAttemptInfo(
                    LastAttempt.CaptureSessionId,
                    LastAttempt.CaptureState.ToString(), record);
            }
        }

        /// <summary>
        /// The production ICommonSignClassifier port, forwarding every call to the real
        /// frozen classifier unchanged and counting them. It duplicates no mathematics and
        /// alters no input or output; it exists so the fixture can prove the real common
        /// classifier ran exactly once per attempt.
        /// </summary>
        sealed class CountingCommonClassifier : ICommonSignClassifier
        {
            readonly ICommonSignClassifier inner;

            public int ForwardCalls { get; private set; }

            public CountingCommonClassifier(ICommonSignClassifier inner)
            {
                this.inner = inner;
            }

            public string ArtifactVersion { get { return inner.ArtifactVersion; } }
            public string ArtifactSha256 { get { return inner.ArtifactSha256; } }

            public CommonSignClassification Classify(
                double[] coordinates, bool[] validMask, int frameCount,
                string requestedTarget)
            {
                ForwardCalls++;
                return inner.Classify(coordinates, validMask, frameCount, requestedTarget);
            }
        }

        /// <summary>
        /// The production IQuestHybridRecognitionSession port. It returns one fixed valid
        /// result and counts its calls; it never touches the practice session and never
        /// reimplements the session's quest policy.
        /// </summary>
        enum QuestHybridSpyBehaviour
        {
            ReturnFixedResult,
            ReturnNull,
            Throw,
        }

        sealed class QuestHybridSessionSpy : IQuestHybridRecognitionSession
        {
            readonly TrustedHybridRecognitionResult fixedResult;

            public int RecognizeCalls { get; private set; }
            public string LastQuerySessionId { get; private set; }
            public string LastExpectedWord { get; private set; }

            public QuestHybridSessionSpy(TrustedHybridOutcomeKind kind)
                : this(kind, QuestHybridSpyBehaviour.ReturnFixedResult)
            {
            }

            /// <summary>
            /// Step 10W v1.0. The legacy port is also allowed to return nothing or to
            /// throw, because the repair has to hold when the legacy pass fails, not only
            /// when it disagrees.
            /// </summary>
            public QuestHybridSessionSpy(
                TrustedHybridOutcomeKind kind, QuestHybridSpyBehaviour behaviour)
            {
                FixedKind = kind;
                Behaviour = behaviour;
                fixedResult = new TrustedHybridRecognitionResult(
                    kind, "GREETING", null, null, null);
            }

            public TrustedHybridOutcomeKind FixedKind { get; }
            public QuestHybridSpyBehaviour Behaviour { get; }
            public long Generation { get { return 7L; } }

            public TrustedHybridRecognitionResult Recognize(
                string querySessionId, RecognitionSequence query, string expectedWord)
            {
                RecognizeCalls++;
                LastQuerySessionId = querySessionId;
                LastExpectedWord = expectedWord;
                if (Behaviour == QuestHybridSpyBehaviour.ReturnNull) return null;
                if (Behaviour == QuestHybridSpyBehaviour.Throw)
                    throw new InvalidOperationException(
                        "invented legacy hybrid failure");
                return fixedResult;
            }
        }

        // ------------------------------------------------------------------ json
        /// <summary>
        /// A real recursive JSON value. The exported convention and geometry members are
        /// top-level objects whose "observations" key holds an ARRAY of per-frame
        /// objects, so a flat key/value dictionary cannot represent them: it would turn
        /// the array into text and hide every nested field. This decoder keeps the
        /// structure, which is what makes the per-frame comparison below real.
        /// </summary>
        sealed class JsonValue
        {
            public enum JsonKind { Object, Array, String, Number, Boolean, Null }

            public JsonKind Kind;
            public readonly Dictionary<string, JsonValue> Members =
                new Dictionary<string, JsonValue>(StringComparer.Ordinal);
            public readonly List<JsonValue> Items = new List<JsonValue>();
            public string Text;
            public double Number;
            public bool Boolean;

            public JsonValue Member(string name)
            {
                Assert.That(Kind, Is.EqualTo(JsonKind.Object),
                    "not an object when reading " + name);
                Assert.That(Members.ContainsKey(name), Is.True, "missing member " + name);
                return Members[name];
            }

            public string AsString(string where)
            {
                Assert.That(Kind, Is.EqualTo(JsonKind.String), where + " is not a string");
                return Text;
            }

            public long AsInteger(string where)
            {
                Assert.That(Kind, Is.EqualTo(JsonKind.Number), where + " is not a number");
                return (long)Number;
            }

            public bool AsBoolean(string where)
            {
                Assert.That(Kind, Is.EqualTo(JsonKind.Boolean),
                    where + " is not a boolean");
                return Boolean;
            }
        }

        /// <summary>A strict recursive-descent JSON reader. It accepts no bare NaN.</summary>
        sealed class JsonReader
        {
            readonly string text;
            int index;

            JsonReader(string text)
            {
                this.text = text;
            }

            public static JsonValue Parse(string document)
            {
                var reader = new JsonReader(document);
                reader.SkipWhitespace();
                JsonValue value = reader.ReadValue();
                reader.SkipWhitespace();
                Assert.That(reader.index, Is.EqualTo(document.Length),
                    "trailing content after the JSON document");
                return value;
            }

            void SkipWhitespace()
            {
                while (index < text.Length && char.IsWhiteSpace(text[index])) index++;
            }

            void Expect(char expected)
            {
                Assert.That(index, Is.LessThan(text.Length),
                    "unexpected end of JSON, wanted " + expected);
                Assert.That(text[index], Is.EqualTo(expected),
                    "wanted " + expected + " at " + index);
                index++;
            }

            JsonValue ReadValue()
            {
                SkipWhitespace();
                Assert.That(index, Is.LessThan(text.Length), "unexpected end of JSON");
                char c = text[index];
                if (c == '{') return ReadObject();
                if (c == '[') return ReadArray();
                if (c == '"') return ReadString();
                if (c == 't' || c == 'f') return ReadBoolean();
                if (c == 'n') return ReadNull();
                return ReadNumber();
            }

            JsonValue ReadObject()
            {
                var value = new JsonValue { Kind = JsonValue.JsonKind.Object };
                Expect('{');
                SkipWhitespace();
                if (index < text.Length && text[index] == '}') { index++; return value; }
                while (true)
                {
                    SkipWhitespace();
                    string name = ReadString().Text;
                    SkipWhitespace();
                    Expect(':');
                    value.Members[name] = ReadValue();
                    SkipWhitespace();
                    Assert.That(index, Is.LessThan(text.Length), "unterminated object");
                    if (text[index] == ',') { index++; continue; }
                    Expect('}');
                    return value;
                }
            }

            JsonValue ReadArray()
            {
                var value = new JsonValue { Kind = JsonValue.JsonKind.Array };
                Expect('[');
                SkipWhitespace();
                if (index < text.Length && text[index] == ']') { index++; return value; }
                while (true)
                {
                    value.Items.Add(ReadValue());
                    SkipWhitespace();
                    Assert.That(index, Is.LessThan(text.Length), "unterminated array");
                    if (text[index] == ',') { index++; continue; }
                    Expect(']');
                    return value;
                }
            }

            JsonValue ReadString()
            {
                Expect('"');
                var builder = new StringBuilder();
                while (true)
                {
                    Assert.That(index, Is.LessThan(text.Length), "unterminated string");
                    char c = text[index++];
                    if (c == '"') break;
                    if (c != '\\') { builder.Append(c); continue; }
                    Assert.That(index, Is.LessThan(text.Length), "unterminated escape");
                    char escape = text[index++];
                    if (escape == 'n') builder.Append('\n');
                    else if (escape == 'r') builder.Append('\r');
                    else if (escape == 't') builder.Append('\t');
                    else if (escape == 'b') builder.Append('\b');
                    else if (escape == 'f') builder.Append('\f');
                    else if (escape == 'u')
                    {
                        Assert.That(index + 4, Is.LessThanOrEqualTo(text.Length),
                            "truncated unicode escape");
                        builder.Append((char)Convert.ToInt32(
                            text.Substring(index, 4), 16));
                        index += 4;
                    }
                    else builder.Append(escape);
                }
                return new JsonValue
                {
                    Kind = JsonValue.JsonKind.String,
                    Text = builder.ToString(),
                };
            }

            JsonValue ReadBoolean()
            {
                if (string.CompareOrdinal(text, index, "true", 0, 4) == 0)
                {
                    index += 4;
                    return new JsonValue
                    {
                        Kind = JsonValue.JsonKind.Boolean, Boolean = true,
                    };
                }
                Assert.That(string.CompareOrdinal(text, index, "false", 0, 5), Is.Zero,
                    "invalid literal at " + index);
                index += 5;
                return new JsonValue { Kind = JsonValue.JsonKind.Boolean, Boolean = false };
            }

            JsonValue ReadNull()
            {
                Assert.That(string.CompareOrdinal(text, index, "null", 0, 4), Is.Zero,
                    "invalid literal at " + index);
                index += 4;
                return new JsonValue { Kind = JsonValue.JsonKind.Null };
            }

            JsonValue ReadNumber()
            {
                int start = index;
                while (index < text.Length &&
                       ("+-.eE0123456789".IndexOf(text[index]) >= 0)) index++;
                string raw = text.Substring(start, index - start);
                Assert.That(raw.Length, Is.GreaterThan(0), "not a JSON value at " + start);
                Assert.That(raw, Is.Not.EqualTo("NaN"), "a bare NaN is never valid JSON");
                return new JsonValue
                {
                    Kind = JsonValue.JsonKind.Number,
                    Number = double.Parse(raw, CultureInfo.InvariantCulture),
                    Text = raw,
                };
            }
        }

        /// <summary>
        /// The production camera-readiness port. The practice session refuses to start
        /// while the camera is not ready, so this spy stands in for the real startup
        /// route and nothing else.
        /// </summary>
        sealed class CameraReadinessSpy : IPetRobotCandidateCameraReadiness
        {
            public event Action Changed;

            public CandidateCameraStartupState CameraState
            {
                get { return CandidateCameraStartupState.Ready; }
            }

            public bool IsReady { get { return true; } }
            public bool CameraStarted { get { return true; } }
            public bool HasReceivedFrame { get { return true; } }
            public bool ProviderObservationReady { get { return true; } }
            public int CameraFrameCount { get { return 1; } }
            public int ProviderObservationCount { get { return 1; } }
            public string LastErrorCode { get { return string.Empty; } }
            public string LastErrorMessage { get { return string.Empty; } }

            /// <summary>Never called; present so the event is not reported unused.</summary>
            public void RaiseChanged()
            {
                if (Changed != null) Changed();
            }
        }

        // ------------------------------------------------------------------ fixture
        static FrozenCommonSignClassifier LoadRealClassifier()
        {
            FrozenCommonSignClassifier classifier =
                FrozenCommonSignClassifier.Load(CommonSignClassifierProvider.DefaultModelDirectory);
            Assert.That(classifier.ModelAvailable, Is.True,
                "the frozen common-2D artifact must load for this gate: " +
                classifier.LoadDetail);
            return classifier;
        }

        static RawLandmarkObservationSet BuildHand(
            int count, double seed, double drift, bool includeAbsent)
        {
            var coordinates = new double[count * 3];
            var valid = new bool[count];
            var reappearance = new bool[count];
            var boundary = new bool[count];
            for (int i = 0; i < count; i++)
            {
                bool present = !(includeAbsent && i == 1);
                valid[i] = present;
                reappearance[i] = i == 2;
                boundary[i] = i == 3;
                for (int axis = 0; axis < 3; axis++)
                {
                    coordinates[i * 3 + axis] = present
                        ? seed + 0.0031 * i + 0.0007 * axis + drift
                        : double.NaN;
                }
            }
            return new RawLandmarkObservationSet(count, coordinates, valid, reappearance, boundary);
        }

        static Step9DInferenceFrameGeometry BuildGeometry()
        {
            return new Step9DInferenceFrameGeometry(
                Step9DInferenceGeometryProvenance.TestInjected,
                640, 480, 30, 640, 480, 0, false, 640, 480, 640, 480, "SRGBA",
                Step9DPreInferenceTransformState.None, Step9DPreInferenceTransformState.None, 0,
                Step9DPreInferenceTransformState.None, Step9DPreInferenceTransformState.None,
                Step9DPreInferenceTransformState.None, null, 0L, null, "TopDown");
        }

        static List<RawMediaPipeObservation> BuildObservations(int count, bool includeAbsent)
        {
            var observations = new List<RawMediaPipeObservation>(count);
            for (int i = 0; i < count; i++)
            {
                double t = i * 0.05;
                var pose = new double[PoseLandmarks * 3];
                var poseValid = new bool[PoseLandmarks];
                var poseReappear = new bool[PoseLandmarks];
                var poseBoundary = new bool[PoseLandmarks];
                for (int landmark = 0; landmark < PoseLandmarks; landmark++)
                {
                    poseValid[landmark] = true;
                    poseReappear[landmark] = landmark == 12 && i == 2;
                    poseBoundary[landmark] = landmark == 14 && i == 3;
                    double baseX = 0.30 + 0.02 * (landmark % 5);
                    double baseY = 0.25 + 0.015 * (landmark % 7);
                    if (landmark == 11) { baseX = 0.40; baseY = 0.45; }
                    if (landmark == 12) { baseX = 0.60; baseY = 0.45; }
                    if (landmark == 23) { baseX = 0.44; baseY = 0.75; }
                    if (landmark == 24) { baseX = 0.56; baseY = 0.75; }
                    if (landmark == 13) { baseX = 0.36 + 0.03 * Math.Sin(t * 3.0); baseY = 0.58; }
                    if (landmark == 14) { baseX = 0.64 - 0.03 * Math.Sin(t * 3.0); baseY = 0.58; }
                    if (landmark == 15) { baseX = 0.34 + 0.05 * Math.Sin(t * 3.0); baseY = 0.68; }
                    if (landmark == 16) { baseX = 0.66 - 0.05 * Math.Sin(t * 3.0); baseY = 0.68; }
                    pose[landmark * 3] = baseX;
                    pose[landmark * 3 + 1] = baseY;
                    pose[landmark * 3 + 2] = 0.01 * Math.Cos(t * 2.0);
                }
                var poseSet = new RawLandmarkObservationSet(
                    PoseLandmarks, pose, poseValid, poseReappear, poseBoundary);
                Step9DInferenceFrameGeometry geometry = BuildGeometry();
                var merged = new Step9DMergedInferenceGeometry(
                    Step9DGeometryAcquisitionResult.ForAvailable(geometry),
                    Step9DGeometryAcquisitionResult.ForAvailable(geometry));
                observations.Add(new RawMediaPipeObservation(
                    0.10 + t, RawObservationConvention.LiveMediaPipe, poseSet,
                    BuildHand(HandLandmarks, 0.34, 0.004 * i, includeAbsent && i == 0),
                    BuildHand(HandLandmarks, 0.62, -0.004 * i, includeAbsent && i == 0),
                    merged));
            }
            return observations;
        }

        static PersonalLiveRuntimePaths BuildPaths(string root)
        {
            Directory.CreateDirectory(root);
            string config = Path.Combine(root, "thresholds.json");
            if (!File.Exists(config)) File.WriteAllText(config, "{}");
            return new PersonalLiveRuntimePaths(root, config, root);
        }

        /// <summary>One capture through the real AdaptCommon2D and the real classifier.</summary>
        static CommonSignClassification RunRealCapture(
            FrozenCommonSignClassifier classifier, string sessionId, string target,
            List<RawMediaPipeObservation> observations,
            out CommonTwoDimensionalCapture capture)
        {
            Common2DDiagnosticTraceBuffer.SafeBeginSession(
                sessionId, target, observations.Count);
            string failure;
            capture = ApprovedLivePreprocessingAdapter.AdaptCommon2D(observations, out failure);
            Assert.That(capture, Is.Not.Null, "AdaptCommon2D rejected the fixture: " + failure);
            CommonSignClassification classification = classifier.Classify(
                capture.CopyCoordinates(), capture.CopyValidMask(), capture.FrameCount, target);
            Common2DDiagnosticTraceBuffer.SafeMarkAttemptComplete(target);
            return classification;
        }

        Common2DDiagnosticExportResult CommitRealExport(
            FrozenCommonSignClassifier classifier, string sessionId, string target,
            int frameCount, bool includeAbsent,
            out CommonSignClassification classification,
            out CommonTwoDimensionalCapture capture,
            out Common2DDiagnosticTrace trace)
        {
            Assert.That(Common2DDiagnosticTraceBuffer.TryArm(owner),
                Is.EqualTo(Common2DArmResult.Armed));
            classification = RunRealCapture(
                classifier, sessionId, target,
                BuildObservations(frameCount, includeAbsent), out capture);
            trace = Common2DDiagnosticTraceBuffer.Active;
            Assert.That(Common2DDiagnosticTraceBuffer.MarkTerminalAccepted(owner), Is.True);
            Common2DDiagnosticExportResult export =
                Common2DDiagnosticTraceExporter.ExportToScratchRoot(trace, scratchRoot);
            Assert.That(export.Committed, Is.True, export.FailureReason.ToString());
            return export;
        }

        static ulong Bits(double value)
        {
            return unchecked((ulong)BitConverter.DoubleToInt64Bits(value));
        }

        static string Sha256Hex(byte[] bytes)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(bytes);
                var text = new StringBuilder(digest.Length * 2);
                for (int i = 0; i < digest.Length; i++)
                    text.Append(digest[i].ToString("x2", CultureInfo.InvariantCulture));
                return text.ToString();
            }
        }

        static double[] DecodeDoubles(byte[] data)
        {
            var values = new double[data.Length / 8];
            for (int i = 0; i < values.Length; i++)
            {
                ulong word = 0;
                for (int b = 7; b >= 0; b--) word = (word << 8) | data[i * 8 + b];
                values[i] = BitConverter.Int64BitsToDouble(unchecked((long)word));
            }
            return values;
        }

        static double[] ReadDoubles(string path)
        {
            return DecodeDoubles(File.ReadAllBytes(path));
        }

        /// <summary>Decodes a packed bool block in the declared LSB-first order.</summary>
        static bool[] DecodePackedBools(byte[] data, int count)
        {
            var values = new bool[count];
            for (int i = 0; i < count; i++)
                values[i] = (data[i / 8] & (1 << (i % 8))) != 0;
            return values;
        }

        /// <summary>The six frozen parameter blocks, read read-only for the replay.</summary>
        static double[] ReadFrozenParameter(string fileName, int expectedCount)
        {
            string path = Path.Combine(
                CommonSignClassifierProvider.DefaultModelDirectory, fileName);
            double[] values = ReadDoubles(path);
            Assert.That(values.Length, Is.EqualTo(expectedCount), fileName);
            return values;
        }

        /// <summary>
        /// Resolves payload_manifest.json structurally with the fixture's own recursive
        /// JSON reader. The substring reader this replaces stopped at the first comma it
        /// saw, so a quoted shape such as "[12,33,3]" was silently truncated to "[12" and
        /// every declared shape was compared against a fragment of itself.
        /// </summary>
        static Dictionary<string, JsonValue> ManifestMembers(string manifest)
        {
            JsonValue members = JsonReader.Parse(manifest).Member("members");
            Assert.That(members.Kind, Is.EqualTo(JsonValue.JsonKind.Array),
                "the manifest members must be an array");
            var table = new Dictionary<string, JsonValue>(StringComparer.Ordinal);
            for (int i = 0; i < members.Items.Count; i++)
            {
                JsonValue entry = members.Items[i];
                Assert.That(entry.Kind, Is.EqualTo(JsonValue.JsonKind.Object),
                    "manifest member at " + i + " is not an object");
                string name = entry.Member("name").AsString("member name at " + i);
                Assert.That(table.ContainsKey(name), Is.False,
                    "exactly one manifest entry may carry the name " + name);
                table.Add(name, entry);
            }
            return table;
        }

        /// <summary>The single manifest entry with this exact name, or a failure.</summary>
        static JsonValue ManifestMember(Dictionary<string, JsonValue> members, string name)
        {
            Assert.That(members.ContainsKey(name), Is.True,
                "member not in manifest: " + name);
            return members[name];
        }

        static string ManifestText(
            Dictionary<string, JsonValue> members, string name, string field)
        {
            return ManifestMember(members, name).Member(field)
                .AsString(name + "." + field);
        }

        static long ManifestInteger(
            Dictionary<string, JsonValue> members, string name, string field)
        {
            return ManifestMember(members, name).Member(field)
                .AsInteger(name + "." + field);
        }


        /// <summary>
        /// The REAL production persistence port. This is not a private counter on a test
        /// adapter: it implements IPersonalizationCandidateRepository and is handed to the
        /// actual PetRobotSignPracticeSession, so every count below is a call the
        /// production consumer made through the interface it really writes through.
        /// </summary>
        sealed class CandidateRepositorySpy : IPersonalizationCandidateRepository
        {
            readonly string root;

            public int SaveCalls { get; private set; }
            public int ReloadCalls { get; private set; }
            public int CountsReads { get; private set; }
            public readonly List<string> SavedWords = new List<string>();

            public CandidateRepositorySpy(string root)
            {
                this.root = root;
            }

            public string RootDirectory { get { return root; } }
            public string LearnerId { get { return "fixture-learner"; } }

            public CandidateCountSnapshot Counts
            {
                get { CountsReads++; return EmptySnapshot(); }
            }

            public CandidateCountSnapshot ReloadAndVerify()
            {
                ReloadCalls++;
                return EmptySnapshot();
            }

            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                SavedWords.Add(request == null || request.Attempt == null
                    ? string.Empty : request.Attempt.ExpectedWord);
                return new CandidateSaveResult(
                    CandidateStorageStatus.NotRequested, string.Empty,
                    "fixture port records the call and writes nothing",
                    EmptySnapshot());
            }

            static CandidateCountSnapshot EmptySnapshot()
            {
                var eligible = new Dictionary<string, int>
                {
                    { "GREETING", 0 }, { "LOVE", 0 }, { "THANK_YOU", 0 },
                };
                var reviewOnly = new Dictionary<string, int>
                {
                    { "GREETING", 0 }, { "LOVE", 0 }, { "THANK_YOU", 0 },
                };
                return new CandidateCountSnapshot(eligible, reviewOnly);
            }
        }

        /// <summary>
        /// Invokes the named PRIVATE MonoBehaviour lifecycle callback body on a real
        /// production component. EditMode has no runtime lifecycle, so SetActive(false)
        /// and DestroyImmediate deliver no OnDisable/OnDestroy the way a player run
        /// would. The callback is resolved on the component's own declared type and the
        /// lookup is asserted, so a rename or a removal of the production callback fails
        /// this test loudly instead of silently skipping the check.
        /// </summary>
        static void InvokeRealLifecycleCallback(MonoBehaviour component, string callback)
        {
            Assert.That(component, Is.Not.Null, callback);
            System.Reflection.MethodInfo method = component.GetType().GetMethod(
                callback,
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.DeclaredOnly,
                null, System.Type.EmptyTypes, null);
            Assert.That(method, Is.Not.Null,
                "the production component must declare a private " + callback +
                " callback: " + component.GetType().FullName);
            Assert.That(method.IsPrivate, Is.True, callback + " must stay private");
            method.Invoke(component, null);
        }

        /// <summary>
        /// Creates a REAL Windows directory junction and asserts it actually carries
        /// reparse semantics before it is used. A branch that quietly does nothing when
        /// the alias is an ordinary directory would prove nothing, so the precondition is
        /// an assertion, not an if. The link path is registered so TearDown can remove
        /// the reparse point itself rather than walking through it.
        /// </summary>
        string CreateRealDirectoryJunction(string linkPath, string targetPath)
        {
            Directory.CreateDirectory(targetPath);
            createdJunctionLinks.Add(linkPath);
            string parent = Path.GetDirectoryName(linkPath);
            if (!string.IsNullOrEmpty(parent)) Directory.CreateDirectory(parent);
            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = Environment.GetEnvironmentVariable("ComSpec") ?? "cmd.exe",
                Arguments = "/c mklink /J \"" + linkPath + "\" \"" + targetPath + "\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            using (System.Diagnostics.Process process =
                   System.Diagnostics.Process.Start(startInfo))
            {
                string output = process.StandardOutput.ReadToEnd() +
                                process.StandardError.ReadToEnd();
                process.WaitForExit();
                Assert.That(process.ExitCode, Is.Zero,
                    "the junction must actually be created: " + output);
            }
            Assert.That(Directory.Exists(linkPath), Is.True, "junction not present");
            var info = new DirectoryInfo(linkPath);
            Assert.That((info.Attributes & FileAttributes.ReparsePoint),
                Is.EqualTo(FileAttributes.ReparsePoint),
                "the alias must really carry reparse semantics before it proves anything");
            return linkPath;
        }


        /// <summary>
        /// One complete observation of the real outcome/practice/quest/persistence route
        /// for a single run. Every field is read from a real production object; nothing
        /// here is computed by the fixture.
        /// </summary>
        sealed class RouteObservation
        {
            /// <summary>
            /// The armed route's diagnostic trace, read while the route is still alive.
            /// ConnectedPracticeRoute.Dispose clears the static Active trace, so a caller
            /// that reads it after RunFullRoute returns always sees null and the armed
            /// and disabled runs become indistinguishable.
            /// </summary>
            public Common2DDiagnosticTrace ObservedTrace;

            public string CoordinatorState;
            public string CoordinatorTarget;
            public int FinalizedCount;
            public int AdapterSetExpectedWordCalls;
            public int AdapterBeginCaptureCalls;
            public int AdapterFinalizeCaptureCalls;
            public int AdapterResetCaptureCalls;
            public int RecordedLines;
            public string ControllerState;
            public string PredictedWord;
            public int RecognizeCalls;

            // strict and fused outcome, straight off GuidedQuestOutcomeEvaluator
            public string OutcomeKind;
            public string OutcomeExpectedWord;
            public string OutcomePredictedWord;
            public bool StrictResultAvailable;
            public bool StrictAccepted;
            public bool StrictExpectedWordClassAccept;
            public bool StrictQuestSuccess;
            public bool GuidedProgressAllowed;
            public string OutcomeMessage;
            public string CommonEvidenceState;
            public bool CommonGrantsSuccess;
            public bool CommonGrantsLearnerError;
            public bool CommonGrantsVeto;

            // progression and quest, straight off the real PetRobotSignPracticeSession
            public string PracticeState;
            public string PracticeCurrentWord;
            public bool PracticeCurrentWordComplete;
            public bool PracticeAllWordsComplete;
            public bool PracticeQuestMode;
            public bool PracticeQuestCompleted;
            public bool PracticeCanRetry;

            // persistence, counted at the real production port before and after the
            // attempt so a policy that legitimately saves nothing still proves it ran
            public int PersistenceSaveCalls;
            public int PersistenceReloadCalls;
            public int PersistenceCountsReads;
            public int PersistenceSaveCallsBefore;
            public int PersistenceSaveCallsAfter;
            public int PersistenceCountsReadsBefore;
            public int PersistenceCountsReadsAfter;

            // the completed real practice attempt
            public string PracticeAttemptCaptureSessionId;
            public string PracticeAttemptOutcomeKind;
            public string PracticeStorageStatus;
            public string PracticeStorageMessage;
            public string PracticeAdapterCaptureState;
            public string PracticeAdapterSessionId;

            // the ACTUAL quest state the real production policy produced
            public bool QuestModeBefore;
            public bool QuestMode;
            public string QuestTokenId;
            public long QuestGeneration;
            public int QuestRecognizeCallsBefore;
            public int QuestRecognizeCallsAfter;
            public bool QuestCompletedBefore;
            public bool QuestCompleted;
            public string QuestAuthoritativeOutcome;
            public bool QuestHybridResultPresent;
            public string QuestHybridResultKind;
            public string QuestHybridExpectedWord;

            // the real adapter delegation the coordinator triggered
            public int AdapterSetExpectedWordDelegations;
            public string DelegatedTarget;

            // exact per-case call counts through the one route
            public int PersonalRecognizeCalls;
            public int CommonClassifierForwardCalls;
            public int PortFinalizeCalls;
            public int PortResetCalls;
            public int PortEmittedObservations;
            public int RawFixtureFrames;
            public bool AttemptIsPortAttempt;

            public ulong DecisionBits;
            public ulong GreetingBits;
            public ulong ThankYouBits;
            public string TopClass;
            public bool ExpectedTargetSupport;
        }

        /// <summary>
        /// Builds the REAL practice route: the real developer-facing recognition harness
        /// component, the real PersonalLivePracticeRecognitionAdapter over it, the real
        /// PetRobotSignPracticeSession and the real production persistence port. Only the
        /// repository is a spy, and it is the production interface the session writes
        /// through.
        /// </summary>
        /// <summary>
        /// One fully connected real practice route: the real controller, the real live
        /// harness bound to it, the real production adapter and the real practice
        /// session, with only the production ports spied.
        /// </summary>
        /// <summary>
        /// The ONE object graph every compared value comes from: one initialized real
        /// controller, the real live harness bound to it, one real production practice
        /// adapter, one real quest-mode practice session, one real coordinator and one
        /// validation port that drives that same session. There is deliberately no second
        /// controller, no second classifier and no second raw list anywhere in it.
        /// </summary>
        sealed class ConnectedPracticeRoute
        {
            public GameObject HarnessObject;
            public PersonalLiveRecognitionHarness LiveHarness;
            public PersonalLiveRecognitionController Controller;
            public ObservationSourceSpy Source;
            public RuntimeFactorySpy Factory;
            public CameraReadinessSpy Camera;
            public CandidateRepositorySpy Repository;
            public CountingCommonClassifier CommonClassifier;
            public QuestHybridSessionSpy QuestSession;
            public PetRobotGreetingQuestToken QuestToken;
            public PersonalLivePracticeRecognitionAdapter PracticeAdapter;
            public PetRobotSignPracticeSession Practice;
            public SessionBackedValidationPort Port;
            public CommonSignalTwoTargetValidationCoordinator Coordinator;

            /// <summary>The single raw fixture list, built once and shared by identity.</summary>
            public List<RawMediaPipeObservation> RawFixture;

            /// <summary>
            /// Tears the route down so the ONE injected controller is disposed exactly
            /// once. The harness's private controller reference is detached first, so
            /// the production OnDestroy cannot dispose the same instance a second time,
            /// and the object is destroyed without ever being reactivated.
            /// </summary>
            public void Dispose()
            {
                if (Practice != null) Practice.Dispose();
                if (LiveHarness != null)
                {
                    SetPrivateField(LiveHarness, "controller", null);
                    ControllerDetachedBeforeDestroy =
                        LiveHarness.Controller == null;
                }
                if (Controller != null)
                {
                    Controller.Dispose();
                    ControllerDisposeCalls++;
                }
                if (HarnessObject != null)
                {
                    UnityEngine.Object.DestroyImmediate(HarnessObject);
                    HarnessDestroyed = true;
                }
            }

            /// <summary>Exactly one dispose of the injected controller.</summary>
            public int ControllerDisposeCalls;
            public bool ControllerDetachedBeforeDestroy;
            public bool HarnessDestroyed;
        }

        /// <summary>
        /// Builds that one graph and PROVES it is connected before anything is measured.
        ///
        /// PersonalLiveRecognitionHarness.EnsureControllerReady returns true immediately
        /// when it already holds a ready, personal-active controller, and only otherwise
        /// falls back to rebuilding one from its serialized Step 9C bridge. A newly added
        /// component has neither, so this setup assigns the existing private controller
        /// reference with a REAL PersonalLiveRecognitionController initialized through its
        /// own production API. That is Unity test setup standing in for serialized wiring;
        /// no outcome, progression, quest or persistence value is ever assigned this way.
        /// </summary>
        ConnectedPracticeRoute BuildConnectedPracticeRoute(
            string label, QuestHybridSessionSpy questSession = null)
        {
            var route = new ConnectedPracticeRoute
            {
                Source = new ObservationSourceSpy { IsBound = true },
                Factory = new RuntimeFactorySpy(),
                Camera = new CameraReadinessSpy(),
                Repository = new CandidateRepositorySpy(
                    Path.Combine(scratchRoot, "candidates-" + label)),
                CommonClassifier = new CountingCommonClassifier(LoadRealClassifier()),
                QuestSession = questSession ?? new QuestHybridSessionSpy(
                    TrustedHybridOutcomeKind.StrictSuccess),
                QuestToken = new PetRobotGreetingQuestToken(),
            };

            // the ONE raw fixture list for this case, built exactly once
            route.RawFixture = BuildObservations(RawFixtureFrameCount, false);
            Assert.That(route.RawFixture.Count, Is.EqualTo(RawFixtureFrameCount));

            // the real controller, initialized through its real API
            route.Controller = new PersonalLiveRecognitionController(
                route.Source, route.Factory);
            Assert.That(route.Controller.Initialize(
                BuildPaths(Path.Combine(scratchRoot, "runtime-" + label))), Is.True,
                "the real controller must initialize before it is bound to the harness");
            Assert.That(route.Controller.State,
                Is.EqualTo(PersonalLiveCaptureState.Ready));
            Assert.That(route.Controller.IsPersonalModeActive, Is.True);
            Assert.That(route.Controller.ActivePersonalProfileId,
                Is.EqualTo(PersonalMultiPrototypeRuntimeProfileLoader.ProfileId),
                "the candidate route requires the multi-prototype profile identity");

            // The real harness, bound to that one controller BEFORE any Unity lifecycle
            // callback can run.
            //
            // Unity raises OnEnable when a component is added to an ACTIVE GameObject,
            // and the production PersonalLiveRecognitionHarness.OnEnable immediately
            // calls CreateController(), which constructs a PersonalLiveRecognitionController
            // of its own. Adding the component to an active object would therefore create
            // a preliminary controller before this fixture injects its one real
            // controller. The object is deactivated first, so no callback runs, and it is
            // never reactivated for the life of this connected route: reactivating it
            // would let OnEnable dispose and replace the injected controller.
            route.HarnessObject = new GameObject("f2f-live-harness-" + label);
            route.HarnessObject.SetActive(false);
            Assert.That(route.HarnessObject.activeSelf, Is.False,
                "the harness object must be inactive before AddComponent, so the " +
                "production OnEnable cannot build a preliminary controller");
            route.LiveHarness =
                route.HarnessObject.AddComponent<PersonalLiveRecognitionHarness>();
            Assert.That(route.LiveHarness.isActiveAndEnabled, Is.False,
                "no Unity lifecycle callback may have run on the harness yet");
            Assert.That(route.LiveHarness.Controller, Is.Null,
                "an inactive AddComponent must not have created any controller");

            SetPrivateField(route.LiveHarness, "profileMode",
                PersonalLiveProfileMode.MultiPrototypeCandidate);
            SetPrivateField(route.LiveHarness, "controller", route.Controller);
            Assert.That(route.LiveHarness.ProfileMode,
                Is.EqualTo(PersonalLiveProfileMode.MultiPrototypeCandidate),
                "the practice route only exists on the explicit candidate profile mode");
            Assert.That(ReferenceEquals(route.LiveHarness.Controller, route.Controller),
                Is.True, "the harness must hold the real controller, not a substitute");

            // EnsureControllerReady must take the already-ready fast path and leave the
            // injected reference untouched: it must not rebuild from an observation
            // bridge, which this inactive harness deliberately does not have.
            Assert.That(route.LiveHarness.ObservationBridge, Is.Null,
                "no serialized observation bridge may supply a lifecycle controller");
            PersonalLiveRecognitionController beforeReady = route.LiveHarness.Controller;
            Assert.That(route.LiveHarness.EnsureControllerReady(), Is.True,
                "the bound harness must report a ready controller before the " +
                "production adapter is constructed");
            Assert.That(ReferenceEquals(route.LiveHarness.Controller, beforeReady),
                Is.True,
                "EnsureControllerReady must not replace the injected controller");
            Assert.That(ReferenceEquals(route.LiveHarness.Controller, route.Controller),
                Is.True, "the one injected controller must still be the harness's");
            Assert.That(route.HarnessObject.activeSelf, Is.False,
                "the connected-route harness stays inactive for its whole lifetime");

            // the real production adapter, with the counting delegator on the existing
            // optional ICommonSignClassifier parameter
            route.PracticeAdapter = new PersonalLivePracticeRecognitionAdapter(
                route.LiveHarness, null, route.CommonClassifier);
            Assert.That(route.PracticeAdapter.CandidateRouteActive, Is.True,
                "the production practice adapter must report the candidate route active");
            Assert.That(route.PracticeAdapter.CaptureState,
                Is.EqualTo(PersonalLiveCaptureState.Ready));

            // the real QUEST-MODE practice session over the production ports
            route.Practice = new PetRobotSignPracticeSession(
                route.PracticeAdapter, route.Repository, route.QuestSession,
                route.QuestToken, route.Camera);
            Assert.That(route.Practice.QuestMode, Is.True,
                "the quest-mode constructor must produce an actual quest session");
            Assert.That(ReferenceEquals(route.Practice.QuestToken, route.QuestToken),
                Is.True, "the exact quest token must be bound");
            Assert.That(route.Practice.QuestSessionGeneration, Is.Not.Zero,
                "the quest session generation must be advertised by the real port");
            Assert.That(route.Practice.QuestSessionGeneration,
                Is.EqualTo(route.QuestSession.Generation));
            Assert.That(route.Practice.State,
                Is.EqualTo(PetRobotPracticeState.Ready));
            Assert.That(route.Practice.CandidateRouteActive, Is.True);
            Assert.That(route.Practice.CameraReady, Is.True);
            Assert.That(route.Practice.QuestCompleted, Is.False);
            Assert.That(route.Practice.QuestHybridResult, Is.Null);
            Assert.That(route.QuestSession.RecognizeCalls, Is.Zero);
            Assert.That(route.CommonClassifier.ForwardCalls, Is.Zero);

            // the one validation port over that session, and the real coordinator over it
            route.Port = new SessionBackedValidationPort(
                route.Practice, route.PracticeAdapter, route.Source, route.RawFixture);
            Assert.That(ReferenceEquals(route.Port.RawFixture, route.RawFixture), Is.True,
                "the port must emit the same raw list instance the route built");
            Assert.That(ReferenceEquals(route.Port.DelegateAdapter,
                route.PracticeAdapter), Is.True,
                "the port must delegate to the exact adapter the real session received");
            route.Coordinator = new CommonSignalTwoTargetValidationCoordinator(
                route.Port, line => { });
            return route;
        }

        /// <summary>The single frame count every case in this fixture uses.</summary>
        const int RawFixtureFrameCount = 13;

        /// <summary>
        /// Drives THE one real attempt through the one route: the coordinator begins,
        /// which starts the real practice session; the coordinator finalizes, which emits
        /// the one raw list and finishes that same session. Every compared value is then
        /// read from that attempt.
        /// </summary>
        void DriveRealPracticeAttempt(ConnectedPracticeRoute route, RouteObservation into)
        {
            into.PersistenceCountsReadsBefore = route.Repository.CountsReads;
            into.PersistenceSaveCallsBefore = route.Repository.SaveCalls;
            into.QuestRecognizeCallsBefore = route.QuestSession.RecognizeCalls;
            into.QuestCompletedBefore = route.Practice.QuestCompleted;
            into.QuestGeneration = route.Practice.QuestSessionGeneration;
            into.QuestTokenId = route.Practice.QuestToken.Id;
            into.QuestModeBefore = route.Practice.QuestMode;

            Assert.That(route.Port.AdapterSetExpectedWordDelegations, Is.Zero,
                "the production adapter must not have been asked before Begin");
            Assert.That(route.Coordinator.Begin(), Is.True,
                "the coordinator must start the real practice session");

            // the coordinator reached THIS port, and this port made the real adapter
            // select the target on the same production controller
            Assert.That(route.Port.SetExpectedWordCalls, Is.GreaterThan(0),
                "the coordinator must reach the port's SetExpectedWord");
            Assert.That(route.Port.AdapterSetExpectedWordDelegations,
                Is.EqualTo(route.Port.SetExpectedWordCalls),
                "every accepted port target selection must reach the real adapter");
            Assert.That(route.Port.LastDelegationResult, Is.True,
                "the production adapter must have accepted the target");
            Assert.That(route.Controller.ExpectedWord,
                Is.EqualTo(route.Port.LastDelegatedTarget),
                "the production controller target must equal the delegated target");
            Assert.That(route.Practice.CurrentWord,
                Is.EqualTo(route.Port.LastDelegatedTarget));
            into.AdapterSetExpectedWordDelegations =
                route.Port.AdapterSetExpectedWordDelegations;
            into.DelegatedTarget = route.Port.LastDelegatedTarget;
            Assert.That(route.Practice.State,
                Is.EqualTo(PetRobotPracticeState.Capturing),
                "Begin must move the real session into Capturing");
            Assert.That(route.PracticeAdapter.CaptureState,
                Is.EqualTo(PersonalLiveCaptureState.Capturing));

            Assert.That(route.Coordinator.Finalize(), Is.True,
                "the coordinator must complete the one real attempt");

            // exactly one of everything, through one route
            Assert.That(route.Factory.Runtime.RecognizeCalls, Is.EqualTo(1),
                "exactly one personal runtime recognition per case");
            Assert.That(route.CommonClassifier.ForwardCalls, Is.EqualTo(1),
                "exactly one real common-classifier forward per case");
            Assert.That(route.QuestSession.RecognizeCalls, Is.EqualTo(1),
                "exactly one quest-session recognition per case");
            Assert.That(route.Coordinator.AdapterFinalizeCaptureCalls, Is.EqualTo(1));
            Assert.That(route.Port.FinalizeCaptureCalls, Is.EqualTo(1));
            Assert.That(route.Port.ResetCaptureCalls, Is.Zero,
                "no retry or reset on the first target");
            Assert.That(route.Port.EmittedObservationCount,
                Is.EqualTo(RawFixtureFrameCount),
                "the one raw list was emitted exactly once");

            // the completed attempt, and every value that will be compared from it
            PracticeRecognitionAttempt attempt = route.Practice.LastAttempt;
            Assert.That(attempt, Is.Not.Null,
                "a completed real attempt must be recorded");
            Assert.That(ReferenceEquals(route.Port.LastAttempt, attempt), Is.True,
                "the port must report the same attempt the session holds");
            Assert.That(string.IsNullOrWhiteSpace(attempt.CaptureSessionId), Is.False,
                "the completed attempt must carry a real capture session identity");
            Assert.That(attempt.Outcome, Is.Not.Null,
                "the completed attempt must carry the production outcome");
            Assert.That(attempt.CommonSignal, Is.Not.Null,
                "the completed attempt must carry the production common signal");
            Assert.That(route.Practice.State,
                Is.Not.EqualTo(PetRobotPracticeState.Ready),
                "the session must leave its initial state after a real attempt");
            Assert.That(route.Practice.LastStorageResult, Is.Not.Null,
                "the production persistence policy must have produced a result");

            // Step 10W v1.0a. The expected canonical World outcome is computed from the
            // PRODUCTION seam over the finalized guided outcome. Before this repair the
            // helper asserted the canonical value equalled route.QuestSession.FixedKind,
            // which hard-coded the very legacy authority Step 10W removed: any case whose
            // legacy spy disagreed with the guided outcome died here, in shared setup,
            // before reaching its own assertions.
            TrustedHybridOutcomeKind expectedCanonical =
                PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(
                    attempt.Outcome.Kind);
            Assert.That(route.Practice.QuestAuthoritativeOutcome,
                Is.EqualTo(expectedCanonical),
                "the canonical World outcome must be the finalized guided outcome " +
                "mapped one-to-one, never the legacy port's reading");

            // The legacy pass is diagnostic only, so what may be asserted about it
            // depends on how the legacy port was told to behave.
            switch (route.QuestSession.Behaviour)
            {
                case QuestHybridSpyBehaviour.ReturnFixedResult:
                    if (expectedCanonical == TrustedHybridOutcomeKind.TechnicalFailure)
                    {
                        Assert.That(route.Practice.QuestHybridResult, Is.Null,
                            "a technical-failure canonical outcome skips the legacy pass");
                        Assert.That(route.Practice.LegacyHybridDiagnosticOutcome, Is.Null);
                    }
                    else
                    {
                        Assert.That(route.Practice.QuestHybridResult, Is.Not.Null,
                            "a returning legacy port must still be recorded as a " +
                            "diagnostic");
                        Assert.That(route.Practice.LegacyHybridDiagnosticOutcome,
                            Is.EqualTo(route.QuestSession.FixedKind),
                            "the legacy diagnostic must report exactly what the port " +
                            "returned");
                        Assert.That(route.QuestSession.LastQuerySessionId,
                            Is.EqualTo(attempt.CaptureSessionId),
                            "the legacy port must have been asked about this attempt");
                    }
                    break;
                default:
                    Assert.That(route.Practice.QuestHybridResult, Is.Null,
                        "a null or throwing legacy pass must leave no hybrid result");
                    Assert.That(route.Practice.LegacyHybridDiagnosticOutcome, Is.Null,
                        "a null or throwing legacy pass must leave no diagnostic kind");
                    break;
            }

            Assert.That(route.Practice.QuestCompleted,
                Is.EqualTo(
                    expectedCanonical == TrustedHybridOutcomeKind.StrictSuccess ||
                    expectedCanonical == TrustedHybridOutcomeKind.LowConfidenceCorrect),
                "quest completion must follow the canonical outcome and nothing else");
            Assert.That(route.Practice.LastStorageResult.Status,
                Is.EqualTo(CandidateStorageStatus.NotRequested),
                "quest mode does not store a personalization candidate, and the real " +
                "policy result says so");

            into.PersistenceCountsReadsAfter = route.Repository.CountsReads;
            into.PersistenceSaveCallsAfter = route.Repository.SaveCalls;
            into.QuestRecognizeCallsAfter = route.QuestSession.RecognizeCalls;
            into.PracticeAttemptCaptureSessionId = attempt.CaptureSessionId;
            into.PracticeAttemptOutcomeKind = attempt.Outcome.Kind.ToString();
            into.PracticeStorageStatus = route.Practice.LastStorageResult.Status.ToString();
            into.PracticeStorageMessage = route.Practice.LastStorageResult.Message;

            // the connected persistence port was reached, whatever the policy decided
            Assert.That(into.PersistenceCountsReadsAfter,
                Is.GreaterThan(into.PersistenceCountsReadsBefore),
                "the production persistence port must be consulted by the real policy");
        }

        /// <summary>Sets one serialized field the Editor would normally set.</summary>
        static void SetPrivateField(object target, string field, object value)
        {
            System.Reflection.FieldInfo info = target.GetType().GetField(
                field,
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.NonPublic);
            Assert.That(info, Is.Not.Null, "serialized field not found: " + field);
            info.SetValue(target, value);
        }

        /// <summary>
        /// Reads every compared value from THE ONE completed attempt and the one route
        /// that produced it. Nothing here reads a second controller, a second classifier
        /// result or a default from an untouched object.
        /// </summary>
        static void CaptureRoute(RouteObservation into, ConnectedPracticeRoute route)
        {
            PracticeRecognitionAttempt attempt = route.Practice.LastAttempt;
            Assert.That(attempt, Is.Not.Null);
            Assert.That(ReferenceEquals(route.Port.LastAttempt, attempt), Is.True);

            into.CoordinatorState = route.Coordinator.State.ToString();
            into.CoordinatorTarget = route.Coordinator.CurrentTarget;
            into.FinalizedCount = route.Coordinator.FinalizedCount;
            into.AdapterSetExpectedWordCalls =
                route.Coordinator.AdapterSetExpectedWordCalls;
            into.AdapterBeginCaptureCalls = route.Coordinator.AdapterBeginCaptureCalls;
            into.AdapterFinalizeCaptureCalls =
                route.Coordinator.AdapterFinalizeCaptureCalls;
            into.AdapterResetCaptureCalls = route.Coordinator.AdapterResetCaptureCalls;
            into.RecordedLines = route.Coordinator.RecordedLines;

            // controller state from the SAME controller that produced this attempt
            into.ControllerState = route.Controller.State.ToString();
            into.PredictedWord = attempt.PredictedWord;
            into.RecognizeCalls = route.Factory.Runtime.RecognizeCalls;

            // common bits ONLY from this attempt's own CommonSignal
            CommonSignClassification common = attempt.CommonSignal;
            Assert.That(common, Is.Not.Null);
            into.DecisionBits = Bits(common.DecisionValue);
            into.GreetingBits = Bits(common.GreetingProbability);
            into.ThankYouBits = Bits(common.ThankYouProbability);
            into.TopClass = common.TopClass;
            into.ExpectedTargetSupport = common.ExpectedTargetSupport;

            // strict and fused outcome ONLY from this same attempt and its Outcome
            GuidedQuestOutcome outcome = attempt.Outcome;
            Assert.That(outcome, Is.Not.Null);
            into.OutcomeKind = outcome.Kind.ToString();
            into.OutcomeExpectedWord = outcome.ExpectedWord;
            into.OutcomePredictedWord = outcome.PredictedWord;
            into.StrictResultAvailable = outcome.StrictResultAvailable;
            into.StrictAccepted = outcome.StrictAccepted;
            into.StrictExpectedWordClassAccept = outcome.StrictExpectedWordClassAccept;
            into.StrictQuestSuccess = outcome.StrictQuestSuccess;
            into.GuidedProgressAllowed = outcome.GuidedProgressAllowed;
            into.OutcomeMessage = outcome.Message;
            into.CommonEvidenceState = outcome.Common.State.ToString();
            into.CommonGrantsSuccess = outcome.Common.GrantsSuccess;
            into.CommonGrantsLearnerError = outcome.Common.GrantsLearnerError;
            into.CommonGrantsVeto = outcome.Common.GrantsVeto;

            // the same session, after the same finalize
            into.PracticeState = route.Practice.State.ToString();
            into.PracticeCurrentWord = route.Practice.CurrentWord;
            into.PracticeCurrentWordComplete = route.Practice.IsCurrentWordComplete;
            into.PracticeAllWordsComplete = route.Practice.AllWordsComplete;
            into.PracticeQuestMode = route.Practice.QuestMode;
            into.PracticeQuestCompleted = route.Practice.QuestCompleted;
            into.PracticeCanRetry = route.Practice.CanRetry;
            into.PracticeAdapterCaptureState =
                route.PracticeAdapter.CaptureState.ToString();
            into.PracticeAdapterSessionId = route.PracticeAdapter.CaptureSessionId;

            // the ACTUAL quest state
            into.QuestMode = route.Practice.QuestMode;
            into.QuestCompleted = route.Practice.QuestCompleted;
            into.QuestAuthoritativeOutcome =
                route.Practice.QuestAuthoritativeOutcome.HasValue
                    ? route.Practice.QuestAuthoritativeOutcome.Value.ToString()
                    : "none";
            into.QuestHybridResultPresent = route.Practice.QuestHybridResult != null;
            into.QuestHybridResultKind = route.Practice.QuestHybridResult == null
                ? "none" : route.Practice.QuestHybridResult.Kind.ToString();
            into.QuestHybridExpectedWord = route.Practice.QuestHybridResult == null
                ? string.Empty : route.Practice.QuestHybridResult.ExpectedWord;

            into.PersistenceSaveCalls = route.Repository.SaveCalls;
            into.PersistenceReloadCalls = route.Repository.ReloadCalls;
            into.PersistenceCountsReads = route.Repository.CountsReads;

            into.PersonalRecognizeCalls = route.Factory.Runtime.RecognizeCalls;
            into.CommonClassifierForwardCalls = route.CommonClassifier.ForwardCalls;
            into.PortFinalizeCalls = route.Port.FinalizeCaptureCalls;
            into.PortResetCalls = route.Port.ResetCaptureCalls;
            into.PortEmittedObservations = route.Port.EmittedObservationCount;
            into.RawFixtureFrames = route.RawFixture.Count;
            into.AttemptIsPortAttempt = ReferenceEquals(route.Port.LastAttempt, attempt);
        }

        /// <summary>Compares every recorded field of two runs, naming each difference.</summary>
        static void AssertRoutesIdentical(RouteObservation off, RouteObservation on)
        {
            Assert.That(on.CoordinatorState, Is.EqualTo(off.CoordinatorState));
            Assert.That(on.CoordinatorTarget, Is.EqualTo(off.CoordinatorTarget));
            Assert.That(on.FinalizedCount, Is.EqualTo(off.FinalizedCount));
            Assert.That(on.AdapterSetExpectedWordCalls,
                Is.EqualTo(off.AdapterSetExpectedWordCalls));
            Assert.That(on.AdapterBeginCaptureCalls,
                Is.EqualTo(off.AdapterBeginCaptureCalls));
            Assert.That(on.AdapterFinalizeCaptureCalls,
                Is.EqualTo(off.AdapterFinalizeCaptureCalls));
            Assert.That(on.AdapterResetCaptureCalls,
                Is.EqualTo(off.AdapterResetCaptureCalls));
            Assert.That(on.RecordedLines, Is.EqualTo(off.RecordedLines));
            Assert.That(on.ControllerState, Is.EqualTo(off.ControllerState));
            Assert.That(on.PredictedWord, Is.EqualTo(off.PredictedWord));
            Assert.That(on.RecognizeCalls, Is.EqualTo(off.RecognizeCalls));

            Assert.That(on.OutcomeKind, Is.EqualTo(off.OutcomeKind));
            Assert.That(on.OutcomeExpectedWord, Is.EqualTo(off.OutcomeExpectedWord));
            Assert.That(on.OutcomePredictedWord, Is.EqualTo(off.OutcomePredictedWord));
            Assert.That(on.StrictResultAvailable, Is.EqualTo(off.StrictResultAvailable));
            Assert.That(on.StrictAccepted, Is.EqualTo(off.StrictAccepted));
            Assert.That(on.StrictExpectedWordClassAccept,
                Is.EqualTo(off.StrictExpectedWordClassAccept));
            Assert.That(on.StrictQuestSuccess, Is.EqualTo(off.StrictQuestSuccess));
            Assert.That(on.GuidedProgressAllowed, Is.EqualTo(off.GuidedProgressAllowed));
            Assert.That(on.OutcomeMessage, Is.EqualTo(off.OutcomeMessage));
            Assert.That(on.CommonEvidenceState, Is.EqualTo(off.CommonEvidenceState));
            Assert.That(on.CommonGrantsSuccess, Is.EqualTo(off.CommonGrantsSuccess));
            Assert.That(on.CommonGrantsLearnerError,
                Is.EqualTo(off.CommonGrantsLearnerError));
            Assert.That(on.CommonGrantsVeto, Is.EqualTo(off.CommonGrantsVeto));

            Assert.That(on.PracticeState, Is.EqualTo(off.PracticeState));
            Assert.That(on.PracticeCurrentWord, Is.EqualTo(off.PracticeCurrentWord));
            Assert.That(on.PracticeCurrentWordComplete,
                Is.EqualTo(off.PracticeCurrentWordComplete));
            Assert.That(on.PracticeAllWordsComplete,
                Is.EqualTo(off.PracticeAllWordsComplete));
            Assert.That(on.PracticeQuestMode, Is.EqualTo(off.PracticeQuestMode));
            Assert.That(on.PracticeQuestCompleted,
                Is.EqualTo(off.PracticeQuestCompleted));
            Assert.That(on.PracticeCanRetry, Is.EqualTo(off.PracticeCanRetry));

            Assert.That(on.PracticeAdapterCaptureState,
                Is.EqualTo(off.PracticeAdapterCaptureState));

            // the ACTUAL quest state produced by the real policy
            Assert.That(on.QuestMode, Is.EqualTo(off.QuestMode));
            Assert.That(on.QuestMode, Is.True);
            Assert.That(on.QuestGeneration, Is.EqualTo(off.QuestGeneration));
            Assert.That(on.QuestGeneration, Is.Not.Zero);
            Assert.That(on.QuestCompleted, Is.EqualTo(off.QuestCompleted));
            Assert.That(on.QuestAuthoritativeOutcome,
                Is.EqualTo(off.QuestAuthoritativeOutcome));
            Assert.That(on.QuestHybridResultPresent,
                Is.EqualTo(off.QuestHybridResultPresent));
            Assert.That(on.QuestHybridResultPresent, Is.True);
            Assert.That(on.QuestHybridResultKind, Is.EqualTo(off.QuestHybridResultKind));
            Assert.That(on.QuestHybridExpectedWord,
                Is.EqualTo(off.QuestHybridExpectedWord));
            Assert.That(on.QuestRecognizeCallsAfter - on.QuestRecognizeCallsBefore,
                Is.EqualTo(off.QuestRecognizeCallsAfter - off.QuestRecognizeCallsBefore));
            Assert.That(on.QuestRecognizeCallsAfter - on.QuestRecognizeCallsBefore,
                Is.EqualTo(1));

            // exactly one of everything, through one route, in both runs
            Assert.That(on.PersonalRecognizeCalls,
                Is.EqualTo(off.PersonalRecognizeCalls));
            Assert.That(on.PersonalRecognizeCalls, Is.EqualTo(1));
            Assert.That(on.CommonClassifierForwardCalls,
                Is.EqualTo(off.CommonClassifierForwardCalls));
            Assert.That(on.CommonClassifierForwardCalls, Is.EqualTo(1));
            Assert.That(on.PortFinalizeCalls, Is.EqualTo(off.PortFinalizeCalls));
            Assert.That(on.PortFinalizeCalls, Is.EqualTo(1));
            Assert.That(on.PortResetCalls, Is.EqualTo(off.PortResetCalls));
            Assert.That(on.PortResetCalls, Is.Zero);
            Assert.That(on.RawFixtureFrames, Is.EqualTo(off.RawFixtureFrames));
            Assert.That(on.PortEmittedObservations,
                Is.EqualTo(off.PortEmittedObservations));
            Assert.That(on.PortEmittedObservations, Is.EqualTo(on.RawFixtureFrames));
            Assert.That(on.AttemptIsPortAttempt, Is.True);
            Assert.That(off.AttemptIsPortAttempt, Is.True);
            Assert.That(on.AdapterSetExpectedWordDelegations,
                Is.EqualTo(off.AdapterSetExpectedWordDelegations));
            Assert.That(on.AdapterSetExpectedWordDelegations, Is.GreaterThan(0));
            Assert.That(on.DelegatedTarget, Is.EqualTo(off.DelegatedTarget));
            Assert.That(on.PracticeAttemptOutcomeKind,
                Is.EqualTo(off.PracticeAttemptOutcomeKind));
            Assert.That(on.PracticeStorageStatus,
                Is.EqualTo(off.PracticeStorageStatus));
            Assert.That(on.PracticeStorageMessage,
                Is.EqualTo(off.PracticeStorageMessage));

            Assert.That(on.PersistenceSaveCalls, Is.EqualTo(off.PersistenceSaveCalls));
            Assert.That(on.PersistenceReloadCalls,
                Is.EqualTo(off.PersistenceReloadCalls));
            Assert.That(on.PersistenceCountsReads,
                Is.EqualTo(off.PersistenceCountsReads));
            Assert.That(on.PersistenceSaveCallsAfter - on.PersistenceSaveCallsBefore,
                Is.EqualTo(off.PersistenceSaveCallsAfter -
                           off.PersistenceSaveCallsBefore));
            Assert.That(on.PersistenceCountsReadsAfter - on.PersistenceCountsReadsBefore,
                Is.EqualTo(off.PersistenceCountsReadsAfter -
                           off.PersistenceCountsReadsBefore));

            Assert.That(on.DecisionBits, Is.EqualTo(off.DecisionBits));
            Assert.That(on.GreetingBits, Is.EqualTo(off.GreetingBits));
            Assert.That(on.ThankYouBits, Is.EqualTo(off.ThankYouBits));
            Assert.That(on.TopClass, Is.EqualTo(off.TopClass));
            Assert.That(on.ExpectedTargetSupport, Is.EqualTo(off.ExpectedTargetSupport));
        }

        /// <summary>
        /// Runs the whole real route once: the real controller and coordinator, and the
        /// connected real harness, production adapter and practice session driven through
        /// a complete StartExpression/FinishExpression attempt over the same raw fixture.
        /// </summary>
        RouteObservation RunFullRoute(string label)
        {
            ConnectedPracticeRoute route = BuildConnectedPracticeRoute(label);
            var observation = new RouteObservation();
            DriveRealPracticeAttempt(route, observation);
            CaptureRoute(observation, route);
            // Observed before disposal, which clears it. Disposal itself stays exactly
            // where it was, so cleanup remains deterministic.
            observation.ObservedTrace = Common2DDiagnosticTraceBuffer.Active;
            route.Dispose();
            return observation;
        }

        /// <summary>
        /// One run of that same unified route, clean or with exactly one collection fault
        /// injected. Everything below is read from the one completed attempt, so a fault
        /// case and the clean baseline differ in nothing but the injected fault.
        /// </summary>
        sealed class StageBaseline
        {
            public ulong DecisionBits;
            public ulong GreetingBits;
            public ulong ThankYouBits;
            public string TopClass;
            public bool ExpectedTargetSupport;
            public string ControllerState;
            public string CoordinatorState;
            public int FinalizedCount;
            public int AdapterFinalizeCaptureCalls;
            public int AdapterResetCaptureCalls;
            public int PersonalRecognizeCalls;
            public int CommonForwardCalls;
            public int QuestRecognizeCalls;
            public string OutcomeKind;
            public bool StrictAccepted;
            public bool StrictExpectedWordClassAccept;
            public bool StrictQuestSuccess;
            public bool GuidedProgressAllowed;
            public string OutcomeMessage;
            public string CommonEvidenceState;

            // the real practice consumer and its actual quest state
            public string PracticeState;
            public string PracticeCurrentWord;
            public bool PracticeCurrentWordComplete;
            public bool PracticeAllWordsComplete;
            public bool PracticeCanRetry;
            public bool QuestMode;
            public long QuestGeneration;
            public string QuestTokenId;
            public bool QuestCompleted;
            public string QuestAuthoritativeOutcome;
            public bool QuestHybridResultPresent;
            public string QuestHybridResultKind;
            public string PracticeAttemptOutcomeKind;
            public string PracticeStorageStatus;
            public string PracticeStorageMessage;
            public int PersistenceSaveCallsDelta;
            public int PersistenceCountsReadsDelta;
            public int RawFixtureFrames;
            public int PortEmittedObservations;

            public bool CaptureFailed;
            public Common2DTraceFailureReason FirstFailureReason;
            public Common2DCollectionFaultInjection FirstFailureStage;
            public Common2DDiagnosticTrace TraceAfterFailure;
            public string Marker;
        }

        StageBaseline RunStageRoute(
            FrozenCommonSignClassifier classifier, string label,
            Common2DCollectionFaultInjection stage)
        {
            Common2DDiagnosticTraceBuffer.ResetAll();
            Common2DDiagnosticTraceBuffer.TryArm(owner);
            Common2DDiagnosticTraceBuffer.CollectionFaultInjection = stage;

            ConnectedPracticeRoute route = BuildConnectedPracticeRoute(label);
            var observation = new RouteObservation();
            DriveRealPracticeAttempt(route, observation);
            CaptureRoute(observation, route);

            var observed = new StageBaseline
            {
                DecisionBits = observation.DecisionBits,
                GreetingBits = observation.GreetingBits,
                ThankYouBits = observation.ThankYouBits,
                TopClass = observation.TopClass,
                ExpectedTargetSupport = observation.ExpectedTargetSupport,
                ControllerState = observation.ControllerState,
                CoordinatorState = observation.CoordinatorState,
                FinalizedCount = observation.FinalizedCount,
                AdapterFinalizeCaptureCalls = observation.AdapterFinalizeCaptureCalls,
                AdapterResetCaptureCalls = observation.AdapterResetCaptureCalls,
                PersonalRecognizeCalls = observation.PersonalRecognizeCalls,
                CommonForwardCalls = observation.CommonClassifierForwardCalls,
                QuestRecognizeCalls = observation.QuestRecognizeCallsAfter -
                                      observation.QuestRecognizeCallsBefore,
                OutcomeKind = observation.OutcomeKind,
                StrictAccepted = observation.StrictAccepted,
                StrictExpectedWordClassAccept =
                    observation.StrictExpectedWordClassAccept,
                StrictQuestSuccess = observation.StrictQuestSuccess,
                GuidedProgressAllowed = observation.GuidedProgressAllowed,
                OutcomeMessage = observation.OutcomeMessage,
                CommonEvidenceState = observation.CommonEvidenceState,

                PracticeState = observation.PracticeState,
                PracticeCurrentWord = observation.PracticeCurrentWord,
                PracticeCurrentWordComplete = observation.PracticeCurrentWordComplete,
                PracticeAllWordsComplete = observation.PracticeAllWordsComplete,
                PracticeCanRetry = observation.PracticeCanRetry,
                QuestMode = observation.QuestMode,
                QuestGeneration = observation.QuestGeneration,
                QuestTokenId = observation.QuestTokenId,
                QuestCompleted = observation.QuestCompleted,
                QuestAuthoritativeOutcome = observation.QuestAuthoritativeOutcome,
                QuestHybridResultPresent = observation.QuestHybridResultPresent,
                QuestHybridResultKind = observation.QuestHybridResultKind,
                PracticeAttemptOutcomeKind = observation.PracticeAttemptOutcomeKind,
                PracticeStorageStatus = observation.PracticeStorageStatus,
                PracticeStorageMessage = observation.PracticeStorageMessage,
                PersistenceSaveCallsDelta = observation.PersistenceSaveCallsAfter -
                                            observation.PersistenceSaveCallsBefore,
                PersistenceCountsReadsDelta = observation.PersistenceCountsReadsAfter -
                                              observation.PersistenceCountsReadsBefore,
                RawFixtureFrames = observation.RawFixtureFrames,
                PortEmittedObservations = observation.PortEmittedObservations,

                CaptureFailed = Common2DDiagnosticTraceBuffer.CaptureFailed,
                FirstFailureReason = Common2DDiagnosticTraceBuffer.FirstCollectionFailure,
                FirstFailureStage =
                    Common2DDiagnosticTraceBuffer.FirstCollectionFailureStage,
                TraceAfterFailure = Common2DDiagnosticTraceBuffer.Active,
                Marker = Common2DDiagnosticLogMarker.BuildFailure(
                    null, Common2DTraceStatus.Failed,
                    Common2DDiagnosticTraceBuffer.FirstCollectionFailure),
            };

            Common2DDiagnosticTraceBuffer.CollectionFaultInjection =
                Common2DCollectionFaultInjection.None;
            route.Dispose();
            Assert.That(classifier.ModelAvailable, Is.True);
            return observed;
        }

        void AssertIsolatedNegative(
            FrozenCommonSignClassifier classifier, string sessionId,
            Func<Common2DDiagnosticTrace, Common2DDiagnosticTrace> mutate,
            Common2DTraceFailureReason expected)
        {
            Common2DDiagnosticTraceBuffer.ResetAll();
            var caseOwner = new object();
            Assert.That(Common2DDiagnosticTraceBuffer.TryArm(caseOwner),
                Is.EqualTo(Common2DArmResult.Armed));
            CommonTwoDimensionalCapture caseCapture;
            RunRealCapture(classifier, sessionId, FrozenCommonSignModel.ClassThankYou,
                BuildObservations(12, false), out caseCapture);
            Common2DDiagnosticTrace source = Common2DDiagnosticTraceBuffer.Active;
            Assert.That(Common2DDiagnosticTraceBuffer.MarkTerminalAccepted(caseOwner),
                Is.True);

            Common2DTraceFailureReason baseline;
            Assert.That(source.Validate(out baseline), Is.True,
                "the source trace must be valid before one field is broken: " + sessionId);

            string caseRoot = Path.Combine(scratchRoot, "isolated", sessionId);
            Common2DDiagnosticExportResult result =
                Common2DDiagnosticTraceExporter.ExportToScratchRoot(mutate(source), caseRoot);
            Assert.That(result.Status, Is.EqualTo(Common2DTraceStatus.Failed), sessionId);
            Assert.That(result.FailureReason, Is.EqualTo(expected), sessionId);
            Assert.That(result.FailureReason,
                Is.Not.EqualTo(Common2DTraceFailureReason.AlreadyCommitted),
                "the duplicate gate must not be what rejected " + sessionId);
            Assert.That(Directory.Exists(caseRoot), Is.False,
                "no root may be created for " + sessionId);
            Common2DDiagnosticTraceBuffer.ResetAll();
        }


        static bool[] ReappearanceOf(RawLandmarkObservationSet set)
        {
            var values = new bool[set.LandmarkCount];
            for (int i = 0; i < values.Length; i++) values[i] = set.IsReappearance(i);
            return values;
        }

        static bool[] BoundaryOf(RawLandmarkObservationSet set)
        {
            var values = new bool[set.LandmarkCount];
            for (int i = 0; i < values.Length; i++)
                values[i] = set.IsStrongDepthBoundary(i);
            return values;
        }

        /// <summary>Compares every recorded geometry field with its source geometry.</summary>
        static void AssertGeometryMatches(
            Common2DGeometryRecord record, Step9DInferenceFrameGeometry origin, string at)
        {
            Assert.That(record.Present, Is.True, "geometry present " + at);
            Assert.That(record.ObservedWebcamWidth,
                Is.EqualTo(origin.ObservedWebcamWidth), "webcam width " + at);
            Assert.That(record.ObservedWebcamHeight,
                Is.EqualTo(origin.ObservedWebcamHeight), "webcam height " + at);
            Assert.That(record.InferenceTextureWidth,
                Is.EqualTo(origin.InferenceTextureWidth), "texture width " + at);
            Assert.That(record.InferenceTextureHeight,
                Is.EqualTo(origin.InferenceTextureHeight), "texture height " + at);
            Assert.That(record.InferenceImageWidth,
                Is.EqualTo(origin.InferenceImageWidth), "image width " + at);
            Assert.That(record.InferenceImageHeight,
                Is.EqualTo(origin.InferenceImageHeight), "image height " + at);
            Assert.That(record.ImageFormat,
                Is.EqualTo(origin.ImageFormat), "image format " + at);
        }


        /// <summary>
        /// The independent hard-expected member schema. It is written out here, in the
        /// test, from the v1.0f1 payload contract - NOT read from
        /// Common2DDiagnosticTraceExporter.CanonicalMemberSchema. If the production table
        /// silently changed, comparing it with itself would prove nothing; comparing it
        /// with this one is what makes the check real.
        ///
        /// Columns: name, group, dtype, shape template.
        /// </summary>
        static readonly string[][] IndependentMemberSchema =
        {
            new[] {"raw_timestamps.bin", "raw_observation_sequence", "<f8", "[N]"},
            new[] {"raw_pose_coordinates.bin", "raw_observation_sequence", "<f8", "[N,33,3]"},
            new[] {"raw_left_hand_coordinates.bin", "raw_observation_sequence", "<f8", "[N,21,3]"},
            new[] {"raw_right_hand_coordinates.bin", "raw_observation_sequence", "<f8", "[N,21,3]"},
            new[] {"raw_pose_valid_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,33]"},
            new[] {"raw_left_hand_valid_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_right_hand_valid_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_pose_reappearance_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,33]"},
            new[] {"raw_left_hand_reappearance_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_right_hand_reappearance_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_pose_strong_depth_boundary_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,33]"},
            new[] {"raw_left_hand_strong_depth_boundary_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_right_hand_strong_depth_boundary_mask.bin", "raw_observation_sequence", "packed_bool_lsb_first", "[N,21]"},
            new[] {"raw_convention.json", "raw_observation_sequence", "utf8_json", "[]"},
            new[] {"raw_geometry.json", "raw_observation_sequence", "utf8_json", "[]"},
            new[] {"normalized_coordinates.bin", "normalized_common2d_coordinates", "<f8", "[T,16,3]"},
            new[] {"input_mask.bin", "common2d_input_mask", "packed_bool_lsb_first", "[T,16]"},
            new[] {"resampled_coordinates.bin", "resampled_common2d_coordinates", "<f8", "[32,16,2]"},
            new[] {"resampled_mask.bin", "resampled_common2d_mask", "packed_bool_lsb_first", "[32,16]"},
            new[] {"feature_vector.bin", "feature_vector", "<f8", "[1536]"},
            new[] {"standardized_feature_vector.bin", "standardized_feature_vector", "<f8", "[1536]"},
            new[] {"pca_projection.bin", "pca_projection", "<f8", "[4]"},
            new[] {"lr_component_contributions.bin", "lr_component_contributions", "<f8", "[4]"},
            new[] {"lr_scalars.bin", "lr_intercept_decision_probabilities", "<f8", "[4]"},
        };

        const int IndependentMemberCount = 24;

        /// <summary>Resolves one independent shape template against the actual counts.</summary>
        static string ResolveShape(string template, int rawFrames, int normalizedFrames)
        {
            return template
                .Replace("N", rawFrames.ToString(CultureInfo.InvariantCulture))
                .Replace("T", normalizedFrames.ToString(CultureInfo.InvariantCulture));
        }

        static long ShapeProduct(string shape)
        {
            string body = shape.Substring(1, shape.Length - 2);
            if (body.Length == 0) return 1;
            string[] parts = body.Split(',');
            long product = 1;
            for (int i = 0; i < parts.Length; i++)
                product *= long.Parse(parts[i], CultureInfo.InvariantCulture);
            return product;
        }

        void AssertSchemaMutationRejected(
            FrozenCommonSignClassifier classifier, string sessionId,
            Common2DMemberSchemaMutation mutation, int memberIndex)
        {
            Common2DDiagnosticTraceBuffer.ResetAll();
            var caseOwner = new object();
            Assert.That(Common2DDiagnosticTraceBuffer.TryArm(caseOwner),
                Is.EqualTo(Common2DArmResult.Armed));
            CommonTwoDimensionalCapture caseCapture;
            RunRealCapture(classifier, sessionId, FrozenCommonSignModel.ClassThankYou,
                BuildObservations(12, false), out caseCapture);
            Common2DDiagnosticTrace source = Common2DDiagnosticTraceBuffer.Active;
            Assert.That(Common2DDiagnosticTraceBuffer.MarkTerminalAccepted(caseOwner),
                Is.True);

            string caseRoot = Path.Combine(scratchRoot, "schema", sessionId);
            Common2DDiagnosticTraceExporter.SchemaMutation = mutation;
            Common2DDiagnosticTraceExporter.SchemaMutationMemberIndex = memberIndex;
            try
            {
                Common2DDiagnosticExportResult result =
                    Common2DDiagnosticTraceExporter.ExportToScratchRoot(source, caseRoot);
                Assert.That(result.Status, Is.EqualTo(Common2DTraceStatus.Failed),
                    sessionId);
                Assert.That(result.FailureReason,
                    Is.EqualTo(Common2DTraceFailureReason.InvalidPayloadShape), sessionId);
                Assert.That(Directory.Exists(caseRoot), Is.False,
                    "the schema check must run before the root is created: " + sessionId);
            }
            finally
            {
                Common2DDiagnosticTraceExporter.SchemaMutation =
                    Common2DMemberSchemaMutation.None;
                Common2DDiagnosticTraceExporter.SchemaMutationMemberIndex = 0;
            }
            Common2DDiagnosticTraceBuffer.ResetAll();
        }


        static double[] Flatten(
            Common2DDiagnosticTrace trace, Func<Common2DRawObservationRecord, double[]> pick)
        {
            var parts = new List<double>();
            for (int i = 0; i < trace.RawObservations.Count; i++)
                parts.AddRange(pick(trace.RawObservations[i]));
            return parts.ToArray();
        }

        static bool[] FlattenBools(
            Common2DDiagnosticTrace trace, Func<Common2DRawObservationRecord, bool[]> pick)
        {
            var parts = new List<bool>();
            for (int i = 0; i < trace.RawObservations.Count; i++)
                parts.AddRange(pick(trace.RawObservations[i]));
            return parts.ToArray();
        }


        /// <summary>One run of the real route, clean or with one collection fault.</summary>

        /// <summary>Compares one frame's convention object field by field.</summary>
        static void AssertConventionJson(
            JsonValue entry, Common2DConventionRecord record, string at)
        {
            Assert.That(entry.Member("id").AsString(at), Is.EqualTo(record.Id), at + " id");
            Assert.That(entry.Member("coordinateSpace").AsString(at),
                Is.EqualTo(record.CoordinateSpace), at + " coordinateSpace");
            Assert.That(entry.Member("axisAndDepth").AsString(at),
                Is.EqualTo(record.AxisAndDepth), at + " axisAndDepth");
            Assert.That(entry.Member("handedness").AsString(at),
                Is.EqualTo(record.Handedness), at + " handedness");
            Assert.That(entry.Member("mirror").AsString(at),
                Is.EqualTo(record.Mirror), at + " mirror");
        }

        /// <summary>Compares one frame's geometry object field by field.</summary>
        static void AssertGeometryJson(
            JsonValue entry, Common2DGeometryRecord record, string at)
        {
            Assert.That(entry.Member("present").AsBoolean(at),
                Is.EqualTo(record.Present), at + " present");
            Assert.That(entry.Member("provenance").AsString(at),
                Is.EqualTo(record.Provenance), at + " provenance");
            Assert.That(entry.Member("observedWebcamWidth").AsInteger(at),
                Is.EqualTo((long)record.ObservedWebcamWidth), at + " observedWebcamWidth");
            Assert.That(entry.Member("observedWebcamHeight").AsInteger(at),
                Is.EqualTo((long)record.ObservedWebcamHeight),
                at + " observedWebcamHeight");
            Assert.That(entry.Member("observedVideoRotationAngleOrNull").AsString(at),
                Is.EqualTo(record.ObservedVideoRotationAngleOrNull),
                at + " observedVideoRotationAngleOrNull");
            Assert.That(entry.Member("observedVideoVerticallyMirroredOrNull").AsString(at),
                Is.EqualTo(record.ObservedVideoVerticallyMirroredOrNull),
                at + " observedVideoVerticallyMirroredOrNull");
            Assert.That(entry.Member("inferenceTextureWidth").AsInteger(at),
                Is.EqualTo((long)record.InferenceTextureWidth),
                at + " inferenceTextureWidth");
            Assert.That(entry.Member("inferenceTextureHeight").AsInteger(at),
                Is.EqualTo((long)record.InferenceTextureHeight),
                at + " inferenceTextureHeight");
            Assert.That(entry.Member("inferenceImageWidth").AsInteger(at),
                Is.EqualTo((long)record.InferenceImageWidth), at + " inferenceImageWidth");
            Assert.That(entry.Member("inferenceImageHeight").AsInteger(at),
                Is.EqualTo((long)record.InferenceImageHeight),
                at + " inferenceImageHeight");
            Assert.That(entry.Member("imageFormat").AsString(at),
                Is.EqualTo(record.ImageFormat), at + " imageFormat");
            Assert.That(entry.Member("preInferenceHorizontalFlip").AsString(at),
                Is.EqualTo(record.PreInferenceHorizontalFlip),
                at + " preInferenceHorizontalFlip");
            Assert.That(entry.Member("preInferenceVerticalFlip").AsString(at),
                Is.EqualTo(record.PreInferenceVerticalFlip),
                at + " preInferenceVerticalFlip");
            Assert.That(entry.Member("preInferenceRotationDegreesOrNull").AsString(at),
                Is.EqualTo(record.PreInferenceRotationDegreesOrNull),
                at + " preInferenceRotationDegreesOrNull");
            Assert.That(entry.Member("crop").AsString(at),
                Is.EqualTo(record.Crop), at + " crop");
            Assert.That(entry.Member("resize").AsString(at),
                Is.EqualTo(record.Resize), at + " resize");
            Assert.That(entry.Member("letterbox").AsString(at),
                Is.EqualTo(record.Letterbox), at + " letterbox");
            Assert.That(entry.Member("pixelBufferRowOrder").AsString(at),
                Is.EqualTo(record.PixelBufferRowOrder), at + " pixelBufferRowOrder");
            Assert.That(entry.Member("providerFrameIndexOrNull").AsString(at),
                Is.EqualTo(record.ProviderFrameIndexOrNull),
                at + " providerFrameIndexOrNull");
            Assert.That(entry.Member("mediaPipeTimestampMs").AsInteger(at),
                Is.EqualTo(record.MediaPipeTimestampMs), at + " mediaPipeTimestampMs");
            Assert.That(entry.Member("unityFrameCountOrNull").AsString(at),
                Is.EqualTo(record.UnityFrameCountOrNull), at + " unityFrameCountOrNull");
        }

        /// <summary>Compares one frame's agreement object field by field.</summary>
        static void AssertAgreementJson(
            JsonValue entry, Common2DGeometryAgreementRecord record, string at)
        {
            Assert.That(entry.Member("present").AsBoolean(at),
                Is.EqualTo(record.Present), at + " present");
            Assert.That(entry.Member("bothAvailable").AsBoolean(at),
                Is.EqualTo(record.BothAvailable), at + " bothAvailable");
            Assert.That(entry.Member("fullContractCompatible").AsBoolean(at),
                Is.EqualTo(record.FullContractCompatible),
                at + " fullContractCompatible");
            Assert.That(entry.Member("firstMismatchDescription").AsString(at),
                Is.EqualTo(record.FirstMismatchDescription),
                at + " firstMismatchDescription");
            Assert.That(entry.Member("handPoseTimestampSkewMs").AsString(at),
                Is.EqualTo(record.HandPoseTimestampSkewMs),
                at + " handPoseTimestampSkewMs");
            Assert.That(entry.Member("handResultStatus").AsString(at),
                Is.EqualTo(record.HandResultStatus), at + " handResultStatus");
            Assert.That(entry.Member("poseResultStatus").AsString(at),
                Is.EqualTo(record.PoseResultStatus), at + " poseResultStatus");
            Assert.That(entry.Member("handResultDiagnosticCode").AsString(at),
                Is.EqualTo(record.HandResultDiagnosticCode),
                at + " handResultDiagnosticCode");
            Assert.That(entry.Member("poseResultDiagnosticCode").AsString(at),
                Is.EqualTo(record.PoseResultDiagnosticCode),
                at + " poseResultDiagnosticCode");
        }
        // ------------------------------------------------------------------ tests
        // ------------------------------------------------------------------------
        // Step 10T: the World guided-progression policy.
        //
        // These drive the REAL InWorldGreetingNpcDialogueController.ApplyOutcome switch
        // through its private body, the same way the f4 lifecycle tests drive the real
        // harness callbacks, so the policy under test is the shipped one and not a copy.
        // They assert World state only; no recognition, threshold or model behaviour is
        // touched, and LowConfidenceCorrect is never treated as StrictSuccess.
        // ------------------------------------------------------------------------

        InWorldGreetingNpcDialogueController BuildWorldDialogueController(string label)
        {
            var host = new GameObject("step10t-world-" + label);
            worldControllerObjects.Add(host);
            return host.AddComponent<InWorldGreetingNpcDialogueController>();
        }

        static void ApplyRealWorldOutcome(
            InWorldGreetingNpcDialogueController controller, TrustedHybridOutcomeKind kind)
        {
            System.Reflection.MethodInfo method = controller.GetType().GetMethod(
                "ApplyOutcome",
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.DeclaredOnly);
            Assert.That(method, Is.Not.Null,
                "the World controller must declare a private ApplyOutcome");
            method.Invoke(controller, new object[] { kind });
        }

        [Test]
        public void WorldLowConfidenceCorrect_GrantsItemAndCompletesQuestWithoutRelabelling()
        {
            InWorldGreetingNpcDialogueController controller =
                BuildWorldDialogueController("low-confidence");
            ApplyRealWorldOutcome(
                controller, TrustedHybridOutcomeKind.LowConfidenceCorrect);

            Assert.That(controller.LastOutcome,
                Is.EqualTo(InWorldDialogueOutcome.LowConfidenceCorrect),
                "the outcome label must stay LowConfidenceCorrect");
            Assert.That(controller.LastOutcome,
                Is.Not.EqualTo(InWorldDialogueOutcome.StrictSuccess),
                "guided progression must never relabel this as StrictSuccess");
            Assert.That(controller.ItemGranted, Is.True,
                "guided progression must grant the quest item");
            Assert.That(controller.State,
                Is.EqualTo(InWorldDialogueState.QuestComplete),
                "the World quest must complete rather than wait on a retry");
            Assert.That(controller.CanRetry, Is.False,
                "a completed quest must not offer retry");
        }

        [Test]
        public void WorldLowConfidenceCorrectFeedback_IsTruthfulAndAsksForNoRetry()
        {
            InWorldGreetingNpcDialogueController controller =
                BuildWorldDialogueController("low-confidence-text");
            ApplyRealWorldOutcome(
                controller, TrustedHybridOutcomeKind.LowConfidenceCorrect);

            string feedback = controller.LastFeedbackText;
            Assert.That(feedback, Is.Not.Empty);
            Assert.That(feedback.Contains("R\uc744"), Is.False,
                "the feedback must not instruct the learner to press R");
            Assert.That(feedback.IndexOf("Strict", StringComparison.OrdinalIgnoreCase),
                Is.LessThan(0), "the feedback must not claim strict success");
            Assert.That(feedback.Contains("\uc54c\uc544\ubd24\uc5b4"), Is.True,
                "the feedback must say the greeting was recognised");
            Assert.That(feedback.Contains("\uc815\ud655\ub3c4 \uae30\uc900"), Is.True,
                "the feedback must say the accuracy bar was not reached");
        }

        [Test]
        public void WorldStrictSuccess_StillGrantsItemAndCompletesQuest()
        {
            InWorldGreetingNpcDialogueController controller =
                BuildWorldDialogueController("strict");
            ApplyRealWorldOutcome(controller, TrustedHybridOutcomeKind.StrictSuccess);

            Assert.That(controller.LastOutcome,
                Is.EqualTo(InWorldDialogueOutcome.StrictSuccess));
            Assert.That(controller.ItemGranted, Is.True);
            Assert.That(controller.State,
                Is.EqualTo(InWorldDialogueState.QuestComplete));
        }

        [Test]
        public void WorldWrongWord_GrantsNothingAndStaysInFeedback()
        {
            InWorldGreetingNpcDialogueController controller =
                BuildWorldDialogueController("wrong-word");
            ApplyRealWorldOutcome(controller, TrustedHybridOutcomeKind.WrongWord);

            Assert.That(controller.LastOutcome,
                Is.EqualTo(InWorldDialogueOutcome.WrongWord));
            Assert.That(controller.ItemGranted, Is.False,
                "a wrong word must never grant the quest item");
            Assert.That(controller.State, Is.EqualTo(InWorldDialogueState.Feedback));
            Assert.That(controller.CanRetry, Is.True,
                "a wrong word must remain retryable");
        }

        [Test]
        public void WorldTechnicalFailure_GrantsNothingAndStaysInFeedback()
        {
            InWorldGreetingNpcDialogueController controller =
                BuildWorldDialogueController("technical-failure");
            ApplyRealWorldOutcome(controller, TrustedHybridOutcomeKind.TechnicalFailure);

            Assert.That(controller.LastOutcome,
                Is.EqualTo(InWorldDialogueOutcome.TechnicalFailure));
            Assert.That(controller.ItemGranted, Is.False,
                "a technical failure must never grant the quest item");
            Assert.That(controller.State, Is.EqualTo(InWorldDialogueState.Feedback));
            Assert.That(controller.CanRetry, Is.True,
                "a technical failure must remain retryable");
        }

        [Test]
        public void EvaluatorInvariant_LowConfidenceCorrectStillAllowsGuidedProgress()
        {
            GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Completed,
                true,
                FrozenCommonSignModel.ClassGreeting,
                FrozenCommonSignModel.ClassGreeting,
                false,
                false,
                false,
                false,
                false);

            Assert.That(outcome, Is.Not.Null);
            Assert.That(outcome.Kind,
                Is.EqualTo(GuidedQuestOutcomeKind.LowConfidenceCorrect),
                "right word with strictQuestSuccess false is LowConfidenceCorrect");
            Assert.That(outcome.GuidedProgressAllowed, Is.True,
                "the evaluator already allows guided progress for this outcome");
            Assert.That(outcome.StrictQuestSuccess, Is.False,
                "guided progress must not imply strict quest success");
            Assert.That(outcome.StrictAccepted, Is.False);
        }

        [Test]
        public void SessionInvariant_StrictSuccessAndLowConfidenceCorrectBothCompleteQuest()
        {
            // Step 10W v1.0a. This used to grep PetRobotSignPracticeSession.cs for a
            // source line that branched the World on the LEGACY hybrid kind. Step 10W
            // deleted that line on purpose, so the old test asserted the presence of the
            // very defect the repair removed. The literal is not repeated here, because a
            // substring was never the invariant in the first place: the invariant is that
            // StrictSuccess and LowConfidenceCorrect both complete the quest and neither
            // is relabelled. That is now asserted through the production API and a real
            // driven attempt instead of through source text.
            Assert.That(PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(
                GuidedQuestOutcomeKind.StrictSuccess),
                Is.EqualTo(TrustedHybridOutcomeKind.StrictSuccess));
            Assert.That(PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(
                GuidedQuestOutcomeKind.LowConfidenceCorrect),
                Is.EqualTo(TrustedHybridOutcomeKind.LowConfidenceCorrect),
                "LowConfidenceCorrect must never be relabelled as StrictSuccess");

            foreach (PersonalRoutingSpyMode mode in new[]
                     {
                         PersonalRoutingSpyMode.StrictSuccess,
                         PersonalRoutingSpyMode.LowConfidenceCorrect,
                     })
            {
                ConnectedPracticeRoute route = BuildConnectedPracticeRoute(
                    "invariant-" + mode);
                try
                {
                    route.Factory.Runtime.Mode = mode;
                    DriveRealPracticeAttempt(route, new RouteObservation());

                    GuidedQuestOutcomeKind guided =
                        route.Practice.LastAttempt.Outcome.Kind;
                    Assert.That(guided, Is.EqualTo(
                        mode == PersonalRoutingSpyMode.StrictSuccess
                            ? GuidedQuestOutcomeKind.StrictSuccess
                            : GuidedQuestOutcomeKind.LowConfidenceCorrect),
                        "the fixture mode must produce the guided outcome it names");
                    Assert.That(route.Practice.QuestCompleted, Is.True,
                        "both StrictSuccess and LowConfidenceCorrect must complete the " +
                        "quest");
                    Assert.That(route.Practice.QuestAuthoritativeOutcome,
                        Is.EqualTo(
                            PetRobotSignPracticeSession
                                .MapGuidedToCanonicalWorldOutcome(guided)));
                }
                finally { route.Dispose(); }
            }
        }

        [Test]
        public void DisabledByDefault_NoBufferAllocatedAndNoExportWritten()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.False);
            Assert.That(Common2DDiagnosticTraceBuffer.Status,
                Is.EqualTo(Common2DTraceStatus.Disabled));
            Common2DDiagnosticTraceBuffer.ResetCounters();

            CommonTwoDimensionalCapture capture;
            CommonSignClassification classification = RunRealCapture(
                classifier, "disabled-default", FrozenCommonSignModel.ClassThankYou,
                BuildObservations(14, false), out capture);

            Assert.That(classification.Eligible, Is.True,
                "the ordinary real path must still produce its result while disabled");
            Assert.That(capture.FrameCount, Is.EqualTo(14));
            Assert.That(Common2DDiagnosticTraceBuffer.DiagnosticAllocationCount, Is.Zero);
            Assert.That(Common2DDiagnosticTraceBuffer.DiagnosticArrayCopyCount, Is.Zero);
            Assert.That(Common2DDiagnosticTraceBuffer.GuardRejectionCount,
                Is.GreaterThan(0), "the ownership guard must actually be consulted");
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null);
            Assert.That(Directory.Exists(scratchRoot), Is.False);
        }

        [Test]
        public void EnabledOnlyUnderExplicitEditorValidationOptIn()
        {
            // The named behaviour is the DEVELOPER HARNESS opt-in route, so this test
            // drives DeveloperCommonSignalTwoTargetValidationHarness itself. Direct
            // TryArm calls appear only where the harness is not the subject: proving a
            // foreign owner cannot steal or release the harness's arm.
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            var source = new ObservationSourceSpy { IsBound = true };
            var factory = new RuntimeFactorySpy();
            var controller = new PersonalLiveRecognitionController(source, factory);
            Assert.That(controller.Initialize(
                BuildPaths(Path.Combine(scratchRoot, "runtime"))), Is.True);

            // 1. default off: ordinary controller use with no opt-in traces nothing
            Common2DDiagnosticTraceBuffer.ResetCounters();
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.False,
                "diagnostics must be off before any developer opt-in");
            controller.SetExpectedWord(FrozenCommonSignModel.ClassGreeting);
            controller.BeginCapture("optin-off");
            foreach (RawMediaPipeObservation observation in BuildObservations(12, false))
                source.Emit(observation);
            Assert.That(controller.FinalizeCapture(), Is.True);
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null);
            Assert.That(Common2DDiagnosticTraceBuffer.DiagnosticAllocationCount, Is.Zero);

            // 2. the real developer harness component is the only thing that arms it
            var harnessObject = new GameObject("f2c-developer-harness");
            DeveloperCommonSignalTwoTargetValidationHarness harness =
                harnessObject.AddComponent<
                    DeveloperCommonSignalTwoTargetValidationHarness>();
            Assert.That(harness.Common2DDiagnosticTraceEnabled, Is.False,
                "the harness must not arm anything on construction");

            Common2DArmResult armed = harness.EnableCommon2DDiagnosticTrace();
            Assert.That(armed, Is.EqualTo(Common2DArmResult.Armed));
            Assert.That(harness.LastArmResult, Is.EqualTo(Common2DArmResult.Armed));
            Assert.That(harness.Common2DDiagnosticTraceEnabled, Is.True);
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.True);
            Assert.That(ReferenceEquals(
                Common2DDiagnosticTraceBuffer.Owner, harness), Is.True,
                "the harness component itself must own the arm");

            // 3. the same harness asking again is idempotent, not a second arm
            Assert.That(harness.EnableCommon2DDiagnosticTrace(),
                Is.EqualTo(Common2DArmResult.AlreadyOwned));

            // 4. a foreign owner cannot steal the harness's arm
            var foreignOwner = new object();
            Assert.That(Common2DDiagnosticTraceBuffer.TryArm(foreignOwner),
                Is.EqualTo(Common2DArmResult.RejectedOtherOwnerLive));
            Assert.That(Common2DDiagnosticTraceBuffer.TryArm(null),
                Is.EqualTo(Common2DArmResult.RejectedNullOwner));
            Assert.That(ReferenceEquals(
                Common2DDiagnosticTraceBuffer.Owner, harness), Is.True);
            Assert.That(Common2DDiagnosticTraceBuffer.ReleaseOwner(foreignOwner),
                Is.False, "a foreign owner must not be able to release the harness arm");
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.True);

            // 5. with the harness arm live the ordinary route now traces
            controller.Reset();
            controller.SetExpectedWord(FrozenCommonSignModel.ClassGreeting);
            controller.BeginCapture("optin-on");
            foreach (RawMediaPipeObservation observation in BuildObservations(12, false))
                source.Emit(observation);
            Assert.That(controller.FinalizeCapture(), Is.True);
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Not.Null);

            // 6. the harness's own disable route releases its own arm and drops the trace
            harness.DisableCommon2DDiagnosticTrace();
            Assert.That(harness.Common2DDiagnosticTraceEnabled, Is.False);
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.False);
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null);
            Assert.That(Common2DDiagnosticTraceBuffer.Owner, Is.Null);

            // 7. after the harness released, the ordinary route is untraced again and
            //    nothing re-arms by itself
            Common2DDiagnosticTraceBuffer.ResetCounters();
            controller.Reset();
            controller.SetExpectedWord(FrozenCommonSignModel.ClassGreeting);
            controller.BeginCapture("optin-after-release");
            foreach (RawMediaPipeObservation observation in BuildObservations(12, false))
                source.Emit(observation);
            Assert.That(controller.FinalizeCapture(), Is.True);
            Assert.That(Common2DDiagnosticTraceBuffer.DiagnosticAllocationCount, Is.Zero);
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null);
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.False);

            // 8. the REAL OnDisable body releases the arm again. An EditMode test gets
            //    no runtime MonoBehaviour lifecycle callback out of SetActive(false), so
            //    the production callback itself is invoked deterministically. The
            //    production harness is not modified and no ExecuteAlways is introduced:
            //    the body that runs here is the same one the Editor would run.
            Assert.That(harness.EnableCommon2DDiagnosticTrace(),
                Is.EqualTo(Common2DArmResult.Armed));
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.True);
            controller.Reset();
            controller.SetExpectedWord(FrozenCommonSignModel.ClassGreeting);
            controller.BeginCapture("optin-lifecycle-disable");
            foreach (RawMediaPipeObservation observation in BuildObservations(12, false))
                source.Emit(observation);
            Assert.That(controller.FinalizeCapture(), Is.True);
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Not.Null,
                "the arm under test must hold a live trace before OnDisable runs, or " +
                "clearing it afterwards would prove nothing");
            InvokeRealLifecycleCallback(harness, "OnDisable");
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.False,
                "OnDisable must release the arm the harness took");
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null);
            Assert.That(Common2DDiagnosticTraceBuffer.Owner, Is.Null);

            // 9. re-enabling the object must NOT silently re-arm; only a new explicit
            //    developer opt-in may
            harnessObject.SetActive(false);
            harnessObject.SetActive(true);
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.False,
                "re-enabling the component must not re-arm diagnostics by itself");
            Assert.That(harness.EnableCommon2DDiagnosticTrace(),
                Is.EqualTo(Common2DArmResult.Armed));
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.True);

            // 10. the REAL OnDestroy body independently releases a second, freshly taken
            //     arm and drops its trace. This is asserted before the object is
            //     destroyed for real, so it is a separate observation from step 8 rather
            //     than a latent consequence of it.
            controller.Reset();
            controller.SetExpectedWord(FrozenCommonSignModel.ClassGreeting);
            controller.BeginCapture("optin-lifecycle-destroy");
            foreach (RawMediaPipeObservation observation in BuildObservations(12, false))
                source.Emit(observation);
            Assert.That(controller.FinalizeCapture(), Is.True);
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Not.Null,
                "the arm under test must hold a live trace before OnDestroy runs");
            InvokeRealLifecycleCallback(harness, "OnDestroy");
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.False,
                "OnDestroy must release the arm the harness took");
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null);
            Assert.That(Common2DDiagnosticTraceBuffer.Owner, Is.Null);

            UnityEngine.Object.DestroyImmediate(harnessObject);
            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.False);
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null);
            Assert.That(Common2DDiagnosticTraceBuffer.Owner, Is.Null);

            controller.Dispose();
        }

        [Test]
        public void DecisionAndProbabilities_BitIdenticalEnabledVersusDisabled()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            CommonTwoDimensionalCapture disabledCapture;
            CommonSignClassification disabled = RunRealCapture(
                classifier, "bitwise-off", FrozenCommonSignModel.ClassThankYou,
                BuildObservations(14, false), out disabledCapture);

            Common2DDiagnosticTraceBuffer.TryArm(owner);
            CommonTwoDimensionalCapture enabledCapture;
            CommonSignClassification enabled = RunRealCapture(
                classifier, "bitwise-on", FrozenCommonSignModel.ClassThankYou,
                BuildObservations(14, false), out enabledCapture);

            Assert.That(Bits(enabled.DecisionValue), Is.EqualTo(Bits(disabled.DecisionValue)));
            Assert.That(Bits(enabled.GreetingProbability),
                Is.EqualTo(Bits(disabled.GreetingProbability)));
            Assert.That(Bits(enabled.ThankYouProbability),
                Is.EqualTo(Bits(disabled.ThankYouProbability)));

            double[] disabledCoordinates = disabledCapture.CopyCoordinates();
            double[] enabledCoordinates = enabledCapture.CopyCoordinates();
            Assert.That(enabledCoordinates.Length, Is.EqualTo(disabledCoordinates.Length));
            for (int i = 0; i < enabledCoordinates.Length; i++)
                Assert.That(Bits(enabledCoordinates[i]), Is.EqualTo(Bits(disabledCoordinates[i])),
                    "the normalization itself must be unchanged at element " + i);
        }

        [Test]
        public void OutcomeProgressionQuestAndPersistence_IdenticalEnabledVersusDisabled()
        {
            // The whole real route runs twice over the identical raw fixture: once with
            // diagnostics off and once with the developer trace armed. Every strict and
            // fused outcome field, every progression and quest value read off the real
            // PetRobotSignPracticeSession, and every call counted at the real production
            // persistence port must be identical, along with the decision and both
            // probability bits.
            FrozenCommonSignClassifier classifier = LoadRealClassifier();

            Assert.That(Common2DDiagnosticTraceBuffer.IsArmed, Is.False);
            RouteObservation off = RunFullRoute("off");
            Assert.That(off.ObservedTrace, Is.Null,
                "the unarmed run must not have produced a trace");

            Assert.That(Common2DDiagnosticTraceBuffer.TryArm(owner),
                Is.EqualTo(Common2DArmResult.Armed));
            RouteObservation on = RunFullRoute("on");
            Assert.That(on.ObservedTrace, Is.Not.Null,
                "the armed run must actually have produced a trace, or the comparison " +
                "proves nothing");

            AssertRoutesIdentical(off, on);

            // the production persistence port was really reached in both runs, and the
            // diagnostic changed nothing about what the policy decided there
            Assert.That(on.PersistenceCountsReadsAfter,
                Is.GreaterThan(on.PersistenceCountsReadsBefore));
            Assert.That(off.PersistenceCountsReadsAfter,
                Is.GreaterThan(off.PersistenceCountsReadsBefore));
            Assert.That(on.PracticeStorageStatus, Is.Not.Empty);
            Assert.That(on.PracticeAttemptCaptureSessionId, Is.Not.Empty);
            Assert.That(on.PracticeState,
                Is.Not.EqualTo(PetRobotPracticeState.Ready.ToString()));

            // and the observed classification really was produced, not defaulted
            Assert.That(on.TopClass, Is.Not.Empty);
            Assert.That(on.FinalizedCount, Is.EqualTo(1));
            Assert.That(on.AdapterFinalizeCaptureCalls, Is.GreaterThan(0));
            Assert.That(classifier.ModelAvailable, Is.True);
        }

        [Test]
        public void RawObservationSequence_CapturedWithMasksConventionAndBothGeometries()
        {
            // Every frame, every coordinate block, every one of the nine mask blocks, the
            // timestamp, the convention and both geometries are compared value by value
            // against the exact raw observations that were fed in. No representative
            // index and no length-only check.
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            Common2DDiagnosticTraceBuffer.TryArm(owner);
            List<RawMediaPipeObservation> observations = BuildObservations(11, true);
            CommonTwoDimensionalCapture capture;
            RunRealCapture(classifier, "raw-sequence", FrozenCommonSignModel.ClassGreeting,
                observations, out capture);

            Common2DDiagnosticTrace trace = Common2DDiagnosticTraceBuffer.Active;
            Assert.That(trace.RawObservations.Count, Is.EqualTo(11));
            Assert.That(trace.SourceFrameCount, Is.EqualTo(11));
            Assert.That(trace.NormalizedFrameCount, Is.EqualTo(11));
            Assert.That(Common2DRawObservationRecord.FieldCount, Is.EqualTo(17));

            int comparedCoordinates = 0;
            int comparedMaskBits = 0;
            for (int i = 0; i < trace.RawObservations.Count; i++)
            {
                Common2DRawObservationRecord record = trace.RawObservations[i];
                RawMediaPipeObservation origin = observations[i];
                string at = " at frame " + i;

                Assert.That(record.HasExactShapes(), Is.True, "shapes" + at);
                Assert.That(Bits(record.TimestampSeconds),
                    Is.EqualTo(Bits(origin.TimestampSeconds)), "timestamp" + at);

                // the three coordinate blocks, element by element and bit for bit
                double[][] expectedCoordinates =
                {
                    origin.Pose.CopyCoordinates(),
                    origin.LeftHand.CopyCoordinates(),
                    origin.RightHand.CopyCoordinates(),
                };
                double[][] actualCoordinates =
                {
                    record.PoseCoordinates,
                    record.LeftHandCoordinates,
                    record.RightHandCoordinates,
                };
                string[] coordinateNames = { "pose", "left hand", "right hand" };
                for (int block = 0; block < expectedCoordinates.Length; block++)
                {
                    Assert.That(actualCoordinates[block].Length,
                        Is.EqualTo(expectedCoordinates[block].Length),
                        coordinateNames[block] + " length" + at);
                    for (int k = 0; k < expectedCoordinates[block].Length; k++)
                    {
                        Assert.That(Bits(actualCoordinates[block][k]),
                            Is.EqualTo(Bits(expectedCoordinates[block][k])),
                            coordinateNames[block] + " element " + k + at);
                        comparedCoordinates++;
                    }
                }

                // all nine mask blocks: valid, reappearance and strong-depth boundary for
                // pose, left hand and right hand
                // built from the per-landmark accessors on the source observation, so the
                // expected values are read independently of the copy the trace made
                bool[][] expectedMasks =
                {
                    origin.Pose.CopyValidMask(),
                    origin.LeftHand.CopyValidMask(),
                    origin.RightHand.CopyValidMask(),
                    ReappearanceOf(origin.Pose),
                    ReappearanceOf(origin.LeftHand),
                    ReappearanceOf(origin.RightHand),
                    BoundaryOf(origin.Pose),
                    BoundaryOf(origin.LeftHand),
                    BoundaryOf(origin.RightHand),
                };
                bool[][] actualMasks =
                {
                    record.PoseValidMask,
                    record.LeftHandValidMask,
                    record.RightHandValidMask,
                    record.PoseReappearanceMask,
                    record.LeftHandReappearanceMask,
                    record.RightHandReappearanceMask,
                    record.PoseStrongDepthBoundaryMask,
                    record.LeftHandStrongDepthBoundaryMask,
                    record.RightHandStrongDepthBoundaryMask,
                };
                string[] maskNames =
                {
                    "pose valid", "left valid", "right valid",
                    "pose reappearance", "left reappearance", "right reappearance",
                    "pose boundary", "left boundary", "right boundary",
                };
                Assert.That(expectedMasks.Length, Is.EqualTo(9));
                Assert.That(actualMasks.Length, Is.EqualTo(9));
                for (int block = 0; block < expectedMasks.Length; block++)
                {
                    Assert.That(actualMasks[block].Length,
                        Is.EqualTo(expectedMasks[block].Length),
                        maskNames[block] + " length" + at);
                    for (int k = 0; k < expectedMasks[block].Length; k++)
                    {
                        Assert.That(actualMasks[block][k],
                            Is.EqualTo(expectedMasks[block][k]),
                            maskNames[block] + " bit " + k + at);
                        comparedMaskBits++;
                    }
                }

                // the convention tuple, both geometries and the agreement, field by field
                Assert.That(Common2DClosedVocabulary.IsApprovedConvention(record.Convention),
                    Is.True, "convention" + at);
                Assert.That(record.Convention.Id,
                    Is.EqualTo(origin.Convention.Id), "convention id" + at);
                Assert.That(record.Convention.CoordinateSpace,
                    Is.EqualTo(origin.Convention.CoordinateSpace.ToString()),
                    "convention coordinate space" + at);
                Assert.That(record.Convention.AxisAndDepth,
                    Is.EqualTo(origin.Convention.AxisAndDepth), "convention axis" + at);
                Assert.That(record.Convention.Handedness,
                    Is.EqualTo(origin.Convention.Handedness),
                    "convention handedness" + at);
                Assert.That(record.Convention.Mirror,
                    Is.EqualTo(origin.Convention.Mirror), "convention mirror" + at);
                Assert.That(record.MergedGeometryHand.Present, Is.True, "hand geometry" + at);
                Assert.That(record.MergedGeometryPose.Present, Is.True, "pose geometry" + at);
                AssertGeometryMatches(record.MergedGeometryHand,
                    origin.MergedGeometry.Hand, "hand" + at);
                AssertGeometryMatches(record.MergedGeometryPose,
                    origin.MergedGeometry.Pose, "pose" + at);
                Assert.That(record.MergedGeometryAgreement.BothAvailable, Is.True,
                    "agreement availability" + at);
                Assert.That(record.MergedGeometryAgreement.FullContractCompatible, Is.True,
                    "agreement compatibility" + at);
                Assert.That(Common2DClosedVocabulary.IsApprovedGeometry(
                    record.MergedGeometryHand), Is.True, "approved hand geometry" + at);
                Assert.That(Common2DClosedVocabulary.IsApprovedGeometry(
                    record.MergedGeometryPose), Is.True, "approved pose geometry" + at);
                Assert.That(Common2DClosedVocabulary.IsApprovedAgreement(
                    record.MergedGeometryAgreement), Is.True, "approved agreement" + at);
            }

            // the loop really compared the whole sequence, not a sample of it
            Assert.That(comparedCoordinates,
                Is.EqualTo(11 * (PoseLandmarks + HandLandmarks + HandLandmarks) * 3));
            Assert.That(comparedMaskBits,
                Is.EqualTo(11 * (PoseLandmarks + HandLandmarks + HandLandmarks) * 3));
        }

        [Test]
        public void AbsentLandmarkNaN_RoundTripsThroughCanonicalRepresentation()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            CommonSignClassification classification;
            CommonTwoDimensionalCapture capture;
            Common2DDiagnosticTrace trace;
            Common2DDiagnosticExportResult export = CommitRealExport(
                classifier, "nan-roundtrip", FrozenCommonSignModel.ClassGreeting,
                12, true, out classification, out capture, out trace);

            Assert.That(double.IsNaN(trace.RawObservations[0].LeftHandCoordinates[3]), Is.True);

            byte[] bytes = File.ReadAllBytes(
                Path.Combine(scratchRoot, export.ExportId, "raw_left_hand_coordinates.bin"));
            ulong word = 0;
            for (int b = 7; b >= 0; b--) word = (word << 8) | bytes[3 * 8 + b];
            Assert.That(word, Is.EqualTo(Common2DDiagnosticTraceExporter.CanonicalQuietNaNBits));
            Assert.That(double.IsNaN(BitConverter.Int64BitsToDouble(unchecked((long)word))), Is.True);

            string manifest = File.ReadAllText(Path.Combine(
                scratchRoot, export.ExportId, Common2DDiagnosticTraceExporter.ManifestName));
            Assert.That(manifest.Contains(": NaN"), Is.False);
            Assert.That(manifest.Contains("\"jsonToken\": \"NaN\""), Is.True);
        }

        [Test]
        public void EveryCanonicalPayloadField_PresentWithDeclaredShapeAndDtype()
        {
            // The oracle is the fixture's own IndependentMemberSchema, not the production
            // CanonicalMemberSchema. The production table is compared AGAINST it, so a
            // silent change to production is a failure rather than a moving target.
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            CommonSignClassification classification;
            CommonTwoDimensionalCapture capture;
            Common2DDiagnosticTrace trace;
            Common2DDiagnosticExportResult export = CommitRealExport(
                classifier, "fields", FrozenCommonSignModel.ClassGreeting,
                12, false, out classification, out capture, out trace);

            Assert.That(IndependentMemberSchema.Length, Is.EqualTo(IndependentMemberCount));
            Assert.That(Common2DDiagnosticTraceExporter.CanonicalMemberCount,
                Is.EqualTo(IndependentMemberCount),
                "the production member count must equal the independent expectation");
            Assert.That(Common2DDiagnosticTraceExporter.CanonicalMemberSchema.Length,
                Is.EqualTo(IndependentMemberCount));
            Assert.That(export.MemberCount, Is.EqualTo(IndependentMemberCount));

            string directory = Path.Combine(scratchRoot, export.ExportId);
            string manifest = File.ReadAllText(Path.Combine(
                directory, Common2DDiagnosticTraceExporter.ManifestName));
            Dictionary<string, JsonValue> manifestMembers = ManifestMembers(manifest);
            Assert.That(manifestMembers.Count, Is.EqualTo(IndependentMemberCount),
                "the manifest must declare exactly the independent member count");

            int rawFrames = trace.RawObservations.Count;
            int normalizedFrames = trace.NormalizedFrameCount;
            var seenGroups = new List<string>();
            for (int i = 0; i < IndependentMemberCount; i++)
            {
                string[] expected = IndependentMemberSchema[i];
                string[] produced = Common2DDiagnosticTraceExporter.CanonicalMemberSchema[i];
                string name = expected[0];

                // production order, name, group and dtype must equal the independent table
                Assert.That(produced[0], Is.EqualTo(expected[0]), "name at " + i);
                Assert.That(produced[1], Is.EqualTo(expected[1]), "group at " + i);
                Assert.That(produced[2], Is.EqualTo(expected[2]), "dtype at " + i);
                Assert.That(produced[3], Is.EqualTo(expected[3]), "shape at " + i);

                string shape = ResolveShape(expected[3], rawFrames, normalizedFrames);
                Assert.That(File.Exists(Path.Combine(directory, name)), Is.True, name);
                Assert.That(ManifestText(manifestMembers, name, "group"),
                    Is.EqualTo(expected[1]), name);
                Assert.That(ManifestText(manifestMembers, name, "dtype"),
                    Is.EqualTo(expected[2]), name);
                Assert.That(ManifestText(manifestMembers, name, "shape"),
                    Is.EqualTo(shape), name);

                byte[] bytes = File.ReadAllBytes(Path.Combine(directory, name));
                long elements = expected[2] == "utf8_json" ? 1 : ShapeProduct(shape);
                Assert.That(ManifestInteger(manifestMembers, name, "elementCount"),
                    Is.EqualTo(elements), name);
                if (expected[2] == "<f8")
                    Assert.That(bytes.Length, Is.EqualTo(elements * 8), name);
                else if (expected[2] == "packed_bool_lsb_first")
                    Assert.That(bytes.Length, Is.EqualTo((elements + 7) / 8), name);
                else
                    Assert.That(bytes.Length, Is.GreaterThan(0), name);
                Assert.That(ManifestInteger(manifestMembers, name, "byteCount"),
                    Is.EqualTo((long)bytes.Length), name);
                Assert.That(ManifestText(manifestMembers, name, "sha256"),
                    Is.EqualTo(Sha256Hex(bytes)), name);

                if (!seenGroups.Contains(expected[1])) seenGroups.Add(expected[1]);
            }

            // the ten canonical groups, in the order the independent table first names them
            Assert.That(seenGroups.Count, Is.EqualTo(10));
            for (int g = 0; g < seenGroups.Count; g++)
                Assert.That(Common2DDiagnosticTraceExporter.CanonicalPayloadGroups[g],
                    Is.EqualTo(seenGroups[g]), "group order at " + g);

            // and the isolated negatives really execute against the member schema
            AssertIsolatedNegative(classifier, "fields-short-feature",
                t => t.CloneWithOverride(featureVector: new double[1]),
                Common2DTraceFailureReason.InvalidPayloadShape);
            AssertSchemaMutationRejected(classifier, "fields-name",
                Common2DMemberSchemaMutation.Name, 0);
            AssertSchemaMutationRejected(classifier, "fields-order",
                Common2DMemberSchemaMutation.Order, 0);
            AssertSchemaMutationRejected(classifier, "fields-group",
                Common2DMemberSchemaMutation.Group, 0);
            AssertSchemaMutationRejected(classifier, "fields-dtype",
                Common2DMemberSchemaMutation.Dtype, 0);
            AssertSchemaMutationRejected(classifier, "fields-shape",
                Common2DMemberSchemaMutation.Shape, 0);
            AssertSchemaMutationRejected(classifier, "fields-elements",
                Common2DMemberSchemaMutation.ElementCount, 0);
            AssertSchemaMutationRejected(classifier, "fields-bytes",
                Common2DMemberSchemaMutation.ByteCount, 0);
            AssertSchemaMutationRejected(classifier, "fields-pad",
                Common2DMemberSchemaMutation.PadBit, 4);
        }

        [Test]
        public void SerializedPayload_RoundTripsBitExactAndPerArrayShaMatches()
        {
            // Every one of the twenty-four members is read back and compared VALUE by
            // VALUE against an expected sequence selected independently of the export:
            // the raw observations that were fed in, the trace the real path built and
            // the frozen parameters on disk. Lengths and digests alone would not show the
            // bytes are the right ones.
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            CommonSignClassification classification;
            CommonTwoDimensionalCapture capture;
            Common2DDiagnosticTrace trace;
            Common2DDiagnosticExportResult export = CommitRealExport(
                classifier, "roundtrip", FrozenCommonSignModel.ClassGreeting,
                12, false, out classification, out capture, out trace);

            string directory = Path.Combine(scratchRoot, export.ExportId);
            string manifestPath = Path.Combine(
                directory, Common2DDiagnosticTraceExporter.ManifestName);
            byte[] manifestBytes = File.ReadAllBytes(manifestPath);
            Assert.That(Sha256Hex(manifestBytes), Is.EqualTo(export.PayloadSha256));
            string manifest = Encoding.UTF8.GetString(manifestBytes);
            Assert.That(manifest.Contains(export.PayloadSha256), Is.False,
                "the manifest may not carry its own digest");
            Assert.That(manifestBytes[0], Is.Not.EqualTo(0xEF), "no BOM");
            Dictionary<string, JsonValue> manifestMembers = ManifestMembers(manifest);
            Assert.That(manifestMembers.Count, Is.EqualTo(IndependentMemberCount),
                "the manifest must declare exactly the independent member count");

            int rawFrames = trace.RawObservations.Count;
            int normalizedFrames = trace.NormalizedFrameCount;

            // the independently expected value sequence for every binary member
            var expectedDoubles = new Dictionary<string, double[]>(StringComparer.Ordinal);
            var expectedBools = new Dictionary<string, bool[]>(StringComparer.Ordinal);

            var timestamps = new double[rawFrames];
            for (int i = 0; i < rawFrames; i++)
                timestamps[i] = trace.RawObservations[i].TimestampSeconds;
            expectedDoubles["raw_timestamps.bin"] = timestamps;
            expectedDoubles["raw_pose_coordinates.bin"] =
                Flatten(trace, r => r.PoseCoordinates);
            expectedDoubles["raw_left_hand_coordinates.bin"] =
                Flatten(trace, r => r.LeftHandCoordinates);
            expectedDoubles["raw_right_hand_coordinates.bin"] =
                Flatten(trace, r => r.RightHandCoordinates);
            expectedBools["raw_pose_valid_mask.bin"] =
                FlattenBools(trace, r => r.PoseValidMask);
            expectedBools["raw_left_hand_valid_mask.bin"] =
                FlattenBools(trace, r => r.LeftHandValidMask);
            expectedBools["raw_right_hand_valid_mask.bin"] =
                FlattenBools(trace, r => r.RightHandValidMask);
            expectedBools["raw_pose_reappearance_mask.bin"] =
                FlattenBools(trace, r => r.PoseReappearanceMask);
            expectedBools["raw_left_hand_reappearance_mask.bin"] =
                FlattenBools(trace, r => r.LeftHandReappearanceMask);
            expectedBools["raw_right_hand_reappearance_mask.bin"] =
                FlattenBools(trace, r => r.RightHandReappearanceMask);
            expectedBools["raw_pose_strong_depth_boundary_mask.bin"] =
                FlattenBools(trace, r => r.PoseStrongDepthBoundaryMask);
            expectedBools["raw_left_hand_strong_depth_boundary_mask.bin"] =
                FlattenBools(trace, r => r.LeftHandStrongDepthBoundaryMask);
            expectedBools["raw_right_hand_strong_depth_boundary_mask.bin"] =
                FlattenBools(trace, r => r.RightHandStrongDepthBoundaryMask);
            expectedDoubles["normalized_coordinates.bin"] = trace.NormalizedCoordinates;
            expectedBools["input_mask.bin"] = trace.InputMask;
            expectedDoubles["resampled_coordinates.bin"] = trace.ResampledCoordinates;
            expectedBools["resampled_mask.bin"] = trace.ResampledMask;
            expectedDoubles["feature_vector.bin"] = trace.FeatureVector;
            expectedDoubles["standardized_feature_vector.bin"] =
                trace.StandardizedFeatureVector;
            expectedDoubles["pca_projection.bin"] = trace.PcaProjection;
            expectedDoubles["lr_component_contributions.bin"] =
                trace.LrComponentContributions;
            expectedDoubles["lr_scalars.bin"] = new[]
            {
                trace.LrIntercept, trace.DecisionValue,
                trace.GreetingProbability, trace.ThankYouProbability,
            };

            int floatMembers = 0;
            int boolMembers = 0;
            int jsonMembers = 0;
            int comparedElements = 0;
            int comparedBits = 0;
            int comparedJsonFields = 0;
            for (int i = 0; i < IndependentMemberCount; i++)
            {
                string[] entry = IndependentMemberSchema[i];
                string name = entry[0];
                string path = Path.Combine(directory, name);
                byte[] data = File.ReadAllBytes(path);
                string shape = ResolveShape(entry[3], rawFrames, normalizedFrames);

                Assert.That(Sha256Hex(data),
                    Is.EqualTo(ManifestText(manifestMembers, name, "sha256")), name);
                Assert.That((long)data.Length,
                    Is.EqualTo(ManifestInteger(manifestMembers, name, "byteCount")), name);
                Assert.That(ManifestText(manifestMembers, name, "shape"),
                    Is.EqualTo(shape), name);
                Assert.That(ManifestText(manifestMembers, name, "group"),
                    Is.EqualTo(entry[1]), name);
                Assert.That(ManifestText(manifestMembers, name, "dtype"),
                    Is.EqualTo(entry[2]), name);
                Assert.That(export.ArraySha256s.Contains(Sha256Hex(data)), Is.True, name);

                if (entry[2] == "<f8")
                {
                    floatMembers++;
                    double[] values = DecodeDoubles(data);
                    double[] expected = expectedDoubles[name];
                    Assert.That(values.Length, Is.EqualTo(expected.Length), name);
                    Assert.That((long)values.Length, Is.EqualTo(ShapeProduct(shape)), name);
                    for (int k = 0; k < values.Length; k++)
                    {
                        Assert.That(Bits(values[k]), Is.EqualTo(Bits(expected[k])),
                            name + " element " + k);
                        comparedElements++;
                    }
                }
                else if (entry[2] == "packed_bool_lsb_first")
                {
                    boolMembers++;
                    bool[] expected = expectedBools[name];
                    Assert.That((long)expected.Length, Is.EqualTo(ShapeProduct(shape)), name);
                    bool[] values = DecodePackedBools(data, expected.Length);
                    for (int k = 0; k < expected.Length; k++)
                    {
                        Assert.That(values[k], Is.EqualTo(expected[k]),
                            name + " bit " + k);
                        comparedBits++;
                    }
                    int padding = data.Length * 8 - expected.Length;
                    Assert.That(padding, Is.LessThan(8), name);
                    for (int bit = 8 - padding; padding > 0 && bit < 8; bit++)
                        Assert.That((data[data.Length - 1] & (1 << bit)), Is.Zero,
                            "padding bit " + bit + " must be zero in " + name);
                }
                else
                {
                    jsonMembers++;
                    // Both JSON members are TOP-LEVEL OBJECTS whose "observations" key
                    // holds an array of per-frame objects. A flat parser would turn that
                    // array into text, compare it to a frame count and look for "id" at
                    // the top level, none of which exists. The document is decoded
                    // structurally and every nested field is compared per frame.
                    string json = Encoding.UTF8.GetString(data);
                    Assert.That(json.Contains(": NaN"), Is.False, name);
                    JsonValue document = JsonReader.Parse(json);
                    Assert.That(document.Kind, Is.EqualTo(JsonValue.JsonKind.Object),
                        name + " must be a top-level object");
                    JsonValue observations = document.Member("observations");
                    Assert.That(observations.Kind, Is.EqualTo(JsonValue.JsonKind.Array),
                        name + " observations must be an array, not a scalar");
                    Assert.That(observations.Items.Count, Is.EqualTo(rawFrames),
                        name + " must carry one entry per raw observation");
                    Assert.That(document.Members.ContainsKey("id"), Is.False,
                        name + " has no top-level id; id lives inside each frame");

                    for (int frame = 0; frame < observations.Items.Count; frame++)
                    {
                        JsonValue frameObject = observations.Items[frame];
                        Common2DRawObservationRecord record =
                            trace.RawObservations[frame];
                        string at = name + " frame " + frame;
                        Assert.That(frameObject.Kind,
                            Is.EqualTo(JsonValue.JsonKind.Object), at);
                        if (name == "raw_convention.json")
                        {
                            AssertConventionJson(frameObject, record.Convention, at);
                            comparedJsonFields += 5;
                        }
                        else
                        {
                            AssertGeometryJson(frameObject.Member("hand"),
                                record.MergedGeometryHand, at + " hand");
                            AssertGeometryJson(frameObject.Member("pose"),
                                record.MergedGeometryPose, at + " pose");
                            AssertAgreementJson(frameObject.Member("agreement"),
                                record.MergedGeometryAgreement, at + " agreement");
                            comparedJsonFields += 51;
                        }
                    }
                }
            }

            Assert.That(floatMembers + boolMembers + jsonMembers,
                Is.EqualTo(IndependentMemberCount));
            Assert.That(jsonMembers, Is.EqualTo(2));
            Assert.That(floatMembers, Is.EqualTo(11));
            Assert.That(boolMembers, Is.EqualTo(11));

            // the loops really compared values, not just lengths
            Assert.That(comparedElements, Is.GreaterThan(3000));
            Assert.That(comparedBits, Is.GreaterThan(1000));
            Assert.That(comparedJsonFields,
                Is.EqualTo(rawFrames * 5 + rawFrames * 51),
                "every nested JSON field of every frame must have been compared");
        }

        [Test]
        public void DecisionRecomputedFromExportedZScorePcaAndContributions_Matches()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            CommonSignClassification classification;
            CommonTwoDimensionalCapture capture;
            Common2DDiagnosticTrace trace;
            Common2DDiagnosticExportResult export = CommitRealExport(
                classifier, "replay", FrozenCommonSignModel.ClassThankYou,
                15, false, out classification, out capture, out trace);

            string directory = Path.Combine(scratchRoot, export.ExportId);
            double[] exportedFeature = ReadDoubles(Path.Combine(directory, "feature_vector.bin"));
            double[] exportedStandardized = ReadDoubles(
                Path.Combine(directory, "standardized_feature_vector.bin"));
            double[] exportedPca = ReadDoubles(Path.Combine(directory, "pca_projection.bin"));
            double[] exportedContributions = ReadDoubles(
                Path.Combine(directory, "lr_component_contributions.bin"));
            double[] exportedScalars = ReadDoubles(Path.Combine(directory, "lr_scalars.bin"));

            // the frozen parameters, read read-only
            const int featureCount = FrozenCommonSignModel.FeatureCount;
            const int components = FrozenCommonSignModel.PcaComponents;
            double[] scalerMean = ReadFrozenParameter("scaler_mean.bin", featureCount);
            double[] scalerScale = ReadFrozenParameter("scaler_scale.bin", featureCount);
            double[] pcaMean = ReadFrozenParameter("pca_mean.bin", featureCount);
            double[] pcaComponents = ReadFrozenParameter(
                "pca_components.bin", components * featureCount);
            double[] lrCoef = ReadFrozenParameter("lr_coef.bin", components);
            double[] lrIntercept = ReadFrozenParameter("lr_intercept.bin", 1);

            // recompute the z-score from the exported feature, never reread it
            var recomputedStandardized = new double[featureCount];
            for (int i = 0; i < featureCount; i++)
                recomputedStandardized[i] =
                    (exportedFeature[i] - scalerMean[i]) / scalerScale[i];
            for (int i = 0; i < featureCount; i++)
                Assert.That(Bits(recomputedStandardized[i]),
                    Is.EqualTo(Bits(exportedStandardized[i])),
                    "recomputed z-score differs at " + i);

            // PCA from the recomputed z-score, contributions from PCA and the frozen LR
            var recomputedPca = new double[components];
            var recomputedContributions = new double[components];
            double recomputedDecision = lrIntercept[0];
            for (int k = 0; k < components; k++)
            {
                double projection = 0.0;
                int row = k * featureCount;
                for (int i = 0; i < featureCount; i++)
                    projection += (recomputedStandardized[i] - pcaMean[i]) *
                                  pcaComponents[row + i];
                double contribution = projection * lrCoef[k];
                recomputedDecision += contribution;
                recomputedPca[k] = projection;
                recomputedContributions[k] = contribution;
            }
            for (int k = 0; k < components; k++)
            {
                Assert.That(Bits(recomputedPca[k]), Is.EqualTo(Bits(exportedPca[k])));
                Assert.That(Bits(recomputedContributions[k]),
                    Is.EqualTo(Bits(exportedContributions[k])));
            }
            Assert.That(Bits(lrIntercept[0]), Is.EqualTo(Bits(exportedScalars[0])));
            Assert.That(Bits(recomputedDecision), Is.EqualTo(Bits(exportedScalars[1])));
            Assert.That(Bits(recomputedDecision), Is.EqualTo(Bits(classification.DecisionValue)));

            double greeting, thankYou;
            FrozenCommonSignModel.ToProbabilities(recomputedDecision, out greeting, out thankYou);
            Assert.That(Bits(greeting), Is.EqualTo(Bits(exportedScalars[2])));
            Assert.That(Bits(thankYou), Is.EqualTo(Bits(exportedScalars[3])));
            Assert.That(Bits(thankYou), Is.EqualTo(Bits(classification.ThankYouProbability)));
        }

        [Test]
        public void LogMarker_CarriesOnlyWhitelistedFieldsAndNoPathOrFreeText()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            CommonSignClassification classification;
            CommonTwoDimensionalCapture capture;
            Common2DDiagnosticTrace trace;
            Common2DDiagnosticExportResult export = CommitRealExport(
                classifier, "marker", FrozenCommonSignModel.ClassGreeting,
                12, false, out classification, out capture, out trace);

            string marker = export.Marker;
            Assert.That(marker.StartsWith(
                Common2DDiagnosticLogMarker.Prefix, StringComparison.Ordinal), Is.True);
            Assert.That(marker.Length,
                Is.LessThanOrEqualTo(Common2DDiagnosticLogMarker.MaxLength));
            Assert.That(Common2DDiagnosticLogMarker.AllowedFields.Length, Is.EqualTo(17));

            string[] parts = marker.Split(' ');
            var fields = new List<string>();
            string shapes = null;
            string digests = null;
            for (int i = 1; i < parts.Length; i++)
            {
                int separator = parts[i].IndexOf('=');
                Assert.That(separator, Is.GreaterThan(0), parts[i]);
                string field = parts[i].Substring(0, separator);
                fields.Add(field);
                Assert.That(Array.IndexOf(Common2DDiagnosticLogMarker.AllowedFields, field),
                    Is.GreaterThanOrEqualTo(0), field);
                if (field == "arrayShapes") shapes = parts[i].Substring(separator + 1);
                if (field == "arraySha256s") digests = parts[i].Substring(separator + 1);
            }
            Assert.That(fields.Count, Is.EqualTo(17));
            Assert.That(shapes.Split('|').Length, Is.EqualTo(export.MemberCount));
            string[] digestParts = digests.Split('|');
            Assert.That(digestParts.Length, Is.EqualTo(export.MemberCount));
            foreach (string digest in digestParts)
                Assert.That(digest.Length, Is.EqualTo(64));

            Assert.That(marker.Contains(scratchRoot), Is.False);
            Assert.That(marker.Contains("/"), Is.False);
            Assert.That(marker.Contains("\\"), Is.False);
            Assert.That(marker.Contains(Environment.UserName), Is.False);
            Assert.That(marker.Contains(Environment.MachineName), Is.False);

            string invalid;
            Common2DTraceFailureReason reason;
            Assert.That(Common2DDiagnosticLogMarker.TryBuildSuccess(
                trace.CloneWithOverride(captureSessionId: "C:/private.json"),
                export.ExportId, export.PayloadSha256, export.ArrayShapes,
                export.ArraySha256s, out invalid, out reason), Is.False);
            Assert.That(reason, Is.EqualTo(Common2DTraceFailureReason.InvalidBinding));
            string failureMarker = Common2DDiagnosticLogMarker.BuildFailure(
                trace.CloneWithOverride(captureSessionId: "C:/private.json"),
                Common2DTraceStatus.Failed, Common2DTraceFailureReason.InvalidBinding);
            Assert.That(failureMarker.Contains("private"), Is.False);
            Assert.That(failureMarker.Contains("captureSessionId=none"), Is.True);
        }

        [Test]
        public void ExistingExport_NeverOverwrittenOrTruncated()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            CommonSignClassification classification;
            CommonTwoDimensionalCapture capture;
            Common2DDiagnosticTrace trace;
            Common2DDiagnosticExportResult first = CommitRealExport(
                classifier, "duplicate", FrozenCommonSignModel.ClassGreeting,
                12, false, out classification, out capture, out trace);

            string manifestPath = Path.Combine(
                scratchRoot, first.ExportId, Common2DDiagnosticTraceExporter.ManifestName);
            byte[] originalBytes = File.ReadAllBytes(manifestPath);
            DateTime originalWrite = File.GetLastWriteTimeUtc(manifestPath);

            Common2DDiagnosticExportResult second =
                Common2DDiagnosticTraceExporter.ExportToScratchRoot(trace, scratchRoot);
            Assert.That(second.Status, Is.EqualTo(Common2DTraceStatus.Duplicate));
            Assert.That(second.FailureReason,
                Is.EqualTo(Common2DTraceFailureReason.AlreadyCommitted));
            Assert.That(File.ReadAllBytes(manifestPath), Is.EqualTo(originalBytes));
            Assert.That(File.GetLastWriteTimeUtc(manifestPath), Is.EqualTo(originalWrite));

            Common2DDiagnosticTraceBuffer.Clear();
            CommonTwoDimensionalCapture nextCapture;
            RunRealCapture(classifier, "staging-conflict", FrozenCommonSignModel.ClassThankYou,
                BuildObservations(11, false), out nextCapture);
            Common2DDiagnosticTrace next = Common2DDiagnosticTraceBuffer.Active;
            Assert.That(Common2DDiagnosticTraceBuffer.MarkTerminalAccepted(owner), Is.True);
            string staging = Path.Combine(
                scratchRoot, Common2DDiagnosticTraceExporter.TemporaryPrefix +
                Common2DDiagnosticTraceExporter.ComputeExportId(
                    "staging-conflict", FrozenCommonSignModel.ClassThankYou,
                    classifier.ArtifactVersion, classifier.ArtifactSha256));
            Directory.CreateDirectory(staging);
            File.WriteAllText(Path.Combine(staging, "keep.txt"), "pre-existing");
            Common2DDiagnosticExportResult conflict =
                Common2DDiagnosticTraceExporter.ExportToScratchRoot(next, scratchRoot);
            Assert.That(conflict.FailureReason,
                Is.EqualTo(Common2DTraceFailureReason.StagingConflict));
            Assert.That(File.ReadAllText(Path.Combine(staging, "keep.txt")),
                Is.EqualTo("pre-existing"));
        }

        [Test]
        public void WriteFailure_ProducesTypedBoundedMarkerAndLeavesCaptureUnchanged()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            Common2DDiagnosticTraceBuffer.TryArm(owner);
            CommonTwoDimensionalCapture cleanCapture;
            CommonSignClassification clean = RunRealCapture(
                classifier, "writefail", FrozenCommonSignModel.ClassThankYou,
                BuildObservations(12, false), out cleanCapture);
            Common2DDiagnosticTrace trace = Common2DDiagnosticTraceBuffer.Active;
            Common2DDiagnosticTraceBuffer.MarkTerminalAccepted(owner);

            ulong decisionBefore = Bits(clean.DecisionValue);

            // A clean baseline of the whole real route: controller state, coordinator
            // progression, strict and fused outcome, quest values and persistence calls.
            // Every fault stage below is compared against exactly this.
            StageBaseline baseline = RunStageRoute(
                classifier, "stage-clean", Common2DCollectionFaultInjection.None);

            Common2DDiagnosticTraceExporter.FaultInjection =
                Common2DExportFaultInjection.SerializationThrows;
            Assert.That(Common2DDiagnosticTraceExporter
                .ExportToScratchRoot(trace, scratchRoot).FailureReason,
                Is.EqualTo(Common2DTraceFailureReason.SerializationFailed));
            Common2DDiagnosticTraceExporter.FaultInjection =
                Common2DExportFaultInjection.WriteThrows;
            Common2DDiagnosticExportResult write =
                Common2DDiagnosticTraceExporter.ExportToScratchRoot(trace, scratchRoot);
            Common2DDiagnosticTraceExporter.FaultInjection = Common2DExportFaultInjection.None;
            Assert.That(write.FailureReason, Is.EqualTo(Common2DTraceFailureReason.WriteFailed));
            Assert.That(write.Marker.Contains("typedFailureReason=WriteFailed"), Is.True);
            Assert.That(write.Marker.Contains("arrayShapes=none"), Is.True);
            Assert.That(Directory.Exists(Path.Combine(scratchRoot, write.ExportId)), Is.False);
            Assert.That(Bits(clean.DecisionValue), Is.EqualTo(decisionBefore));

            // every collection stage, independently, stays closed and non-interfering
            Common2DCollectionFaultInjection[] stages =
            {
                Common2DCollectionFaultInjection.SinkAllocationThrows,
                Common2DCollectionFaultInjection.BeginSessionThrows,
                Common2DCollectionFaultInjection.RawObservationsThrow,
                Common2DCollectionFaultInjection.NormalizedCaptureThrows,
                Common2DCollectionFaultInjection.FeaturePathThrows,
                Common2DCollectionFaultInjection.ModelPathThrows,
                Common2DCollectionFaultInjection.SealThrows,
            };
            Assert.That(stages.Length, Is.EqualTo(7));
            foreach (Common2DCollectionFaultInjection stage in stages)
            {
                StageBaseline faulted = RunStageRoute(
                    classifier, "stage-" + stage, stage);

                // the injected diagnostic fault changed nothing the learner can observe
                Assert.That(faulted.DecisionBits,
                    Is.EqualTo(baseline.DecisionBits), stage.ToString());
                // The clean StageBaseline above is the only valid comparison here: it is
                // the same 13-frame connected route. decisionBefore comes from the
                // unrelated 12-frame direct capture and is asserted where it belongs,
                // against that same direct capture, above.
                Assert.That(faulted.GreetingBits,
                    Is.EqualTo(baseline.GreetingBits), stage.ToString());
                Assert.That(faulted.ThankYouBits,
                    Is.EqualTo(baseline.ThankYouBits), stage.ToString());
                Assert.That(faulted.ControllerState,
                    Is.EqualTo(baseline.ControllerState), stage.ToString());
                Assert.That(faulted.CoordinatorState,
                    Is.EqualTo(baseline.CoordinatorState), stage.ToString());
                Assert.That(faulted.FinalizedCount,
                    Is.EqualTo(baseline.FinalizedCount), stage.ToString());
                Assert.That(faulted.AdapterFinalizeCaptureCalls,
                    Is.EqualTo(baseline.AdapterFinalizeCaptureCalls), stage.ToString());
                Assert.That(faulted.AdapterResetCaptureCalls,
                    Is.EqualTo(baseline.AdapterResetCaptureCalls),
                    "no retry may be triggered by a diagnostic fault: " + stage);
                // exactly one of everything, in every case, through one route
                Assert.That(faulted.PersonalRecognizeCalls,
                    Is.EqualTo(baseline.PersonalRecognizeCalls),
                    "no reclassification may happen: " + stage);
                Assert.That(faulted.PersonalRecognizeCalls, Is.EqualTo(1),
                    "exactly one personal recognition: " + stage);
                Assert.That(faulted.CommonForwardCalls,
                    Is.EqualTo(baseline.CommonForwardCalls), stage.ToString());
                Assert.That(faulted.CommonForwardCalls, Is.EqualTo(1),
                    "exactly one real common-classifier forward: " + stage);
                Assert.That(faulted.QuestRecognizeCalls,
                    Is.EqualTo(baseline.QuestRecognizeCalls), stage.ToString());
                Assert.That(faulted.QuestRecognizeCalls, Is.EqualTo(1),
                    "exactly one quest-session recognition: " + stage);
                Assert.That(faulted.AdapterFinalizeCaptureCalls, Is.EqualTo(1),
                    "exactly one finalize: " + stage);
                Assert.That(faulted.AdapterResetCaptureCalls, Is.Zero,
                    "no reset on the first target: " + stage);
                Assert.That(faulted.RawFixtureFrames,
                    Is.EqualTo(baseline.RawFixtureFrames),
                    "the same raw frame count in every case: " + stage);
                Assert.That(faulted.PortEmittedObservations,
                    Is.EqualTo(baseline.PortEmittedObservations), stage.ToString());
                Assert.That(faulted.TopClass,
                    Is.EqualTo(baseline.TopClass), stage.ToString());
                Assert.That(faulted.ExpectedTargetSupport,
                    Is.EqualTo(baseline.ExpectedTargetSupport), stage.ToString());
                Assert.That(faulted.OutcomeKind,
                    Is.EqualTo(baseline.OutcomeKind), stage.ToString());
                Assert.That(faulted.StrictAccepted,
                    Is.EqualTo(baseline.StrictAccepted), stage.ToString());
                Assert.That(faulted.StrictExpectedWordClassAccept,
                    Is.EqualTo(baseline.StrictExpectedWordClassAccept), stage.ToString());
                Assert.That(faulted.StrictQuestSuccess,
                    Is.EqualTo(baseline.StrictQuestSuccess), stage.ToString());
                Assert.That(faulted.GuidedProgressAllowed,
                    Is.EqualTo(baseline.GuidedProgressAllowed), stage.ToString());
                Assert.That(faulted.OutcomeMessage,
                    Is.EqualTo(baseline.OutcomeMessage), stage.ToString());
                Assert.That(faulted.CommonEvidenceState,
                    Is.EqualTo(baseline.CommonEvidenceState), stage.ToString());
                // the REAL practice consumer reached the same state through the same
                // connected route, and its persistence policy made the same decision
                Assert.That(faulted.PracticeState,
                    Is.EqualTo(baseline.PracticeState), stage.ToString());
                Assert.That(faulted.PracticeCurrentWord,
                    Is.EqualTo(baseline.PracticeCurrentWord), stage.ToString());
                Assert.That(faulted.PracticeCurrentWordComplete,
                    Is.EqualTo(baseline.PracticeCurrentWordComplete), stage.ToString());
                Assert.That(faulted.PracticeAllWordsComplete,
                    Is.EqualTo(baseline.PracticeAllWordsComplete), stage.ToString());
                Assert.That(faulted.PracticeCanRetry,
                    Is.EqualTo(baseline.PracticeCanRetry), stage.ToString());

                // the ACTUAL quest state the real policy produced, not a default
                Assert.That(faulted.QuestMode,
                    Is.EqualTo(baseline.QuestMode), stage.ToString());
                Assert.That(faulted.QuestMode, Is.True, stage.ToString());
                Assert.That(faulted.QuestGeneration,
                    Is.EqualTo(baseline.QuestGeneration), stage.ToString());
                Assert.That(faulted.QuestGeneration, Is.Not.Zero, stage.ToString());
                Assert.That(faulted.QuestCompleted,
                    Is.EqualTo(baseline.QuestCompleted), stage.ToString());
                Assert.That(faulted.QuestAuthoritativeOutcome,
                    Is.EqualTo(baseline.QuestAuthoritativeOutcome), stage.ToString());
                Assert.That(faulted.QuestHybridResultPresent,
                    Is.EqualTo(baseline.QuestHybridResultPresent), stage.ToString());
                Assert.That(faulted.QuestHybridResultPresent, Is.True, stage.ToString());
                Assert.That(faulted.QuestHybridResultKind,
                    Is.EqualTo(baseline.QuestHybridResultKind), stage.ToString());
                Assert.That(faulted.PracticeStorageMessage,
                    Is.EqualTo(baseline.PracticeStorageMessage), stage.ToString());
                Assert.That(faulted.PracticeAttemptOutcomeKind,
                    Is.EqualTo(baseline.PracticeAttemptOutcomeKind), stage.ToString());
                Assert.That(faulted.PracticeStorageStatus,
                    Is.EqualTo(baseline.PracticeStorageStatus), stage.ToString());
                Assert.That(faulted.PersistenceSaveCallsDelta,
                    Is.EqualTo(baseline.PersistenceSaveCallsDelta), stage.ToString());
                Assert.That(faulted.PersistenceCountsReadsDelta,
                    Is.EqualTo(baseline.PersistenceCountsReadsDelta), stage.ToString());
                Assert.That(faulted.PersistenceCountsReadsDelta, Is.GreaterThan(0),
                    "the real persistence policy must run in every case: " + stage);

                // and the diagnostic itself failed closed, with the first cause retained
                Assert.That(faulted.CaptureFailed, Is.True, stage.ToString());
                Assert.That(faulted.FirstFailureReason,
                    Is.EqualTo(Common2DTraceFailureReason.CollectionFailed),
                    stage.ToString());
                Assert.That(faulted.FirstFailureStage, Is.EqualTo(stage));
                Assert.That(faulted.TraceAfterFailure, Is.Null,
                    "a later stage must not rebuild a partial trace: " + stage);
                Assert.That(faulted.Marker.Contains("typedFailureReason=CollectionFailed"),
                    Is.True, stage.ToString());
                Assert.That(faulted.Marker.Contains("Exception"), Is.False, stage.ToString());
            }

            // the clean baseline itself must have produced a real trace, or "identical"
            // would be a comparison of two empty runs
            Assert.That(baseline.CaptureFailed, Is.False);
            Assert.That(baseline.TraceAfterFailure, Is.Not.Null);
            Assert.That(baseline.PersonalRecognizeCalls, Is.EqualTo(1));
            Assert.That(baseline.CommonForwardCalls, Is.EqualTo(1));
            Assert.That(baseline.QuestRecognizeCalls, Is.EqualTo(1));
            Assert.That(baseline.QuestMode, Is.True);
            Assert.That(baseline.QuestHybridResultPresent, Is.True);
            Assert.That(baseline.PracticeStorageStatus,
                Is.EqualTo(CandidateStorageStatus.NotRequested.ToString()),
                "quest mode does not store a candidate, and the real policy says so");
        }

        [Test]
        public void NoDiagnosticWriteLandsInAnyProtectedPersistenceRoot()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            Common2DDiagnosticTraceBuffer.TryArm(owner);
            CommonTwoDimensionalCapture capture;
            RunRealCapture(classifier, "roots", FrozenCommonSignModel.ClassGreeting,
                BuildObservations(12, false), out capture);
            Common2DDiagnosticTrace trace = Common2DDiagnosticTraceBuffer.Active;
            Common2DDiagnosticTraceBuffer.MarkTerminalAccepted(owner);

            string[] rejected =
            {
                Path.Combine(Path.GetFullPath("."), "Profiles"),
                Path.Combine(Path.GetFullPath("."), "PersonalData"),
                Path.Combine(Path.GetFullPath("."), "Assets", "Diag"),
                Path.Combine(Path.GetFullPath("."), "Recordings"),
                Path.Combine(Path.GetFullPath("."), "TrainingSamples"),
                Path.Combine(Path.GetFullPath("."), "PersonalizationCandidates"),
                Path.Combine(scratchRoot, "..", "escape"),
                "relative/diag",
            };
            foreach (string root in rejected)
            {
                Common2DDiagnosticExportResult result =
                    Common2DDiagnosticTraceExporter.ExportToScratchRoot(trace, root);
                Assert.That(result.Committed, Is.False, root);
                Assert.That(result.FailureReason,
                    Is.EqualTo(Common2DTraceFailureReason.ExportRootRejected), root);
                Assert.That(Directory.Exists(root), Is.False, root);
            }

            // A REAL reparse alias. The junction is created here, its reparse semantics
            // are asserted as a precondition rather than tested with an if, and only then
            // is the export attempted through it.
            string aliasTarget = Path.Combine(scratchRoot, "alias-target");
            string aliasLink = Path.Combine(scratchRoot, "alias-link");
            CreateRealDirectoryJunction(aliasLink, aliasTarget);
            Assert.That(Directory.GetFileSystemEntries(aliasTarget).Length, Is.Zero,
                "the junction target must start empty");

            Common2DDiagnosticExportResult alias =
                Common2DDiagnosticTraceExporter.ExportToScratchRoot(trace, aliasLink);
            Assert.That(alias.Committed, Is.False,
                "a reparse alias must never be accepted as an export root");
            Assert.That(alias.FailureReason,
                Is.EqualTo(Common2DTraceFailureReason.ExportRootRejected));
            Assert.That(Directory.GetFileSystemEntries(aliasTarget).Length, Is.Zero,
                "no payload may reach the junction target");
            Assert.That(Directory.GetFileSystemEntries(aliasLink).Length, Is.Zero,
                "no payload may reach the junction itself");

            // the same alias pointed at a protected persistence root is refused too
            string protectedTarget = Path.Combine(scratchRoot, "Profiles");
            string protectedLink = Path.Combine(scratchRoot, "profiles-alias");
            CreateRealDirectoryJunction(protectedLink, protectedTarget);
            Common2DDiagnosticExportResult protectedAlias =
                Common2DDiagnosticTraceExporter.ExportToScratchRoot(trace, protectedLink);
            Assert.That(protectedAlias.Committed, Is.False);
            Assert.That(protectedAlias.FailureReason,
                Is.EqualTo(Common2DTraceFailureReason.ExportRootRejected));
            Assert.That(Directory.GetFileSystemEntries(protectedTarget).Length, Is.Zero);

            Assert.That(Common2DDiagnosticTraceExporter.DeveloperExportRoot,
                Is.EqualTo("Validation/Common2DDiagnosticExports/v1"));
            foreach (string name in Common2DDiagnosticTraceExporter.ProtectedRootNames)
                Assert.That(Common2DDiagnosticTraceExporter.DeveloperExportRoot.Contains(name),
                    Is.False, name);
        }

        [Test]
        public void BufferClearedOnInitializeBeginResetAndCommitsAtMostOncePerSession()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            var source = new ObservationSourceSpy { IsBound = true };
            var factory = new RuntimeFactorySpy();
            var controller = new PersonalLiveRecognitionController(source, factory);
            Assert.That(Common2DDiagnosticTraceBuffer.TryArm(owner),
                Is.EqualTo(Common2DArmResult.Armed));

            // Initialize clears
            Assert.That(controller.Initialize(
                BuildPaths(Path.Combine(scratchRoot, "runtime"))), Is.True);
            Assert.That(controller.State, Is.EqualTo(PersonalLiveCaptureState.Ready));
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null, "initialize");
            Assert.That(controller.IsObservationSubscribed, Is.True);

            // first session through the real controller
            controller.SetExpectedWord(FrozenCommonSignModel.ClassGreeting);
            Assert.That(controller.BeginCapture("session-first"), Is.True);
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null, "begin");
            foreach (RawMediaPipeObservation observation in BuildObservations(12, false))
                source.Emit(observation);
            Assert.That(controller.FinalizeCapture(), Is.True);
            Common2DDiagnosticTrace first = Common2DDiagnosticTraceBuffer.Active;
            Assert.That(first, Is.Not.Null);
            Assert.That(first.CaptureSessionId, Is.EqualTo("session-first"));

            CommonTwoDimensionalCapture firstCapture = controller.Result.Common2DCapture;
            classifier.Classify(firstCapture.CopyCoordinates(), firstCapture.CopyValidMask(),
                firstCapture.FrameCount, FrozenCommonSignModel.ClassGreeting);
            Common2DDiagnosticTraceBuffer.SafeMarkAttemptComplete(
                FrozenCommonSignModel.ClassGreeting);
            Assert.That(Common2DDiagnosticTraceBuffer.MarkTerminalAccepted(owner), Is.True);
            Common2DDiagnosticExportResult committed =
                Common2DDiagnosticTraceExporter.ExportToScratchRoot(first, scratchRoot);
            Assert.That(committed.Committed, Is.True, committed.FailureReason.ToString());
            Assert.That(Common2DDiagnosticTraceExporter
                .ExportToScratchRoot(first, scratchRoot).Status,
                Is.EqualTo(Common2DTraceStatus.Duplicate));
            Assert.That(Common2DDiagnosticTraceBuffer.HasCommitted("session-first"), Is.True);

            // Reset clears
            Assert.That(controller.Reset(), Is.True);
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null, "reset");

            // Cancel clears an incomplete trace without changing capture semantics
            controller.SetExpectedWord(FrozenCommonSignModel.ClassThankYou);
            Assert.That(controller.BeginCapture("session-cancel"), Is.True);
            foreach (RawMediaPipeObservation observation in BuildObservations(5, false))
                source.Emit(observation);
            Assert.That(controller.BufferedRawFrameCount, Is.EqualTo(5));
            Assert.That(controller.CancelCapture(), Is.True);
            Assert.That(controller.State, Is.EqualTo(PersonalLiveCaptureState.Cancelled));
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null, "cancel");

            // second session is fully isolated from the first
            Assert.That(controller.Reset(), Is.True);
            controller.SetExpectedWord(FrozenCommonSignModel.ClassThankYou);
            Assert.That(controller.BeginCapture("session-second"), Is.True);
            foreach (RawMediaPipeObservation observation in BuildObservations(9, false))
                source.Emit(observation);
            Assert.That(controller.FinalizeCapture(), Is.True);
            Common2DDiagnosticTrace second = Common2DDiagnosticTraceBuffer.Active;
            Assert.That(ReferenceEquals(first, second), Is.False);
            Assert.That(second.CaptureSessionId, Is.EqualTo("session-second"));
            Assert.That(second.RawObservations.Count, Is.EqualTo(9));
            Assert.That(second.TerminalAccepted, Is.False);

            // Disable drops the incomplete trace and unsubscribes
            controller.Disable();
            Assert.That(Common2DDiagnosticTraceBuffer.Active, Is.Null, "disable");
            Assert.That(controller.IsObservationSubscribed, Is.False);
            Assert.That(ReferenceEquals(Common2DDiagnosticTraceBuffer.Owner, owner), Is.True);
            controller.Dispose();
        }

        [Test]
        public void ObserverReceivesCopies_MutationDoesNotReachClassifierInputOrModel()
        {
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            Common2DDiagnosticTraceBuffer.TryArm(owner);
            List<RawMediaPipeObservation> observations = BuildObservations(12, false);
            CommonTwoDimensionalCapture capture;
            CommonSignClassification first = RunRealCapture(
                classifier, "copies", FrozenCommonSignModel.ClassGreeting,
                observations, out capture);

            double[] captureBefore = capture.CopyCoordinates();
            double rawBefore = observations[0].Pose.CopyCoordinates()[0];
            Common2DDiagnosticTrace trace = Common2DDiagnosticTraceBuffer.Active;

            trace.NormalizedCoordinates[0] = 999.0;
            trace.FeatureVector[0] = 999.0;
            trace.StandardizedFeatureVector[0] = 999.0;
            trace.PcaProjection[0] = 999.0;
            trace.ResampledCoordinates[0] = 999.0;
            trace.RawObservations[0].PoseCoordinates[0] = 999.0;

            double[] captureAfter = capture.CopyCoordinates();
            for (int i = 0; i < captureBefore.Length; i++)
                Assert.That(Bits(captureAfter[i]), Is.EqualTo(Bits(captureBefore[i])));
            Assert.That(Bits(observations[0].Pose.CopyCoordinates()[0]),
                Is.EqualTo(Bits(rawBefore)));

            CommonSignClassification second = classifier.Classify(
                captureBefore, capture.CopyValidMask(), capture.FrameCount,
                FrozenCommonSignModel.ClassGreeting);
            Assert.That(Bits(second.DecisionValue), Is.EqualTo(Bits(first.DecisionValue)));
        }

        [Test]
        public void PayloadBoundToCaptureSessionRequestedTargetAndModelIdentity()
        {
            // Each negative is EXECUTED through the real exporter on its own fresh,
            // uncommitted trace, so its closed reason is an observed export result rather
            // than a vocabulary predicate returning false.
            FrozenCommonSignClassifier classifier = LoadRealClassifier();
            CommonSignClassification classification;
            CommonTwoDimensionalCapture capture;
            Common2DDiagnosticTrace trace;
            Common2DDiagnosticExportResult export = CommitRealExport(
                classifier, "binding", FrozenCommonSignModel.ClassThankYou,
                12, false, out classification, out capture, out trace);

            Assert.That(trace.CaptureSessionId, Is.EqualTo("binding"));
            Assert.That(trace.RequestedTarget, Is.EqualTo(FrozenCommonSignModel.ClassThankYou));
            Assert.That(trace.SealTarget, Is.EqualTo(trace.RequestedTarget));
            Assert.That(trace.ArtifactVersion, Is.EqualTo(FrozenCommonSignModel.ArtifactVersion));
            Assert.That(trace.ArtifactSha256,
                Is.EqualTo(FrozenCommonSignModel.ExpectedCombinedSha256));
            Assert.That(trace.SourceFrameCount, Is.EqualTo(12));
            Assert.That(trace.NormalizedFrameCount, Is.EqualTo(trace.SourceFrameCount));
            Assert.That(trace.RawObservations.Count, Is.EqualTo(trace.SourceFrameCount));
            Assert.That(export.ExportId, Is.EqualTo(
                Common2DDiagnosticTraceExporter.ComputeExportId(
                    "binding", FrozenCommonSignModel.ClassThankYou,
                    FrozenCommonSignModel.ArtifactVersion, classifier.ArtifactSha256)));

            string manifest = File.ReadAllText(Path.Combine(
                scratchRoot, export.ExportId, Common2DDiagnosticTraceExporter.ManifestName));
            Assert.That(manifest.Contains("\"captureSessionId\": \"binding\""), Is.True);
            Assert.That(manifest.Contains("\"requestedTarget\": \"THANK_YOU\""), Is.True);
            Assert.That(manifest.Contains(FrozenCommonSignModel.ExpectedCombinedSha256), Is.True);
            Assert.That(manifest.Contains(Environment.UserName), Is.False);

            // an unmodified clone must validate exactly like its source, otherwise a
            // dropped field and not the injected defect would explain every negative
            Common2DTraceFailureReason sourceReason;
            bool sourceValidates = trace.Validate(out sourceReason);
            Common2DTraceFailureReason cloneReason;
            bool cloneValidates = trace.CloneWithOverride().Validate(out cloneReason);
            Assert.That(cloneValidates, Is.EqualTo(sourceValidates));
            Assert.That(cloneReason, Is.EqualTo(sourceReason));
            Assert.That(trace.CloneWithOverride().SealTarget, Is.EqualTo(trace.SealTarget));

            // each isolated negative on its own fresh uncommitted trace
            AssertIsolatedNegative(classifier, "bind-session",
                t => t.CloneWithOverride(captureSessionId: "C:/private.json"),
                Common2DTraceFailureReason.InvalidBinding);
            AssertIsolatedNegative(classifier, "bind-relabel",
                t => t.CloneWithOverride(
                    requestedTarget: FrozenCommonSignModel.ClassGreeting),
                Common2DTraceFailureReason.InvalidBinding);
            AssertIsolatedNegative(classifier, "bind-seal",
                t => t.CloneWithOverride(
                    sealTarget: FrozenCommonSignModel.ClassGreeting),
                Common2DTraceFailureReason.InvalidBinding);
            AssertIsolatedNegative(classifier, "bind-digest",
                t => t.CloneWithOverride(artifactSha256:
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
                Common2DTraceFailureReason.InvalidBinding);
            AssertIsolatedNegative(classifier, "bind-source-count",
                t => t.CloneWithOverride(
                    sourceFrameCount: t.RawObservations.Count + 1),
                Common2DTraceFailureReason.InvalidPayloadShape);
            AssertIsolatedNegative(classifier, "bind-normalized-count",
                t => t.CloneWithOverride(
                    normalizedFrameCount: t.NormalizedFrameCount + 1),
                Common2DTraceFailureReason.InvalidPayloadShape);
            AssertIsolatedNegative(classifier, "bind-short-feature",
                t => t.CloneWithOverride(featureVector: new double[1]),
                Common2DTraceFailureReason.InvalidPayloadShape);
            AssertIsolatedNegative(classifier, "bind-short-pca",
                t => t.CloneWithOverride(pcaProjection: new double[3]),
                Common2DTraceFailureReason.InvalidPayloadShape);
        }

        // ------------------------------------------------- Step 10W v1.0 world outcome
        //
        // Step 10V observed ONE capture produce guided LowConfidenceCorrect and World
        // WrongWord at the same time, because the World branch was taken from a second,
        // independent legacy decision over the same sequence. These cases pin the repair
        // at the PRODUCER, driving the real quest-mode session through the real
        // coordinator, rather than calling the World controller's outcome application
        // directly: the defect lived exactly in the layer between the finalized guided
        // outcome and the World branch, so a test that starts below that layer cannot
        // see it.


        // ------------------------------------------ Step 10W v1.0a World consumer doubles
        //
        // These exist so the exact Step 10V conflict can be carried across the real
        // producer-to-consumer bridge. They stand in for scene wiring only: the gate, the
        // camera, the player lock and the NPC animator. Nothing here produces, maps or
        // vetoes an outcome, and the session handed to the controller is the real one the
        // fixture just finished driving.

        sealed class AlwaysReadyGateVerifier : IQuestGateVerifier
        {
            public bool VerifyOverallEnrollmentGateReady() { return true; }
            public string LastFailure { get { return string.Empty; } }
        }

        sealed class WorldCameraSpy : IInWorldDialogueCameraController
        {
            public int SaveCalls, EnterCalls, RestoreCalls;
            public void SaveState() { SaveCalls++; }
            public void EnterDialogueCamera() { EnterCalls++; }
            public void RestoreState() { RestoreCalls++; }
        }

        sealed class WorldPlayerLockSpy : IInWorldDialoguePlayerLock
        {
            public int LockCalls, UnlockCalls;
            public void Lock() { LockCalls++; }
            public void Unlock() { UnlockCalls++; }
        }

        sealed class WorldNpcPresenterSpy : IInWorldNpcPresenter
        {
            public int GreetingCalls, ConfusedCalls, GiveItemCalls;
            public void TriggerGreeting() { GreetingCalls++; }
            public void TriggerConfused() { ConfusedCalls++; }
            public void TriggerGiveItem() { GiveItemCalls++; }
        }

        /// <summary>
        /// Hands the World controller the EXACT session instance the fixture drove, so the
        /// canonical outcome the producer decided is the one the consumer reads. A null
        /// coordinator is deliberate: the guided sub-state driver returns immediately, so
        /// the only thing that can move the controller is the session outcome bridge.
        /// </summary>
        sealed class SessionBackedRuntimeOwner : IInWorldRecognitionRuntimeOwner
        {
            readonly PetRobotSignPracticeSession session;
            public int DisposeCalls;

            public SessionBackedRuntimeOwner(PetRobotSignPracticeSession session)
            {
                this.session = session;
            }

            public bool IsActive { get { return true; } }
            public long FrozenGeneration { get { return session.QuestSessionGeneration; } }
            public PetRobotSignPracticeSession Session { get { return session; } }
            public PetRobotNeutralAutoCaptureCoordinator Coordinator { get { return null; } }
            public string InitializationFailure { get { return string.Empty; } }
            public bool RequestAttempt() { return true; }
            public bool RetrySession() { return false; }

            public InWorldRecognitionReadinessSnapshot SnapshotReadiness()
            {
                return new InWorldRecognitionReadinessSnapshot(
                    true, true, true, true,
                    RawMediaPipeNeutralAutoSegmenterState.Inactive, string.Empty);
            }

            public void Dispose() { DisposeCalls++; }
        }

        /// <summary>
        /// The press-R retry phrase the WrongWord and TechnicalFailure feedback strings
        /// carry and the LowConfidenceCorrect one must not. Written as escapes so this
        /// fixture stays ASCII, exactly like the production strings it checks.
        /// </summary>
        const string RetryInstruction = "R을 눌러";

        static readonly GuidedQuestOutcomeKind[] AllGuidedKinds =
        {
            GuidedQuestOutcomeKind.StrictSuccess,
            GuidedQuestOutcomeKind.LowConfidenceCorrect,
            GuidedQuestOutcomeKind.WrongWord,
            GuidedQuestOutcomeKind.TechnicalFailure,
        };

        [Test]
        public void Step10W_CanonicalWorldOutcomeMapping_IsExactTotalAndLabelPreserving()
        {
            Assert.That(PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(
                GuidedQuestOutcomeKind.StrictSuccess),
                Is.EqualTo(TrustedHybridOutcomeKind.StrictSuccess));
            Assert.That(PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(
                GuidedQuestOutcomeKind.LowConfidenceCorrect),
                Is.EqualTo(TrustedHybridOutcomeKind.LowConfidenceCorrect),
                "LowConfidenceCorrect must stay LowConfidenceCorrect and must never be " +
                "promoted to StrictSuccess");
            Assert.That(PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(
                GuidedQuestOutcomeKind.WrongWord),
                Is.EqualTo(TrustedHybridOutcomeKind.WrongWord));
            Assert.That(PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(
                GuidedQuestOutcomeKind.TechnicalFailure),
                Is.EqualTo(TrustedHybridOutcomeKind.TechnicalFailure));

            Assert.That(Enum.GetValues(typeof(GuidedQuestOutcomeKind)).Length,
                Is.EqualTo(AllGuidedKinds.Length),
                "every declared guided outcome must be covered by this mapping test");

            var mapped = new HashSet<TrustedHybridOutcomeKind>();
            foreach (GuidedQuestOutcomeKind guided in AllGuidedKinds)
                Assert.That(mapped.Add(
                    PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(guided)),
                    Is.True, "the mapping must be one-to-one, not many-to-one");
        }

        [Test]
        public void Step10W_ExactLowConfidenceConflict_CanonicalOutcomeReachesWorldThroughSameSession()
        {
            // The exact Step 10V shape, forced rather than hoped for:
            //   predicted == expected == GREETING, all three strict flags false,
            //   guided LowConfidenceCorrect, legacy frozen hybrid WrongWord.
            // Before the Step 10W repair the World read the legacy value and showed the
            // retry path. Every assertion below is unconditional; a conditional
            // "if the outcomes happen to differ" check is not this case.
            ConnectedPracticeRoute route = BuildConnectedPracticeRoute(
                "w-exact-conflict",
                new QuestHybridSessionSpy(TrustedHybridOutcomeKind.WrongWord));
            var npcObject = new GameObject("Step10W-NPC");
            var playerObject = new GameObject("Step10W-Player");
            var controllerObject = new GameObject("Step10W-DialogueSystem");
            try
            {
                route.Factory.Runtime.Mode = PersonalRoutingSpyMode.LowConfidenceCorrect;
                DriveRealPracticeAttempt(route, new RouteObservation());

                PracticeRecognitionAttempt attempt = route.Practice.LastAttempt;

                // ---- the exact guided shape
                Assert.That(attempt.PredictedWord, Is.EqualTo(attempt.ExpectedWord),
                    "the conflict case requires the predicted word to be the expected one");
                Assert.That(attempt.Outcome.Kind,
                    Is.EqualTo(GuidedQuestOutcomeKind.LowConfidenceCorrect));
                Assert.That(attempt.Outcome.GuidedProgressAllowed, Is.True);
                Assert.That(attempt.StrictAccepted, Is.False);
                Assert.That(attempt.StrictExpectedWordClassAccept, Is.False);
                Assert.That(attempt.StrictQuestSuccess, Is.False);

                // ---- the canonical World outcome, and the legacy value it ignored
                Assert.That(route.Practice.QuestAuthoritativeOutcome,
                    Is.EqualTo(TrustedHybridOutcomeKind.LowConfidenceCorrect),
                    "the World branch must be the guided outcome, mapped");
                Assert.That(route.Practice.QuestAuthoritativeOutcome,
                    Is.Not.EqualTo(TrustedHybridOutcomeKind.StrictSuccess),
                    "LowConfidenceCorrect must never be promoted to StrictSuccess");
                Assert.That(route.Practice.LegacyHybridDiagnosticOutcome,
                    Is.EqualTo(TrustedHybridOutcomeKind.WrongWord),
                    "the disagreeing legacy reading must still be recorded");
                Assert.That(route.Practice.QuestCompleted, Is.True);
                Assert.That(route.Repository.SaveCalls, Is.Zero,
                    "quest mode must not store a personalization candidate");

                // ---- the SAME session, across the real World bridge
                var controller =
                    controllerObject.AddComponent<InWorldGreetingNpcDialogueController>();
                var camera = new WorldCameraSpy();
                var playerLock = new WorldPlayerLockSpy();
                var npc = new WorldNpcPresenterSpy();
                var owner = new SessionBackedRuntimeOwner(route.Practice);
                float clock = 0f;

                controller.ConfigureForTests(
                    npcObject.transform, playerObject.transform, 3f,
                    new AlwaysReadyGateVerifier(),
                    new PetRobotLearnerRootIdentity(
                        "local_learner", Path.Combine(scratchRoot, "w-exact-conflict-root")),
                    owner, camera, playerLock, npc);
                controller.SetTimeSourceForTest(() => clock);
                controller.SetPlayerInsideGreetingTrigger(true);

                Assert.That(controller.TryStartDialogue(), Is.True,
                    "the World dialogue must open on the ready gate");
                Assert.That(controller.State,
                    Is.EqualTo(InWorldDialogueState.QuestPrompt));

                // production DriveUpdate, not a private outcome call: the prompt timer
                // begins recognition, which installs the owner holding the real session,
                // and DriveSessionOutcome then reaches QueueSessionOutcome(session).
                clock += controller.QuestPromptSeconds + 0.01f;
                controller.DriveUpdate();
                Assert.That(controller.State,
                    Is.EqualTo(InWorldDialogueState.Analyzing),
                    "the session outcome bridge must queue the canonical outcome");

                clock += controller.AnalysisDisplaySeconds + 0.01f;
                controller.DriveUpdate();

                Assert.That(controller.LastOutcome,
                    Is.EqualTo(InWorldDialogueOutcome.LowConfidenceCorrect),
                    "the World must show the canonical guided outcome");
                Assert.That(controller.State,
                    Is.EqualTo(InWorldDialogueState.QuestComplete));
                Assert.That(controller.ItemGranted, Is.True);
                Assert.That(npc.GiveItemCalls, Is.EqualTo(1),
                    "the item must be granted exactly once");
                Assert.That(npc.ConfusedCalls, Is.Zero,
                    "the WrongWord presentation must not appear");
                Assert.That(controller.CanRetry, Is.False,
                    "the old Feedback/R retry path must not be offered");
                Assert.That(controller.LastFeedbackText, Is.Not.Empty,
                    "a terminal feedback line must be present");
                Assert.That(controller.LastFeedbackText.Contains(RetryInstruction),
                    Is.False,
                    "the LowConfidenceCorrect feedback must not carry the press-R " +
                    "retry instruction the WrongWord and TechnicalFailure texts carry");

                // a second production update must not grant a second item
                clock += 1f;
                controller.DriveUpdate();
                Assert.That(npc.GiveItemCalls, Is.EqualTo(1),
                    "a further update must not grant the item again");
                Assert.That(controller.State,
                    Is.EqualTo(InWorldDialogueState.QuestComplete));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(controllerObject);
                UnityEngine.Object.DestroyImmediate(playerObject);
                UnityEngine.Object.DestroyImmediate(npcObject);
                route.Dispose();
            }
        }

        [Test]
        public void Step10W_LegacyHybridNullResult_CannotDegradeCanonicalOutcome()
        {
            ConnectedPracticeRoute route = BuildConnectedPracticeRoute(
                "w-legacy-null",
                new QuestHybridSessionSpy(TrustedHybridOutcomeKind.WrongWord,
                    QuestHybridSpyBehaviour.ReturnNull));
            try
            {
                route.Factory.Runtime.Mode = PersonalRoutingSpyMode.LowConfidenceCorrect;
                var observed = new RouteObservation();
                DriveRealPracticeAttempt(route, observed);

                PracticeRecognitionAttempt attempt = route.Practice.LastAttempt;
                Assert.That(attempt.Outcome.Kind,
                    Is.EqualTo(GuidedQuestOutcomeKind.LowConfidenceCorrect));
                Assert.That(route.Practice.QuestAuthoritativeOutcome.Value,
                    Is.EqualTo(TrustedHybridOutcomeKind.LowConfidenceCorrect),
                    "a null legacy result must not turn a valid canonical outcome into " +
                    "a TechnicalFailure");
                Assert.That(route.Practice.LegacyHybridDiagnosticOutcome, Is.Null);
                Assert.That(route.Practice.QuestHybridResult, Is.Null);
            }
            finally { route.Dispose(); }
        }

        [Test]
        public void Step10W_LegacyHybridThrow_CannotDegradeCanonicalOutcome()
        {
            ConnectedPracticeRoute route = BuildConnectedPracticeRoute(
                "w-legacy-throw",
                new QuestHybridSessionSpy(TrustedHybridOutcomeKind.WrongWord,
                    QuestHybridSpyBehaviour.Throw));
            try
            {
                route.Factory.Runtime.Mode = PersonalRoutingSpyMode.LowConfidenceCorrect;
                var observed = new RouteObservation();
                DriveRealPracticeAttempt(route, observed);

                PracticeRecognitionAttempt attempt = route.Practice.LastAttempt;
                Assert.That(attempt.Outcome.Kind,
                    Is.EqualTo(GuidedQuestOutcomeKind.LowConfidenceCorrect));
                Assert.That(route.Practice.QuestAuthoritativeOutcome.Value,
                    Is.EqualTo(TrustedHybridOutcomeKind.LowConfidenceCorrect),
                    "a throwing legacy pass must not change the canonical outcome");
                Assert.That(route.Practice.LegacyHybridDiagnosticOutcome, Is.Null);
                Assert.That(route.Practice.QuestHybridResult, Is.Null);
                Assert.That(route.Practice.State,
                    Is.EqualTo(PetRobotPracticeState.Feedback),
                    "a throwing legacy pass must not break the session state machine");
            }
            finally { route.Dispose(); }
        }

        [Test]
        public void Step10W_WorldOutcomeRecord_ExposesBothLayersExactlyOnce()
        {
            ConnectedPracticeRoute route = BuildConnectedPracticeRoute(
                "w-record",
                new QuestHybridSessionSpy(TrustedHybridOutcomeKind.WrongWord));
            var lines = new List<string>();
            Action<string> previous = CommonSignalAttemptDiagnostics.Sink;
            try
            {
                route.Factory.Runtime.Mode = PersonalRoutingSpyMode.LowConfidenceCorrect;
                CommonSignalAttemptDiagnostics.ResetForTest();
                CommonSignalAttemptDiagnostics.Sink = lines.Add;

                var observed = new RouteObservation();
                DriveRealPracticeAttempt(route, observed);

                Assert.That(lines.Count, Is.EqualTo(1),
                    "an accepted attempt must record exactly one line");
                string line = lines[0];
                Assert.That(line.StartsWith(CommonSignalDiagnosticRecord.Prefix),
                    Is.True, "the grep prefix must not change");

                PracticeRecognitionAttempt attempt = route.Practice.LastAttempt;
                TrustedHybridOutcomeKind canonical =
                    route.Practice.QuestAuthoritativeOutcome.Value;

                StringAssert.Contains(
                    "\"outcomeKind\":\"" + attempt.Outcome.Kind + "\"", line);
                StringAssert.Contains(
                    "\"questAuthoritativeOutcome\":\"" + canonical + "\"", line);
                StringAssert.Contains(
                    "\"worldOutcomeAuthority\":\"" +
                    PetRobotSignPracticeSession.WorldOutcomeAuthority + "\"", line);
                StringAssert.Contains(
                    "\"legacyHybridOutcomeKind\":\"" +
                    TrustedHybridOutcomeKind.WrongWord + "\"", line);
                // unconditional: this case forces guided LowConfidenceCorrect against a
                // legacy WrongWord, so the record must say the legacy value was ignored
                Assert.That(canonical,
                    Is.EqualTo(TrustedHybridOutcomeKind.LowConfidenceCorrect));
                StringAssert.Contains(
                    "\"outcomeAgreement\":\"" +
                    "legacy_disagrees_with_canonical_and_was_ignored\"", line);
                StringAssert.Contains("\"strictAccepted\":false", line);
                StringAssert.Contains("\"strictQuestSuccess\":false", line);
                StringAssert.Contains("\"guidedProgressAllowed\":true", line);

                // strict values, predicted word and grants are reported, never promoted
                StringAssert.Contains("\"strictAccepted\":" +
                    (attempt.StrictAccepted ? "true" : "false"), line);
                StringAssert.Contains("\"strictExpectedWordClassAccept\":" +
                    (attempt.StrictExpectedWordClassAccept ? "true" : "false"), line);
                StringAssert.Contains("\"strictQuestSuccess\":" +
                    (attempt.StrictQuestSuccess ? "true" : "false"), line);
                StringAssert.Contains(
                    "\"predictedWord\":\"" + attempt.PredictedWord + "\"", line);

                // a repeated callback for the same capture adds no second line
                CommonSignalAttemptDiagnostics.Emit(
                    attempt, canonical, TrustedHybridOutcomeKind.WrongWord, true);
                Assert.That(lines.Count, Is.EqualTo(1),
                    "a repeated callback for the same capture must add no record");
            }
            finally
            {
                CommonSignalAttemptDiagnostics.Sink = previous;
                CommonSignalAttemptDiagnostics.ResetForTest();
                route.Dispose();
            }
        }

        [Test]
        public void Step10W_RecordSinkFailure_DoesNotChangeCanonicalOutcomeOrState()
        {
            ConnectedPracticeRoute route = BuildConnectedPracticeRoute(
                "w-sink-fail",
                new QuestHybridSessionSpy(TrustedHybridOutcomeKind.WrongWord));
            Action<string> previous = CommonSignalAttemptDiagnostics.Sink;
            try
            {
                route.Factory.Runtime.Mode = PersonalRoutingSpyMode.LowConfidenceCorrect;
                CommonSignalAttemptDiagnostics.ResetForTest();
                CommonSignalAttemptDiagnostics.Sink =
                    _ => { throw new InvalidOperationException("invented sink failure"); };

                var observed = new RouteObservation();
                DriveRealPracticeAttempt(route, observed);

                PracticeRecognitionAttempt attempt = route.Practice.LastAttempt;
                Assert.That(route.Practice.QuestAuthoritativeOutcome.Value,
                    Is.EqualTo(
                        PetRobotSignPracticeSession.MapGuidedToCanonicalWorldOutcome(
                            attempt.Outcome.Kind)),
                    "a logger failure must not reach the World outcome");
                Assert.That(route.Practice.State,
                    Is.EqualTo(PetRobotPracticeState.Feedback),
                    "a logger failure must not change the session state machine");
            }
            finally
            {
                CommonSignalAttemptDiagnostics.Sink = previous;
                CommonSignalAttemptDiagnostics.ResetForTest();
                route.Dispose();
            }
        }

        [Test]
        public void Step10W_NullAttempt_RecordsNothingAndBuildsEmpty()
        {
            CommonSignalAttemptDiagnostics.ResetForTest();
            var lines = new List<string>();
            Action<string> previous = CommonSignalAttemptDiagnostics.Sink;
            try
            {
                CommonSignalAttemptDiagnostics.Sink = lines.Add;
                Assert.That(CommonSignalAttemptDiagnostics.Emit(null), Is.False,
                    "a null attempt must record nothing");
                Assert.That(lines.Count, Is.Zero);
                Assert.That(CommonSignalAttemptDiagnostics.Build(null),
                    Is.EqualTo(string.Empty));
            }
            finally
            {
                CommonSignalAttemptDiagnostics.Sink = previous;
                CommonSignalAttemptDiagnostics.ResetForTest();
            }
        }
    }
}
