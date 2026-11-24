using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Core;
using WordsInSilence.Pose;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class LiveMediaPipeInputV1Tests
    {
        const double Tolerance = 1e-12;
        static string ProjectRoot => Directory.GetParent(Application.dataPath).FullName;
        static string FixtureRoot => Path.Combine(Application.dataPath, "_Project/Tests/EditMode/Fixtures/LiveMediaPipeInputV1");
        static string Step9AFixtureRoot => Path.Combine(Application.dataPath, "_Project/Tests/EditMode/Fixtures/RuntimeRecognitionServiceV1");
        static string TemplateDirectory => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1/Templates");
        static string ThresholdPath => Path.Combine(Application.streamingAssetsPath, "RecognitionCoreV1/RuntimeGate/frozen_threshold_config.json");

        [Test]
        public void RawObservation_IsImmutableAndRejectsMalformedCountsMetadataAndFiniteMaskCombinations()
        {
            double[] coordinates = NaNCoordinates(21);
            bool[] valid = new bool[21];
            coordinates[0] = 1.0; coordinates[1] = 2.0; coordinates[2] = 3.0; valid[0] = true;
            var set = new RawLandmarkObservationSet(21, coordinates, valid, new bool[21], new bool[21]);
            coordinates[0] = 99.0; valid[0] = false;
            Assert.AreEqual(1.0, set.Coordinate(0, 0));
            Assert.IsTrue(set.IsValid(0));
            double[] copied = set.CopyCoordinates(); copied[0] = -5.0;
            Assert.AreEqual(1.0, set.Coordinate(0, 0));

            Assert.Throws<ArgumentException>(() => new RawLandmarkObservationSet(21, new double[62], new bool[21], new bool[21], new bool[21]));
            Assert.Throws<ArgumentException>(() => new RawLandmarkObservationSet(21, new double[63], new bool[20], new bool[21], new bool[21]));
            double[] invalidFinite = NaNCoordinates(21); invalidFinite[0] = 0.0;
            Assert.Throws<ArgumentException>(() => new RawLandmarkObservationSet(21, invalidFinite, new bool[21], new bool[21], new bool[21]));
            double[] validNaN = NaNCoordinates(21); bool[] validNaNMask = new bool[21]; validNaNMask[0] = true;
            Assert.Throws<ArgumentException>(() => new RawLandmarkObservationSet(21, validNaN, validNaNMask, new bool[21], new bool[21]));
            Assert.Throws<ArgumentException>(() => new RawObservationConvention("", RawObservationCoordinateSpace.MediaPipeNormalizedImage, "axes", "hands", "mirror"));
            Assert.Throws<ArgumentOutOfRangeException>(() => new RawMediaPipeObservation(double.NaN, RawObservationConvention.LiveMediaPipe, AbsentPose(), AbsentHand(), AbsentHand()));
            Assert.Throws<ArgumentException>(() => new RawMediaPipeObservation(0.0, RawObservationConvention.LiveMediaPipe, AbsentHand(), AbsentHand(), AbsentHand()));
        }

        [Test]
        public void Mapper_UsesDirectLeftRightRawTextureConventionAndTracksReappearanceWithoutAliasing()
        {
            var mapper = new MediaPipeHandPoseObservationMapper();
            HandPoseFrame source = MakeProviderFrame(1.25f, includeLeft: true, includeRight: false);
            RawMediaPipeObservation first = mapper.Map(source);
            Assert.AreEqual(RawObservationConvention.LiveMediaPipe.Id, first.Convention.Id);
            StringAssert.Contains("display-only", first.Convention.Mirror);
            StringAssert.Contains("categoryName", first.Convention.Handedness);
            Assert.IsTrue(first.LeftHand.IsValid(0));
            Assert.IsTrue(first.LeftHand.IsReappearance(0));
            Assert.IsFalse(first.RightHand.IsValid(0));
            double before = first.LeftHand.Coordinate(0, 0);
            source.HandFrame.Hands[0].Points[0].X = 99f;
            Assert.AreEqual(before, first.LeftHand.Coordinate(0, 0));

            RawMediaPipeObservation second = mapper.Map(MakeProviderFrame(1.30f, includeLeft: true, includeRight: true));
            Assert.IsFalse(second.LeftHand.IsReappearance(0));
            Assert.IsTrue(second.RightHand.IsReappearance(0));
            Assert.Throws<InvalidOperationException>(() => mapper.Map(MakeDuplicateLeftProviderFrame(1.35f)));
        }

        [Test]
        public void ApprovedRawFixture_MatchesExactLandmarkOrderMaskFramesAndCoordinatesWithinOneEminus12()
        {
            List<RawMediaPipeObservation> raw = LoadRawFixture();
            RecognitionSequence expected = RecognitionSequenceBinary.LoadFile(Path.Combine(FixtureRoot, "query_GREETING_NORMAL_R01.wisrq"));
            ApprovedPreprocessingResult actual = new ApprovedLivePreprocessingAdapter().Adapt(raw);
            Assert.AreEqual(131, actual.RawFrameCount);
            Assert.AreEqual(expected.Length, actual.NormalizedFrameCount);
            Assert.AreEqual(1915, actual.ValidLandmarkCount);
            Assert.AreEqual(29, actual.InterpolatedLandmarkCount);

            double maximumDelta = 0.0;
            int index = 0;
            foreach (NormalizedRecognitionFrame frame in actual.Frames)
            {
                for (int landmark = 0; landmark < 16; landmark++)
                {
                    Assert.AreEqual(expected.IsValid(index, landmark), FrameValid(frame, landmark), $"mask frame={index} landmark={landmark}");
                    for (int axis = 0; axis < 3; axis++)
                    {
                        double wanted = expected.Coordinate(index, landmark, axis);
                        double observed = FrameCoordinate(frame, landmark, axis);
                        if (FrameValid(frame, landmark)) maximumDelta = Math.Max(maximumDelta, Math.Abs(wanted - observed));
                        else Assert.IsTrue(double.IsNaN(observed), $"invalid coordinate must remain NaN at {index}/{landmark}/{axis}");
                    }
                }
                index++;
            }
            TestContext.WriteLine($"STEP9B_PARITY_MAX_DELTA={maximumDelta:R}");
            Assert.LessOrEqual(maximumDelta, Tolerance);
        }

        [Test]
        public void MockProvider_EndToEndMatchesApprovedRecognitionDecisionAndDistances()
        {
            List<RawMediaPipeObservation> raw = LoadRawFixture();
            RecognitionSequence approved = RecognitionSequenceBinary.LoadFile(Path.Combine(FixtureRoot, "query_GREETING_NORMAL_R01.wisrq"));
            RuntimeRecognitionService service = LoadService();
            RuntimeRecognitionResult expected = service.Recognize(approved.SampleId, approved);
            var controller = new ManualDiagnosticCaptureController(service, new ApprovedLivePreprocessingAdapter());
            const string captureId = "mock-approved-greeting-normal-r01";
            controller.Begin(captureId, RawObservationConvention.ApprovedStep7B);
            foreach (RawMediaPipeObservation observation in raw) controller.AddObservation(captureId, observation);
            DiagnosticRecognitionResult actual = controller.FinalizeCapture(captureId);

            Assert.AreEqual(ManualDiagnosticCaptureState.Completed, controller.State);
            Assert.AreEqual(expected.PredictedWord, actual.PredictedWord);
            Assert.AreEqual(expected.WinningPrototypeId, actual.WinningPrototypeId);
            Assert.That(actual.GreetingDistance, Is.EqualTo(expected.GreetingDistance).Within(Tolerance));
            Assert.That(actual.LoveDistance, Is.EqualTo(expected.LoveDistance).Within(Tolerance));
            Assert.That(actual.ThankYouDistance, Is.EqualTo(expected.ThankYouDistance).Within(Tolerance));
            Assert.That(actual.BestDistance, Is.EqualTo(expected.BestDistance).Within(Tolerance));
            Assert.AreEqual(expected.Threshold, actual.Threshold);
            Assert.AreEqual(expected.Accepted, actual.Accepted);
            Assert.AreEqual(131, actual.RawFrameCount);
            Assert.AreEqual(131, actual.NormalizedFrameCount);
        }

        [Test]
        public void Adapter_MaxGapOneInterpolatesOnceButStrongTimestampAndLongGapsStayNaNWithoutSmoothingOrZeroFill()
        {
            var adapter = new ApprovedLivePreprocessingAdapter();
            var oneGap = new List<RawMediaPipeObservation>
            {
                MakeSynthetic(0.00, leftValue: 1.0, leftValid: true),
                MakeSynthetic(0.03, leftValue: 0.0, leftValid: false),
                MakeSynthetic(0.06, leftValue: 3.0, leftValid: true),
            };
            ApprovedPreprocessingResult filled = adapter.Adapt(oneGap);
            Assert.IsTrue(FrameValid(filled.Frames[1], 4));
            Assert.AreEqual(1, filled.InterpolatedLandmarkCount);
            Assert.AreNotEqual(0.0, FrameCoordinate(filled.Frames[1], 4, 0));

            var strong = new List<RawMediaPipeObservation>
            {
                MakeSynthetic(0.00, 1.0, true),
                MakeSynthetic(0.03, 0.0, false, strongLeft: true),
                MakeSynthetic(0.06, 3.0, true),
            };
            AssertInvalidSelected(adapter.Adapt(strong).Frames[1], 4);

            var timeBoundary = new List<RawMediaPipeObservation>
            {
                MakeSynthetic(0.00, 1.0, true),
                MakeSynthetic(0.20, 0.0, false),
                MakeSynthetic(0.23, 3.0, true),
            };
            AssertInvalidSelected(adapter.Adapt(timeBoundary).Frames[1], 4);

            var longGap = new List<RawMediaPipeObservation>
            {
                MakeSynthetic(0.00, 1.0, true),
                MakeSynthetic(0.03, 0.0, false),
                MakeSynthetic(0.06, 0.0, false),
                MakeSynthetic(0.09, 3.0, true),
            };
            AssertInvalidSelected(adapter.Adapt(longGap).Frames[1], 4);
            AssertInvalidSelected(adapter.Adapt(longGap).Frames[2], 4);
        }

        [Test]
        public void Adapter_RejectsDuplicateOutOfOrderAndConventionChanges()
        {
            var adapter = new ApprovedLivePreprocessingAdapter();
            Assert.Throws<ArgumentException>(() => adapter.Adapt(new[] { MakeSynthetic(1.0, 1.0, true), MakeSynthetic(1.0, 2.0, true) }));
            Assert.Throws<ArgumentException>(() => adapter.Adapt(new[] { MakeSynthetic(1.0, 1.0, true), MakeSynthetic(0.9, 2.0, true) }));
            Assert.Throws<ArgumentException>(() => adapter.Adapt(new[]
            {
                MakeSynthetic(1.0, 1.0, true),
                MakeSynthetic(1.1, 2.0, true, convention: RawObservationConvention.LiveMediaPipe),
            }));
        }

        [Test]
        public void Controller_StateTransitionsMetadataMismatchCancelResetAndProhibitedTransitionsAreDeterministic()
        {
            var controller = new ManualDiagnosticCaptureController(LoadService(), new ApprovedLivePreprocessingAdapter());
            Assert.AreEqual(ManualDiagnosticCaptureState.Idle, controller.State);
            Assert.Throws<InvalidOperationException>(() => controller.FinalizeCapture("missing"));
            controller.Begin("capture-a", RawObservationConvention.ApprovedStep7B);
            Assert.AreEqual(ManualDiagnosticCaptureState.Capturing, controller.State);
            Assert.Throws<InvalidOperationException>(() => controller.Begin("again", RawObservationConvention.ApprovedStep7B));
            Assert.Throws<ArgumentException>(() => controller.AddObservation("wrong", MakeSynthetic(0.0, 1.0, true)));
            controller.AddObservation("capture-a", MakeSynthetic(0.0, 1.0, true));
            Assert.Throws<ArgumentException>(() => controller.AddObservation("capture-a", MakeSynthetic(0.0, 2.0, true)));
            controller.Cancel("capture-a");
            Assert.AreEqual(ManualDiagnosticCaptureState.Cancelled, controller.State);
            Assert.Throws<InvalidOperationException>(() => controller.AddObservation("capture-a", MakeSynthetic(0.1, 2.0, true)));
            controller.Reset();
            Assert.AreEqual(ManualDiagnosticCaptureState.Idle, controller.State);
            Assert.AreEqual(0, controller.RawFrameCount);
            Assert.IsNull(controller.Result);
            Assert.IsNull(controller.Failure);
        }

        [Test]
        public void Controller_NoFrameAndAllInvalidCapturesFailWithTypedEvidence()
        {
            var controller = new ManualDiagnosticCaptureController(LoadService(), new ApprovedLivePreprocessingAdapter());
            controller.Begin("empty", RawObservationConvention.ApprovedStep7B);
            Assert.Throws<InvalidOperationException>(() => controller.FinalizeCapture("empty"));
            Assert.AreEqual(ManualDiagnosticCaptureState.Failed, controller.State);
            Assert.AreEqual(typeof(InvalidOperationException).FullName, controller.Failure.ExceptionType);
            StringAssert.Contains("zero observations", controller.Failure.Message);
            controller.Reset();
            controller.Begin("invalid", RawObservationConvention.ApprovedStep7B);
            controller.AddObservation("invalid", new RawMediaPipeObservation(0.0, RawObservationConvention.ApprovedStep7B, AbsentPose(), AbsentHand(), AbsentHand()));
            Assert.Throws<InvalidOperationException>(() => controller.FinalizeCapture("invalid"));
            Assert.AreEqual(ManualDiagnosticCaptureState.Failed, controller.State);
            StringAssert.Contains("no valid normalized landmarks", controller.Failure.Message);
        }

        [Test]
        public void DiagnosticResult_IsImmutableByPublicContract()
        {
            foreach (PropertyInfo property in typeof(DiagnosticRecognitionResult).GetProperties(BindingFlags.Instance | BindingFlags.Public))
                Assert.IsFalse(property.CanWrite, property.Name);
            foreach (PropertyInfo property in typeof(DiagnosticCaptureFailure).GetProperties(BindingFlags.Instance | BindingFlags.Public))
                Assert.IsFalse(property.CanWrite, property.Name);
            Assert.AreEqual(0, typeof(DiagnosticRecognitionResult).GetFields(BindingFlags.Instance | BindingFlags.Public).Length);
        }

        [Test]
        public void ApprovedStep9A_AllNineReplayDecisionsRemainUnchanged()
        {
            RuntimeRecognitionService service = LoadService();
            var expected = LoadExpectedPredictions(Path.Combine(Step9AFixtureRoot, "test_expected_predictions.csv"));
            string queryDirectory = Path.Combine(Step9AFixtureRoot, "TestQueries");
            string[] paths = Directory.GetFiles(queryDirectory, "*.wisrq", SearchOption.TopDirectoryOnly);
            Assert.AreEqual(9, paths.Length);
            foreach (string path in paths)
            {
                RecognitionSequence query = RecognitionSequenceBinary.LoadFile(path);
                RuntimeRecognitionResult result = service.Recognize(query.SampleId, query);
                string[] row = expected[query.SampleId];
                Assert.AreEqual(row[3], result.PredictedWord, query.SampleId);
                Assert.AreEqual(row[5], result.WinningPrototypeId, query.SampleId);
                Assert.That(result.GreetingDistance, Is.EqualTo(Parse(row[6])).Within(Tolerance));
                Assert.That(result.LoveDistance, Is.EqualTo(Parse(row[7])).Within(Tolerance));
                Assert.That(result.ThankYouDistance, Is.EqualTo(Parse(row[8])).Within(Tolerance));
                Assert.AreEqual(bool.Parse(row[13]), result.Accepted);
            }
        }

        static RuntimeRecognitionService LoadService() => RuntimeRecognitionService.Load(TemplateDirectory, ThresholdPath);

        static List<RawMediaPipeObservation> LoadRawFixture()
        {
            string path = Path.Combine(FixtureRoot, "approved_GREETING_NORMAL_R01.raw9b");
            using (var stream = File.OpenRead(path))
            using (var reader = new BinaryReader(stream))
            {
                Assert.AreEqual("WISRAW9B", new string(reader.ReadChars(8)));
                Assert.AreEqual(1, reader.ReadInt32());
                int frameCount = reader.ReadInt32();
                var frames = new List<RawMediaPipeObservation>(frameCount);
                for (int frame = 0; frame < frameCount; frame++)
                {
                    double timestamp = reader.ReadDouble();
                    RawLandmarkObservationSet pose = ReadSet(reader, 33);
                    RawLandmarkObservationSet left = ReadSet(reader, 21);
                    RawLandmarkObservationSet right = ReadSet(reader, 21);
                    frames.Add(new RawMediaPipeObservation(timestamp, RawObservationConvention.ApprovedStep7B, pose, left, right));
                }
                Assert.AreEqual(stream.Length, stream.Position);
                return frames;
            }
        }

        static RawLandmarkObservationSet ReadSet(BinaryReader reader, int count)
        {
            var coordinates = new double[count * 3];
            var valid = new bool[count];
            var reappearance = new bool[count];
            var strong = new bool[count];
            for (int landmark = 0; landmark < count; landmark++)
            {
                int offset = landmark * 3;
                coordinates[offset] = reader.ReadDouble();
                coordinates[offset + 1] = reader.ReadDouble();
                coordinates[offset + 2] = reader.ReadDouble();
                byte flags = reader.ReadByte();
                valid[landmark] = (flags & 1) != 0;
                reappearance[landmark] = (flags & 2) != 0;
                strong[landmark] = (flags & 4) != 0;
            }
            return new RawLandmarkObservationSet(count, coordinates, valid, reappearance, strong);
        }

        static RawMediaPipeObservation MakeSynthetic(
            double timestamp,
            double leftValue,
            bool leftValid,
            bool strongLeft = false,
            RawObservationConvention convention = null)
        {
            var poseCoordinates = NaNCoordinates(33);
            var poseValid = new bool[33];
            WriteValid(poseCoordinates, poseValid, 11, -1.0, 0.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 12, 1.0, 0.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 23, -1.0, 2.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 24, 1.0, 2.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 13, 0.0, -1.0, 0.0);
            var leftCoordinates = NaNCoordinates(21);
            var leftMask = new bool[21];
            if (leftValid) WriteValid(leftCoordinates, leftMask, 0, leftValue, -1.0, 0.0);
            var strong = new bool[21]; strong[0] = strongLeft;
            return new RawMediaPipeObservation(
                timestamp,
                convention ?? RawObservationConvention.ApprovedStep7B,
                new RawLandmarkObservationSet(33, poseCoordinates, poseValid, new bool[33], new bool[33]),
                new RawLandmarkObservationSet(21, leftCoordinates, leftMask, new bool[21], strong),
                AbsentHand());
        }

        static HandPoseFrame MakeProviderFrame(float timestamp, bool includeLeft, bool includeRight)
        {
            var points = new Vector3[33];
            for (int index = 0; index < points.Length; index++) points[index] = new Vector3(index * 0.01f, index * 0.02f, -index * 0.001f);
            var hands = new List<HandLandmarks>();
            if (includeLeft) hands.Add(MakeHand(true, 0.1f));
            if (includeRight) hands.Add(MakeHand(false, 0.6f));
            return new HandPoseFrame
            {
                Timestamp = timestamp,
                TimestampMs = (long)(timestamp * 1000f),
                PoseFrame = new UpperBodyPoseFrame { Timestamp = timestamp, TimestampMs = (long)(timestamp * 1000f), AllPoints = points, Visibility = Enumerable.Repeat(1f, 33).ToArray() },
                HandFrame = new LandmarkFrame { Timestamp = timestamp, Hands = hands.ToArray() },
            };
        }

        static HandPoseFrame MakeDuplicateLeftProviderFrame(float timestamp)
        {
            HandPoseFrame frame = MakeProviderFrame(timestamp, true, false);
            frame.HandFrame.Hands = new[] { MakeHand(true, 0.1f), MakeHand(true, 0.2f) };
            return frame;
        }

        static HandLandmarks MakeHand(bool left, float offset)
        {
            var points = new LandmarkPoint[21];
            for (int index = 0; index < points.Length; index++) points[index] = new LandmarkPoint { X = offset + index * 0.01f, Y = 0.2f + index * 0.01f, Z = -index * 0.001f };
            return new HandLandmarks { IsLeftHand = left, Points = points, Confidence = 1f };
        }

        static Dictionary<string, string[]> LoadExpectedPredictions(string path)
        {
            return File.ReadAllLines(path).Skip(1).Where(line => !string.IsNullOrWhiteSpace(line))
                .Select(line => line.Split(','))
                .ToDictionary(row => row[0], StringComparer.Ordinal);
        }

        static double Parse(string value) => double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);
        static RawLandmarkObservationSet AbsentPose() => RawLandmarkObservationSet.Absent(33);
        static RawLandmarkObservationSet AbsentHand() => RawLandmarkObservationSet.Absent(21);
        static double[] NaNCoordinates(int count) { var result = new double[count * 3]; for (int index = 0; index < result.Length; index++) result[index] = double.NaN; return result; }
        static void WriteValid(double[] coordinates, bool[] valid, int landmark, double x, double y, double z) { int offset = landmark * 3; coordinates[offset] = x; coordinates[offset + 1] = y; coordinates[offset + 2] = z; valid[landmark] = true; }
        static bool FrameValid(NormalizedRecognitionFrame frame, int landmark) => frame.CopyValidMask()[landmark];
        static double FrameCoordinate(NormalizedRecognitionFrame frame, int landmark, int axis) => frame.CopyCoordinates()[landmark * 3 + axis];
        static void AssertInvalidSelected(NormalizedRecognitionFrame frame, int landmark) { Assert.IsFalse(FrameValid(frame, landmark)); for (int axis = 0; axis < 3; axis++) Assert.IsTrue(double.IsNaN(FrameCoordinate(frame, landmark, axis))); }
    }
}
