using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using WordsInSilence.Recognition.Diagnostics;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class Step9DRawIntermediateDiagnosticInstrumentationV1Tests
    {
        [Test]
        public void Adapt_AndAdaptWithTrace_ResultParityIsExact()
        {
            var adapter = new ApprovedLivePreprocessingAdapter();
            List<RawMediaPipeObservation> observations = StandardSequence();
            ApprovedPreprocessingResult ordinary = adapter.Adapt(observations);
            ApprovedPreprocessingTraceResult traced = adapter.AdaptWithTrace(observations);
            AssertResultsExact(ordinary, traced.Result);
        }

        [Test]
        public void RawOrdering_IsPose33ThenLeft21ThenRight21()
        {
            var pose = new Dictionary<int, double[]> { { 0, Point(100.0, 101.0, 102.0) } };
            ApprovedPreprocessingTraceResult traced = new ApprovedLivePreprocessingAdapter().AdaptWithTrace(new[]
            {
                Observation(0.0, true, 200.0, 300.0, false, false, pose),
            });
            double[] raw = traced.Trace.CopyRawCoordinates();
            Assert.AreEqual(100.0, raw[(0 * 75 + 0) * 3], 0.0);
            Assert.AreEqual(200.0, raw[(0 * 75 + 33) * 3], 0.0);
            Assert.AreEqual(300.0, raw[(0 * 75 + 54) * 3], 0.0);
            CollectionAssert.AreEqual(new[] { "pose[0..32]", "left_hand[0..20]", "right_hand[0..20]" }, traced.Trace.CopyRawLandmarkOrdering());
        }

        [Test]
        public void AbsentLandmark_IsNaNFalseAndSerializesAsJsonNull()
        {
            Step9DRawIntermediateDiagnosticSnapshot snapshot = Snapshot(new[] { Observation(0.0, true) });
            double[] raw = snapshot.Trace.CopyRawCoordinates();
            bool[] valid = snapshot.Trace.CopyRawOriginalValidMask();
            Assert.IsTrue(double.IsNaN(raw[(33 + 1) * 3]));
            Assert.IsFalse(valid[33 + 1]);
            string json = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            StringAssert.Contains("null", json);
            StringAssert.DoesNotContain("NaN", json);
            StringAssert.DoesNotContain("Infinity", json);
        }

        [Test]
        public void BoundaryMasks_KeepReappearanceStrongTimestampAndCombinedSeparate()
        {
            var adapter = new ApprovedLivePreprocessingAdapter();
            ApprovedPreprocessingTraceResult traced = adapter.AdaptWithTrace(new[]
            {
                Observation(0.00, true, 0.0, 0.0),
                Observation(0.05, true, 1.0, 1.0, true, true),
                Observation(0.20, true, 2.0, 2.0),
            });
            bool[] reappearance = traced.Trace.CopyRawReappearanceMask();
            bool[] strong = traced.Trace.CopyRawStrongDepthMask();
            bool[] timestamp = traced.Trace.CopyTimestampBoundaryMask();
            bool[] combined = traced.Trace.CopyCombinedBoundaryMask();
            Assert.IsTrue(reappearance[1 * 75 + 33]);
            Assert.IsTrue(strong[1 * 75 + 54]);
            Assert.IsTrue(timestamp[2 * 75 + 33]);
            Assert.IsTrue(combined[1 * 75 + 33]);
            Assert.IsTrue(combined[1 * 75 + 54]);
            Assert.IsTrue(combined[2 * 75 + 33]);
        }

        [Test]
        public void MaxGapOne_InterpolatesSingleGapAndRejectsTwoFrameGap()
        {
            var adapter = new ApprovedLivePreprocessingAdapter();
            ApprovedPreprocessingTraceResult oneGap = adapter.AdaptWithTrace(new[]
            {
                Observation(0.00, true, 0.0),
                Observation(0.03, true),
                Observation(0.06, true, 2.0),
            });
            int middleLeftWrist = 1 * 75 + 33;
            Assert.IsTrue(oneGap.Trace.CopyInterpolationMask()[middleLeftWrist]);
            Assert.AreEqual(1.0, oneGap.Trace.CopyPostGapInterpolationCoordinates()[middleLeftWrist * 3], 0.0);

            ApprovedPreprocessingTraceResult twoGap = adapter.AdaptWithTrace(new[]
            {
                Observation(0.00, true, 0.0),
                Observation(0.02, true),
                Observation(0.04, true),
                Observation(0.06, true, 3.0),
            });
            Assert.IsFalse(twoGap.Trace.CopyInterpolationMask()[1 * 75 + 33]);
            Assert.IsFalse(twoGap.Trace.CopyInterpolationMask()[2 * 75 + 33]);
        }

        [Test]
        public void TorsoProvenance_DistinguishesDirectCarryBackfillAndUnavailable()
        {
            var adapter = new ApprovedLivePreprocessingAdapter();
            ApprovedPreprocessingTraceResult mixed = adapter.AdaptWithTrace(new[]
            {
                Observation(0.00, false),
                Observation(0.03, true),
                Observation(0.06, false),
            });
            CollectionAssert.AreEqual(
                new[]
                {
                    Step9DTorsoTransformProvenance.LEADING_BACKFILLED,
                    Step9DTorsoTransformProvenance.DIRECT,
                    Step9DTorsoTransformProvenance.FORWARD_CARRIED,
                },
                mixed.Trace.CopyTorsoProvenance());
            ApprovedPreprocessingTraceResult unavailable = adapter.AdaptWithTrace(new[]
            {
                Observation(0.00, false),
                Observation(0.03, false),
            });
            CollectionAssert.AreEqual(
                new[] { Step9DTorsoTransformProvenance.UNAVAILABLE, Step9DTorsoTransformProvenance.UNAVAILABLE },
                unavailable.Trace.CopyTorsoProvenance());
        }

        [Test]
        public void TorsoAndNormalizedTrace_AreSameCoreOutputs()
        {
            ApprovedPreprocessingTraceResult traced = new ApprovedLivePreprocessingAdapter().AdaptWithTrace(StandardSequence());
            double[] tracedCoordinates = traced.Trace.CopyNormalizedCoordinates();
            bool[] tracedMask = traced.Trace.CopyNormalizedMask();
            for (int frame = 0; frame < traced.Result.Frames.Count; frame++)
            {
                double[] coordinates = traced.Result.Frames[frame].CopyCoordinates();
                bool[] mask = traced.Result.Frames[frame].CopyValidMask();
                for (int index = 0; index < coordinates.Length; index++) AssertDoubleExact(coordinates[index], tracedCoordinates[frame * 48 + index]);
                for (int index = 0; index < mask.Length; index++) Assert.AreEqual(mask[index], tracedMask[frame * 16 + index]);
            }
            Assert.AreEqual(0.0, traced.Trace.CopyTorsoOrigin()[0], 0.0);
            Assert.AreEqual(2.0, traced.Trace.CopyTorsoShoulderWidthScale()[0], 0.0);
        }

        [Test]
        public void TracePayloadCopies_AreDeeplyImmutable()
        {
            Step9DRawIntermediatePreprocessingTrace trace = new ApprovedLivePreprocessingAdapter().AdaptWithTrace(StandardSequence()).Trace;
            double before = trace.CopyRawCoordinates()[0];
            double[] first = trace.CopyRawCoordinates();
            first[0] = 99999.0;
            bool[] mask = trace.CopyRawOriginalValidMask();
            mask[0] = !mask[0];
            Step9DRawIntermediatePreprocessingTrace copy = trace.DeepCopy();
            AssertDoubleExact(before, trace.CopyRawCoordinates()[0]);
            AssertDoubleExact(before, copy.CopyRawCoordinates()[0]);
            Assert.AreEqual(trace.CopyRawOriginalValidMask()[0], copy.CopyRawOriginalValidMask()[0]);
        }

        [Test]
        public void Collector_BeginFinalizeCancelAndSnapshotRemainFrozen()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run-a", "capture-a", "GREETING");
            RecordUnavailable(collector, Observation(0.0, true), 0);
            RecordUnavailable(collector, Observation(0.03, true), 1);
            Step9DRawIntermediateDiagnosticSnapshot snapshot = collector.FinalizeCapture();
            string frozen = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            Assert.AreEqual(1, collector.AdaptWithTraceInvocationCount);
            collector.Begin("run-b", "capture-b", "LOVE");
            RecordUnavailable(collector, Observation(1.0, false), 0);
            collector.Cancel();
            Assert.AreEqual(frozen, Step9DRawIntermediateEvidenceJson.Serialize(snapshot));
            Assert.AreEqual(1, collector.CompletedSnapshots.Count);
            Assert.AreEqual(0, collector.BufferedObservationCount);
        }

        [Test]
        public void FrameIdentity_CountTimestampAndOrderingAreValidated()
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run", "capture", "GREETING");
            RawMediaPipeObservation first = Observation(0.0, true);
            Step9DRawIntermediateDiagnosticException mismatch = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                collector.RecordObservation(first, Step9DRawIntermediateFrameIdentity.WithoutProviderIdentity(1, 0.0)));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.FRAME_IDENTITY_MISMATCH, mismatch.Code);
            collector.RecordObservation(first, new Step9DRawIntermediateFrameIdentity(0, 0.0, 10, 20, 30));
            collector.RecordObservation(Observation(0.03, true), new Step9DRawIntermediateFrameIdentity(1, 0.03, 10, 21, 31));
            Step9DRawIntermediateDiagnosticException ordering = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() => collector.FinalizeCapture());
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.FRAME_IDENTITY_MISMATCH, ordering.Code);
        }

        [Test]
        public void DeterministicSerializer_SameSnapshotIsByteExactAndFinalNewline()
        {
            Step9DRawIntermediateDiagnosticSnapshot snapshot = Snapshot(StandardSequence());
            string first = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            string second = Step9DRawIntermediateEvidenceJson.Serialize(snapshot);
            Assert.AreEqual(first, second);
            Assert.IsTrue(first.EndsWith("\n", StringComparison.Ordinal));
        }

        [Test]
        public void EvidenceWriter_UsesDetachedChecksumAndNeverOverwrites()
        {
            string directory = Path.Combine(Path.GetTempPath(), "wis-step9d-instrumentation-" + Guid.NewGuid().ToString("N"));
            string path = Path.Combine(directory, "capture.json");
            try
            {
                Step9DRawIntermediateDiagnosticSnapshot snapshot = Snapshot(StandardSequence());
                Step9DRawIntermediateEvidenceReceipt receipt = Step9DRawIntermediateEvidenceJson.Export(path, snapshot);
                Assert.IsTrue(receipt.SchemaValidated);
                Assert.IsTrue(receipt.ChecksumVerified);
                Assert.IsTrue(Step9DRawIntermediateEvidenceJson.Verify(path, path + ".sha256"));
                Step9DRawIntermediateDiagnosticException overwrite = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                    Step9DRawIntermediateEvidenceJson.Export(path, snapshot));
                Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.OUTPUT_ALREADY_EXISTS, overwrite.Code);
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
            }
        }

        [Test]
        public void TraceDisabledAdapt_PerformsNoDiagnosticDiskIo()
        {
            string directory = Path.Combine(Path.GetTempPath(), "wis-step9d-no-io-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            try
            {
                string[] before = Directory.GetFiles(directory);
                ApprovedPreprocessingResult result = new ApprovedLivePreprocessingAdapter().Adapt(StandardSequence());
                string[] after = Directory.GetFiles(directory);
                Assert.AreEqual(2, result.RawFrameCount);
                CollectionAssert.AreEqual(before, after);
            }
            finally
            {
                Directory.Delete(directory, true);
            }
        }

        [Test]
        public void InvalidShapeTimestampAndInfinity_UseTypedFailures()
        {
            Step9DRawIntermediateDiagnosticException shape = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                Step9DRawIntermediateDiagnosticContract.ValidateFlatLength("raw", 224, 1, 75, 3));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.INVALID_SHAPE, shape.Code);
            Step9DRawIntermediateDiagnosticException infinity = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                Step9DRawIntermediateDiagnosticContract.RejectInfinity("value", double.PositiveInfinity));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.NONFINITE_VALUE, infinity.Code);
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("run", "capture", "GREETING");
            RecordUnavailable(collector, Observation(1.0, true), 0);
            Step9DRawIntermediateDiagnosticException timestamp = Assert.Throws<Step9DRawIntermediateDiagnosticException>(() =>
                RecordUnavailable(collector, Observation(0.5, true), 1));
            Assert.AreEqual(Step9DRawIntermediateDiagnosticFailureCode.TIMESTAMP_REGRESSION, timestamp.Code);
        }

        static List<RawMediaPipeObservation> StandardSequence() => new List<RawMediaPipeObservation>
        {
            Observation(0.00, true, 0.0, 0.0),
            Observation(0.03, true, 0.2, 0.2),
        };

        static Step9DRawIntermediateDiagnosticSnapshot Snapshot(IReadOnlyList<RawMediaPipeObservation> observations)
        {
            var collector = new Step9DRawIntermediateDiagnosticCollector();
            collector.Begin("synthetic-source-review-run", "synthetic-source-review-capture", "GREETING");
            for (int index = 0; index < observations.Count; index++) RecordUnavailable(collector, observations[index], index);
            return collector.FinalizeCapture();
        }

        static void RecordUnavailable(Step9DRawIntermediateDiagnosticCollector collector, RawMediaPipeObservation observation, int index) =>
            collector.RecordObservation(observation, Step9DRawIntermediateFrameIdentity.WithoutProviderIdentity(index, observation.TimestampSeconds));

        static RawMediaPipeObservation Observation(
            double timestamp,
            bool directTorso,
            double? leftWristX = null,
            double? rightWristX = null,
            bool leftReappearance = false,
            bool rightStrong = false,
            Dictionary<int, double[]> extraPose = null)
        {
            var posePoints = new Dictionary<int, double[]>();
            if (directTorso)
            {
                posePoints[11] = Point(-1.0, 0.0, 0.0);
                posePoints[12] = Point(1.0, 0.0, 0.0);
                posePoints[23] = Point(-0.5, 1.0, 0.0);
                posePoints[24] = Point(0.5, 1.0, 0.0);
                posePoints[13] = Point(-1.2, 0.4, 0.0);
                posePoints[14] = Point(1.2, 0.4, 0.0);
                posePoints[15] = Point(-1.3, 0.8, 0.0);
                posePoints[16] = Point(1.3, 0.8, 0.0);
            }
            if (extraPose != null)
                foreach (KeyValuePair<int, double[]> pair in extraPose) posePoints[pair.Key] = pair.Value;
            var leftPoints = new Dictionary<int, double[]>();
            if (leftWristX.HasValue) leftPoints[0] = Point(leftWristX.Value, 0.25, 0.1);
            var rightPoints = new Dictionary<int, double[]>();
            if (rightWristX.HasValue) rightPoints[0] = Point(rightWristX.Value, 0.25, -0.1);
            return new RawMediaPipeObservation(
                timestamp,
                RawObservationConvention.ApprovedStep7B,
                Set(33, posePoints, null, null),
                Set(21, leftPoints, leftReappearance ? new HashSet<int> { 0 } : null, null),
                Set(21, rightPoints, null, rightStrong ? new HashSet<int> { 0 } : null));
        }

        static RawLandmarkObservationSet Set(
            int count,
            Dictionary<int, double[]> points,
            HashSet<int> reappearance,
            HashSet<int> strong)
        {
            var coordinates = new double[count * 3];
            for (int index = 0; index < coordinates.Length; index++) coordinates[index] = double.NaN;
            var valid = new bool[count];
            var reappearanceMask = new bool[count];
            var strongMask = new bool[count];
            foreach (KeyValuePair<int, double[]> pair in points)
            {
                valid[pair.Key] = true;
                Array.Copy(pair.Value, 0, coordinates, pair.Key * 3, 3);
            }
            if (reappearance != null) foreach (int index in reappearance) reappearanceMask[index] = true;
            if (strong != null) foreach (int index in strong) strongMask[index] = true;
            return new RawLandmarkObservationSet(count, coordinates, valid, reappearanceMask, strongMask);
        }

        static double[] Point(double x, double y, double z) => new[] { x, y, z };

        static void AssertResultsExact(ApprovedPreprocessingResult left, ApprovedPreprocessingResult right)
        {
            Assert.AreEqual(left.RawFrameCount, right.RawFrameCount);
            Assert.AreEqual(left.NormalizedFrameCount, right.NormalizedFrameCount);
            Assert.AreEqual(left.ValidLandmarkCount, right.ValidLandmarkCount);
            Assert.AreEqual(left.InterpolatedLandmarkCount, right.InterpolatedLandmarkCount);
            for (int frame = 0; frame < left.Frames.Count; frame++)
            {
                AssertDoubleExact(left.Frames[frame].Timestamp, right.Frames[frame].Timestamp);
                CollectionAssert.AreEqual(left.Frames[frame].CopyValidMask(), right.Frames[frame].CopyValidMask());
                double[] leftCoordinates = left.Frames[frame].CopyCoordinates();
                double[] rightCoordinates = right.Frames[frame].CopyCoordinates();
                for (int index = 0; index < leftCoordinates.Length; index++) AssertDoubleExact(leftCoordinates[index], rightCoordinates[index]);
            }
        }

        static void AssertDoubleExact(double expected, double actual)
        {
            if (double.IsNaN(expected)) Assert.IsTrue(double.IsNaN(actual));
            else Assert.AreEqual(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual));
        }
    }
}
