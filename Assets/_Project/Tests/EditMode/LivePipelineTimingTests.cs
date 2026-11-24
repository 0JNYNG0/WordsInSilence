// F54 EditMode tests for the passive timing helper. No camera, scene, model or user data.
// These verify the accumulator contract only; the provider/integrator hook placement is a
// source patch and is verified by review, not by these tests.
using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Core;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class LivePipelineTimingTests
    {
        sealed class Clock
        {
            public long Ticks;                  // one tick == one millisecond in these tests
            public long Read() { return Ticks; }
        }

        static LivePipelineTiming.Channel Channel(Clock clock, List<string> lines,
                                                  double intervalMs = 1000.0,
                                                  int slowSpanLimit = 3)
        {
            return new LivePipelineTiming.Channel(
                "test_component", clock.Read, lines.Add, intervalMs,
                LivePipelineTiming.DefaultSlowThresholdMs, slowSpanLimit, 1.0);
        }

        [Test]
        public void AccumulatorTracksCountsMaximaAndResetsOnFlush()
        {
            var clock = new Clock();
            var lines = new List<string>();
            LivePipelineTiming.Channel channel = Channel(clock, lines);

            channel.MarkEntry(10);
            clock.Ticks = 40;                    // a 40 ms gap between Update entries
            channel.MarkEntry(11);
            channel.AddMilliseconds(TimingStage.Detect, 12.0);
            channel.AddMilliseconds(TimingStage.Detect, 30.0);
            channel.Increment(TimingCounter.FrameUpdated);
            channel.Increment(TimingCounter.EmptyResults);
            channel.Observe(TimingValue.TimestampDiffMs, 87.5);
            channel.Observe(TimingValue.TimestampDiffMs, 12.0);

            Assert.AreEqual(2, channel.Count(TimingStage.Detect));
            Assert.AreEqual(30.0, channel.Max(TimingStage.Detect), 1e-9);
            Assert.AreEqual(42.0, channel.Sum(TimingStage.Detect), 1e-9);
            Assert.AreEqual(40.0, channel.Max(TimingValue.EntryGapMs), 1e-9);
            Assert.AreEqual(87.5, channel.Max(TimingValue.TimestampDiffMs), 1e-9);
            Assert.AreEqual(1, channel.Counter(TimingCounter.FrameUpdated));
            Assert.AreEqual(2, channel.Entries);

            long spanStart = clock.Ticks;
            clock.Ticks += 7;
            channel.AddSpan(TimingStage.Pixels, spanStart);
            Assert.AreEqual(7.0, channel.Max(TimingStage.Pixels), 1e-9);

            clock.Ticks = 2000;
            Assert.IsTrue(channel.Flush());
            Assert.AreEqual(1, lines.Count);
            StringAssert.StartsWith(LivePipelineTiming.LogPrefix, lines[0]);
            StringAssert.Contains("entry_gap_max_ms=40", lines[0]);
            StringAssert.Contains("ts_diff_max_ms=87.5", lines[0]);
            StringAssert.Contains("detect(n=2,max_ms=30", lines[0]);
            StringAssert.Contains("utc=", lines[0]);
            StringAssert.Contains("frame=11", lines[0]);

            Assert.AreEqual(0, channel.Count(TimingStage.Detect), "the window resets");
            Assert.AreEqual(0.0, channel.Max(TimingValue.EntryGapMs), 1e-9);
            Assert.AreEqual(0, channel.Counter(TimingCounter.FrameUpdated));
            Assert.AreEqual(0, channel.Entries);
        }

        [Test]
        public void SummariesAreRateLimitedAndSlowSpansAreCapped()
        {
            var clock = new Clock();
            var lines = new List<string>();
            LivePipelineTiming.Channel channel = Channel(clock, lines, 1000.0, 3);
            channel.MarkEntry(1);

            clock.Ticks = 500;
            Assert.IsFalse(channel.Flush(), "a half-second window must not log");
            Assert.AreEqual(0, lines.Count);

            for (int index = 0; index < 8; index++)
                channel.AddMilliseconds(TimingStage.Detect, 400.0);   // all above 150 ms
            Assert.AreEqual(3, channel.SlowSpansLogged, "slow spans are bounded");
            Assert.AreEqual(3, lines.Count);
            foreach (string line in lines) StringAssert.Contains("stage=Detect", line);

            clock.Ticks = 1600;
            Assert.IsTrue(channel.Flush());
            Assert.AreEqual(4, lines.Count);
            clock.Ticks = 1700;
            Assert.IsFalse(channel.Flush(), "at most one summary per interval");
            Assert.AreEqual(4, lines.Count);

            for (int index = 0; index < 5; index++)
                channel.AddMilliseconds(TimingStage.Detect, 400.0);
            Assert.AreEqual(3, channel.SlowSpansLogged, "the cap survives a window reset");
            Assert.AreEqual(4, lines.Count);
        }

        [Test]
        public void DisabledTelemetryProducesNoChannelAndNoOutput()
        {
            // the production flag is opt-in; without it CreateOrNull must hand back null so
            // every call site short-circuits before building any string
            if (!LivePipelineTiming.Enabled)
            {
                Assert.IsNull(LivePipelineTiming.CreateOrNull("hand_provider"));
                Assert.IsNull(LivePipelineTiming.CreateOrNull("integrator"));
            }
            else
            {
                Assert.IsNotNull(LivePipelineTiming.CreateOrNull("hand_provider"),
                                 "the flag is set in this environment");
            }
            Assert.AreEqual("WIS_LIVE_PIPELINE_TIMING", LivePipelineTiming.EnvironmentFlagName);
            Assert.AreEqual("[LiveTiming]", LivePipelineTiming.LogPrefix);
        }

        [Test]
        public void MalformedChannelConstructionIsRejected()
        {
            var clock = new Clock();
            var lines = new List<string>();
            Assert.Throws<ArgumentException>(
                () => new LivePipelineTiming.Channel("", clock.Read, lines.Add));
            Assert.Throws<ArgumentNullException>(
                () => new LivePipelineTiming.Channel("c", null, lines.Add));
            Assert.Throws<ArgumentNullException>(
                () => new LivePipelineTiming.Channel("c", clock.Read, null));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new LivePipelineTiming.Channel("c", clock.Read, lines.Add, 0.0));
        }
    }
}
