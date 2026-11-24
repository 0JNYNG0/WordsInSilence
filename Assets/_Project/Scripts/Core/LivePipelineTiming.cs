// F54 - opt-in PASSIVE timing telemetry. Diagnostics only, no behaviour change.
//
// Enabled only by WIS_LIVE_PIPELINE_TIMING=1. When disabled, CreateOrNull returns null and
// every call site is a null check: no strings, no allocation, no clock reads.
//
// It records durations, counts and booleans ONLY. No pixels, no landmarks, no body
// coordinates, no identifiers and no files. It never retries, restarts, throttles or
// otherwise touches the camera, the detectors, the timestamps or the scores.
using System;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using Debug = UnityEngine.Debug;

namespace WordsInSilence.Core
{
    /// <summary>Bounded, known stage labels. No free-form strings enter the hot path.</summary>
    public enum TimingStage { Pixels = 0, Detect = 1, Dispatch = 2, Total = 3 }

    public enum TimingCounter
    {
        FrameUpdated = 0, FrameSkipped = 1, Detections = 2, EmptyResults = 3,
        Combined = 4, HandOnly = 5, PoseOnly = 6, Mismatch = 7, CameraPlaying = 8,
    }

    public enum TimingValue { EntryGapMs = 0, TimestampDiffMs = 1 }

    public static class LivePipelineTiming
    {
        public const string EnvironmentFlagName = "WIS_LIVE_PIPELINE_TIMING";
        public const string LogPrefix = "[LiveTiming]";
        public const double DefaultIntervalMs = 1000.0;
        public const double DefaultSlowThresholdMs = 150.0;
        public const int DefaultSlowSpanLimit = 24;

        static readonly bool enabledFlag = ReadFlag();

        public static bool Enabled { get { return enabledFlag; } }

        static bool ReadFlag()
        {
            try
            {
                return string.Equals(Environment.GetEnvironmentVariable(EnvironmentFlagName),
                                     "1", StringComparison.Ordinal);
            }
            catch (Exception) { return false; }
        }

        public static long Now() { return Stopwatch.GetTimestamp(); }

        /// <summary>Null unless the flag is set, so a normal run pays one null check.</summary>
        public static Channel CreateOrNull(string component)
        {
            return enabledFlag ? new Channel(component, Now, WriteLine) : null;
        }

        static void WriteLine(string line) { Debug.Log(line); }

        /// <summary>
        /// One component's accumulator. Monotonic Stopwatch ticks only, never scaled time.
        /// A summary is emitted at most once per interval; individual slow spans are emitted
        /// at most SlowSpanLimit times for the whole process lifetime of this channel.
        /// </summary>
        public sealed class Channel
        {
            const int StageCount = 4, CounterCount = 9, ValueCount = 2;

            readonly string component;
            readonly Func<long> clock;
            readonly Action<string> sink;
            readonly double intervalMs;
            readonly double slowThresholdMs;
            readonly int slowSpanLimit;
            readonly double ticksPerMillisecond;
            readonly double[] stageSum = new double[StageCount];
            readonly double[] stageMax = new double[StageCount];
            readonly int[] stageCount = new int[StageCount];
            readonly long[] counters = new long[CounterCount];
            readonly double[] valueMax = new double[ValueCount];
            readonly StringBuilder text = new StringBuilder(320);

            long lastEntryTicks = -1L;
            long lastFlushTicks = -1L;
            int entries;
            int slowSpansLogged;
            int unityFrame;

            public string Component { get { return component; } }
            public int Entries { get { return entries; } }
            public int SlowSpansLogged { get { return slowSpansLogged; } }
            public long Counter(TimingCounter counter) { return counters[(int)counter]; }
            public int Count(TimingStage stage) { return stageCount[(int)stage]; }
            public double Max(TimingStage stage) { return stageMax[(int)stage]; }
            public double Sum(TimingStage stage) { return stageSum[(int)stage]; }
            public double Max(TimingValue value) { return valueMax[(int)value]; }

            public Channel(string component, Func<long> clock, Action<string> sink,
                           double intervalMs = DefaultIntervalMs,
                           double slowThresholdMs = DefaultSlowThresholdMs,
                           int slowSpanLimit = DefaultSlowSpanLimit,
                           double ticksPerMillisecond = 0.0)
            {
                if (string.IsNullOrEmpty(component)) throw new ArgumentException("component");
                if (clock == null) throw new ArgumentNullException("clock");
                if (sink == null) throw new ArgumentNullException("sink");
                if (!(intervalMs > 0.0)) throw new ArgumentOutOfRangeException("intervalMs");
                if (slowSpanLimit < 0) throw new ArgumentOutOfRangeException("slowSpanLimit");
                this.component = component;
                this.clock = clock;
                this.sink = sink;
                this.intervalMs = intervalMs;
                this.slowThresholdMs = slowThresholdMs;
                this.slowSpanLimit = slowSpanLimit;
                this.ticksPerMillisecond = ticksPerMillisecond > 0.0
                    ? ticksPerMillisecond : Stopwatch.Frequency / 1000.0;
            }

            public double ElapsedMilliseconds(long startTicks)
            {
                return (clock() - startTicks) / ticksPerMillisecond;
            }

            /// <summary>Called at EVERY Update entry, even when no new camera frame arrived.</summary>
            public void MarkEntry(int unityFrameCount)
            {
                unityFrame = unityFrameCount;
                long now = clock();
                entries++;
                if (lastEntryTicks >= 0L)
                    Observe(TimingValue.EntryGapMs, (now - lastEntryTicks) / ticksPerMillisecond);
                lastEntryTicks = now;
                if (lastFlushTicks < 0L) lastFlushTicks = now;
            }

            public void AddSpan(TimingStage stage, long startTicks)
            {
                AddMilliseconds(stage, (clock() - startTicks) / ticksPerMillisecond);
            }

            public void AddMilliseconds(TimingStage stage, double milliseconds)
            {
                int index = (int)stage;
                stageSum[index] += milliseconds;
                stageCount[index]++;
                if (milliseconds > stageMax[index]) stageMax[index] = milliseconds;
                if (milliseconds > slowThresholdMs && slowSpansLogged < slowSpanLimit)
                {
                    slowSpansLogged++;
                    sink(LogPrefix + " slow " + component + " stage=" + stage
                         + " ms=" + Format(milliseconds) + " frame=" + unityFrame
                         + " utc=" + Utc());
                }
            }

            public void Increment(TimingCounter counter) { counters[(int)counter]++; }

            public void Observe(TimingValue value, double magnitude)
            {
                int index = (int)value;
                if (magnitude > valueMax[index]) valueMax[index] = magnitude;
            }

            /// <summary>Emits at most one summary per interval, then resets the window.</summary>
            public bool Flush()
            {
                long now = clock();
                if (lastFlushTicks < 0L) { lastFlushTicks = now; return false; }
                double windowMs = (now - lastFlushTicks) / ticksPerMillisecond;
                if (windowMs < intervalMs) return false;
                lastFlushTicks = now;
                sink(BuildSummary(windowMs));
                ResetWindow();
                return true;
            }

            string BuildSummary(double windowMs)
            {
                text.Length = 0;
                text.Append(LogPrefix).Append(' ').Append(component)
                    .Append(" utc=").Append(Utc())
                    .Append(" frame=").Append(unityFrame)
                    .Append(" window_ms=").Append(Format(windowMs))
                    .Append(" entries=").Append(entries)
                    .Append(" entry_gap_max_ms=").Append(Format(valueMax[(int)TimingValue.EntryGapMs]))
                    .Append(" ts_diff_max_ms=").Append(Format(valueMax[(int)TimingValue.TimestampDiffMs]));
                for (int index = 0; index < CounterCount; index++)
                    text.Append(' ').Append(((TimingCounter)index).ToString().ToLowerInvariant())
                        .Append('=').Append(counters[index]);
                for (int index = 0; index < StageCount; index++)
                {
                    text.Append(' ').Append(((TimingStage)index).ToString().ToLowerInvariant())
                        .Append("(n=").Append(stageCount[index])
                        .Append(",max_ms=").Append(Format(stageMax[index]))
                        .Append(",avg_ms=").Append(Format(stageCount[index] == 0
                            ? 0.0 : stageSum[index] / stageCount[index]))
                        .Append(')');
                }
                text.Append(" slow_logged=").Append(slowSpansLogged);
                return text.ToString();
            }

            void ResetWindow()
            {
                Array.Clear(stageSum, 0, StageCount);
                Array.Clear(stageMax, 0, StageCount);
                Array.Clear(stageCount, 0, StageCount);
                Array.Clear(counters, 0, CounterCount);
                Array.Clear(valueMax, 0, ValueCount);
                entries = 0;
            }

            static string Format(double value)
            {
                return value.ToString("0.##", CultureInfo.InvariantCulture);
            }

            static string Utc()
            {
                return DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ",
                                                CultureInfo.InvariantCulture);
            }
        }
    }
}
