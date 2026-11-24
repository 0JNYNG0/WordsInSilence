using System;
using System.Collections.Generic;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>Timestamp-bounded immutable raw observation history; no media is retained.</summary>
    public sealed class RawMediaPipeObservationRingBuffer
    {
        readonly double windowSeconds;
        readonly List<RawMediaPipeObservation> observations = new List<RawMediaPipeObservation>();
        RawObservationConvention convention;

        public double WindowSeconds => windowSeconds;
        public int Count => observations.Count;
        public double OldestTimestamp => observations.Count == 0 ? double.NaN : observations[0].TimestampSeconds;
        public double NewestTimestamp => observations.Count == 0
            ? double.NaN
            : observations[observations.Count - 1].TimestampSeconds;

        public RawMediaPipeObservationRingBuffer(double windowSeconds)
        {
            if (!IsFinite(windowSeconds) || windowSeconds <= 0d)
                throw new ArgumentOutOfRangeException(nameof(windowSeconds));
            this.windowSeconds = windowSeconds;
        }

        public void Add(RawMediaPipeObservation observation)
        {
            if (observation == null) throw new ArgumentNullException(nameof(observation));
            if (!LiveObservationConventionPolicy.IsAccepted(observation))
                throw new ArgumentException("raw pre-roll accepts only an accepted live MediaPipe convention: " + LiveObservationConventionPolicy.RejectionMessage, nameof(observation));
            if (convention == null) convention = observation.Convention;
            else if (!convention.Equals(observation.Convention))
                throw new ArgumentException("raw observation convention changed within the ring", nameof(observation));
            if (observations.Count > 0 && observation.TimestampSeconds <= NewestTimestamp)
                throw new ArgumentException("raw ring timestamps must be strictly increasing", nameof(observation));

            observations.Add(observation);
            double minimumTimestamp = observation.TimestampSeconds - windowSeconds;
            int removeCount = 0;
            while (removeCount < observations.Count &&
                   observations[removeCount].TimestampSeconds < minimumTimestamp)
                removeCount++;
            if (removeCount > 0) observations.RemoveRange(0, removeCount);
        }

        public IReadOnlyList<RawMediaPipeObservation> Snapshot()
        {
            return Array.AsReadOnly(observations.ToArray());
        }

        public void Clear()
        {
            observations.Clear();
            convention = null;
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
