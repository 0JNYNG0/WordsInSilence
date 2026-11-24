using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using WordsInSilence.Recognition.Core;

namespace WordsInSilence.Recognition.Runtime
{
    public enum RuntimeSequenceSessionState
    {
        Idle,
        Begun,
        Collecting,
        Finalized,
        Cancelled,
    }

    /// <summary>Immutable metadata selected once when a normalized sequence input session begins.</summary>
    public sealed class RuntimeSequenceMetadata
    {
        public const string RuntimeWord = "UNLABELED";
        public const string RuntimeSpeed = "RUNTIME";
        public const string RuntimeSplit = "runtime";

        public string SessionId { get; }
        public string SampleId { get; }
        public string Word { get; }
        public string Speed { get; }
        public string Split { get; }
        public bool PreserveFixtureProvenance { get; }
        public string FixtureSourceSha256 { get; }
        public string FixtureLogicalSha256 { get; }

        RuntimeSequenceMetadata(
            string sessionId,
            string sampleId,
            string word,
            string speed,
            string split,
            bool preserveFixtureProvenance,
            string fixtureSourceSha256,
            string fixtureLogicalSha256)
        {
            if (string.IsNullOrWhiteSpace(sessionId)) throw new ArgumentException("session ID is required", nameof(sessionId));
            if (string.IsNullOrWhiteSpace(sampleId)) throw new ArgumentException("sample ID is required", nameof(sampleId));
            if (string.IsNullOrWhiteSpace(word)) throw new ArgumentException("word metadata is required", nameof(word));
            if (string.IsNullOrWhiteSpace(speed)) throw new ArgumentException("speed metadata is required", nameof(speed));
            if (string.IsNullOrWhiteSpace(split)) throw new ArgumentException("split metadata is required", nameof(split));
            if (preserveFixtureProvenance &&
                (!RecognitionSequence.IsLowerHexSha256(fixtureSourceSha256) || !RecognitionSequence.IsLowerHexSha256(fixtureLogicalSha256)))
                throw new ArgumentException("fixture replay provenance hashes are invalid");

            SessionId = sessionId;
            SampleId = sampleId;
            Word = word;
            Speed = speed;
            Split = split;
            PreserveFixtureProvenance = preserveFixtureProvenance;
            FixtureSourceSha256 = fixtureSourceSha256 ?? string.Empty;
            FixtureLogicalSha256 = fixtureLogicalSha256 ?? string.Empty;
        }

        public static RuntimeSequenceMetadata RuntimeUnlabeled(string sessionId)
        {
            return new RuntimeSequenceMetadata(
                sessionId, sessionId, RuntimeWord, RuntimeSpeed, RuntimeSplit,
                false, string.Empty, string.Empty);
        }

        public static RuntimeSequenceMetadata FixtureReplay(RecognitionSequence source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            return new RuntimeSequenceMetadata(
                source.SampleId, source.SampleId, source.Word, source.Speed, source.Split,
                true, source.SourceNpzSha256, source.LogicalPayloadSha256);
        }
    }

    /// <summary>Deterministic stateful bridge from normalized frames to the immutable RecognitionSequence contract.</summary>
    public sealed class RecognitionSequenceInputSession
    {
        const string ProvenanceDomain = "WordsInSilence.Runtime.SequenceInputBridge.v1";
        const string LogicalDomain = "WordsInSilence.Runtime.RecognitionSequence.v1";

        readonly List<double> coordinates = new List<double>();
        readonly List<bool> validMask = new List<bool>();
        readonly List<double> timestamps = new List<double>();
        RuntimeSequenceMetadata metadata;

        public RuntimeSequenceSessionState State { get; private set; } = RuntimeSequenceSessionState.Idle;
        public int FrameCount => timestamps.Count;
        public string SessionId => metadata == null ? string.Empty : metadata.SessionId;

        public void Begin(RuntimeSequenceMetadata sessionMetadata)
        {
            if (State != RuntimeSequenceSessionState.Idle)
                throw new InvalidOperationException($"begin is invalid in state {State}");
            metadata = sessionMetadata ?? throw new ArgumentNullException(nameof(sessionMetadata));
            coordinates.Clear();
            validMask.Clear();
            timestamps.Clear();
            State = RuntimeSequenceSessionState.Begun;
        }

        public void AddFrame(string sessionId, NormalizedRecognitionFrame frame)
        {
            RequireSession(sessionId);
            if (State != RuntimeSequenceSessionState.Begun && State != RuntimeSequenceSessionState.Collecting)
                throw new InvalidOperationException($"add-frame is invalid in state {State}");
            if (frame == null) throw new ArgumentNullException(nameof(frame));
            if (timestamps.Count > 0 && frame.Timestamp <= timestamps[timestamps.Count - 1])
                throw new ArgumentException("timestamps must be strictly increasing", nameof(frame));

            timestamps.Add(frame.Timestamp);
            coordinates.AddRange(frame.CopyCoordinates());
            validMask.AddRange(frame.CopyValidMask());
            State = RuntimeSequenceSessionState.Collecting;
        }

        public RecognitionSequence FinalizeSequence(string sessionId)
        {
            RequireSession(sessionId);
            if (State == RuntimeSequenceSessionState.Finalized)
                throw new InvalidOperationException("sequence was already finalized");
            if (State != RuntimeSequenceSessionState.Begun && State != RuntimeSequenceSessionState.Collecting)
                throw new InvalidOperationException($"finalize is invalid in state {State}");
            if (timestamps.Count == 0)
                throw new InvalidOperationException("cannot finalize a sequence with zero frames");

            double[] coordinatePayload = coordinates.ToArray();
            bool[] maskPayload = validMask.ToArray();
            string sourceSha;
            string logicalSha;
            if (metadata.PreserveFixtureProvenance)
            {
                sourceSha = metadata.FixtureSourceSha256;
                logicalSha = metadata.FixtureLogicalSha256;
            }
            else
            {
                sourceSha = ComputeProvenanceSha256(metadata, timestamps, coordinatePayload, maskPayload);
                logicalSha = ComputeLogicalSha256(metadata, coordinatePayload, maskPayload);
            }

            var sequence = new RecognitionSequence(
                metadata.SampleId,
                metadata.Word,
                metadata.Speed,
                metadata.Split,
                sourceSha,
                logicalSha,
                timestamps.Count,
                RecognitionSequence.RequiredLandmarkCount,
                coordinatePayload,
                maskPayload);
            State = RuntimeSequenceSessionState.Finalized;
            return sequence;
        }

        public void Cancel(string sessionId)
        {
            RequireSession(sessionId);
            if (State != RuntimeSequenceSessionState.Begun && State != RuntimeSequenceSessionState.Collecting)
                throw new InvalidOperationException($"cancel is invalid in state {State}");
            State = RuntimeSequenceSessionState.Cancelled;
        }

        public void Reset()
        {
            metadata = null;
            coordinates.Clear();
            validMask.Clear();
            timestamps.Clear();
            State = RuntimeSequenceSessionState.Idle;
        }

        void RequireSession(string sessionId)
        {
            if (State == RuntimeSequenceSessionState.Idle || metadata == null)
                throw new InvalidOperationException("no input session has begun");
            if (!string.Equals(sessionId, metadata.SessionId, StringComparison.Ordinal))
                throw new ArgumentException("session metadata does not match the begun session", nameof(sessionId));
        }

        static string ComputeProvenanceSha256(
            RuntimeSequenceMetadata value,
            IReadOnlyList<double> frameTimestamps,
            double[] coordinatePayload,
            bool[] maskPayload)
        {
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                WriteString(writer, ProvenanceDomain);
                WriteString(writer, value.SessionId);
                WriteString(writer, value.SampleId);
                writer.Write(frameTimestamps.Count);
                for (int index = 0; index < frameTimestamps.Count; index++) writer.Write(BitConverter.DoubleToInt64Bits(frameTimestamps[index]));
                WriteCoordinates(writer, coordinatePayload);
                WriteMask(writer, maskPayload);
                writer.Flush();
                return Sha256(stream.ToArray());
            }
        }

        static string ComputeLogicalSha256(RuntimeSequenceMetadata value, double[] coordinatePayload, bool[] maskPayload)
        {
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                WriteString(writer, LogicalDomain);
                WriteString(writer, value.SampleId);
                WriteString(writer, value.Word);
                WriteString(writer, value.Speed);
                WriteString(writer, value.Split);
                WriteCoordinates(writer, coordinatePayload);
                WriteMask(writer, maskPayload);
                writer.Flush();
                return Sha256(stream.ToArray());
            }
        }

        static void WriteString(BinaryWriter writer, string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }

        static void WriteCoordinates(BinaryWriter writer, double[] values)
        {
            writer.Write(values.Length);
            foreach (double value in values) writer.Write(BitConverter.DoubleToInt64Bits(value));
        }

        static void WriteMask(BinaryWriter writer, bool[] values)
        {
            writer.Write(values.Length);
            foreach (bool value in values) writer.Write(value ? (byte)1 : (byte)0);
        }

        static string Sha256(byte[] payload)
        {
            using (var sha = SHA256.Create())
                return RecognitionSequenceBinary.ToLowerHex(sha.ComputeHash(payload));
        }
    }
}
