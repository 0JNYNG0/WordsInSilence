using System;
using System.Collections.Generic;

namespace WordsInSilence.Recognition.Core
{
    /// <summary>Immutable metadata and flattened [T,16,3]/[T,16] payload used by RecognitionCore v1.0.</summary>
    public sealed class RecognitionSequence
    {
        public const int RequiredLandmarkCount = 16;

        readonly double[] coordinates;
        readonly bool[] validMask;

        public string SampleId { get; }
        public string Word { get; }
        public string Speed { get; }
        public string Split { get; }
        public string SourceNpzSha256 { get; }
        public string LogicalPayloadSha256 { get; }
        public string BinaryPayloadSha256 { get; }
        public int Length { get; }
        public int LandmarkCount { get; }

        public RecognitionSequence(
            string sampleId,
            string word,
            string speed,
            string split,
            string sourceNpzSha256,
            string logicalPayloadSha256,
            int length,
            int landmarkCount,
            double[] coordinates,
            bool[] validMask,
            string binaryPayloadSha256 = "")
        {
            if (string.IsNullOrWhiteSpace(sampleId)) throw new ArgumentException("sample ID is required", nameof(sampleId));
            if (string.IsNullOrWhiteSpace(word)) throw new ArgumentException("word is required", nameof(word));
            if (string.IsNullOrWhiteSpace(speed)) throw new ArgumentException("speed is required", nameof(speed));
            if (string.IsNullOrWhiteSpace(split)) throw new ArgumentException("split is required", nameof(split));
            if (!IsLowerHexSha256(sourceNpzSha256)) throw new ArgumentException("source NPZ SHA-256 is invalid", nameof(sourceNpzSha256));
            if (!IsLowerHexSha256(logicalPayloadSha256)) throw new ArgumentException("logical payload SHA-256 is invalid", nameof(logicalPayloadSha256));
            if (!string.IsNullOrEmpty(binaryPayloadSha256) && !IsLowerHexSha256(binaryPayloadSha256))
                throw new ArgumentException("binary payload SHA-256 is invalid", nameof(binaryPayloadSha256));
            if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length), "sequence must not be empty");
            if (landmarkCount != RequiredLandmarkCount)
                throw new ArgumentOutOfRangeException(nameof(landmarkCount), $"landmark count must be {RequiredLandmarkCount}");
            if (coordinates == null) throw new ArgumentNullException(nameof(coordinates));
            if (validMask == null) throw new ArgumentNullException(nameof(validMask));

            int expectedCoordinates = checked(length * landmarkCount * 3);
            int expectedMask = checked(length * landmarkCount);
            if (coordinates.Length != expectedCoordinates)
                throw new ArgumentException($"coordinate shape mismatch: expected {expectedCoordinates}, got {coordinates.Length}", nameof(coordinates));
            if (validMask.Length != expectedMask)
                throw new ArgumentException($"mask shape mismatch: expected {expectedMask}, got {validMask.Length}", nameof(validMask));

            for (int frame = 0; frame < length; frame++)
            for (int landmark = 0; landmark < landmarkCount; landmark++)
            {
                if (!validMask[frame * landmarkCount + landmark]) continue;
                int offset = (frame * landmarkCount + landmark) * 3;
                for (int axis = 0; axis < 3; axis++)
                {
                    double value = coordinates[offset + axis];
                    if (double.IsNaN(value) || double.IsInfinity(value))
                        throw new ArgumentException($"valid coordinate is non-finite at frame={frame}, landmark={landmark}, axis={axis}", nameof(coordinates));
                }
            }

            SampleId = sampleId;
            Word = word;
            Speed = speed;
            Split = split;
            SourceNpzSha256 = sourceNpzSha256;
            LogicalPayloadSha256 = logicalPayloadSha256;
            BinaryPayloadSha256 = binaryPayloadSha256;
            Length = length;
            LandmarkCount = landmarkCount;
            this.coordinates = (double[])coordinates.Clone();
            this.validMask = (bool[])validMask.Clone();
        }

        public double Coordinate(int frame, int landmark, int axis)
        {
            if ((uint)frame >= (uint)Length) throw new ArgumentOutOfRangeException(nameof(frame));
            if ((uint)landmark >= (uint)LandmarkCount) throw new ArgumentOutOfRangeException(nameof(landmark));
            if ((uint)axis >= 3U) throw new ArgumentOutOfRangeException(nameof(axis));
            return coordinates[(frame * LandmarkCount + landmark) * 3 + axis];
        }

        public bool IsValid(int frame, int landmark)
        {
            if ((uint)frame >= (uint)Length) throw new ArgumentOutOfRangeException(nameof(frame));
            if ((uint)landmark >= (uint)LandmarkCount) throw new ArgumentOutOfRangeException(nameof(landmark));
            return validMask[frame * LandmarkCount + landmark];
        }

        public double[] CopyCoordinates() => (double[])coordinates.Clone();
        public bool[] CopyValidMask() => (bool[])validMask.Clone();

        public static bool IsLowerHexSha256(string value)
        {
            if (value == null || value.Length != 64) return false;
            for (int i = 0; i < value.Length; i++)
            {
                char c = value[i];
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
            }
            return true;
        }
    }

    public struct DtwPathPoint
    {
        public int I { get; }
        public int J { get; }

        public DtwPathPoint(int i, int j)
        {
            I = i;
            J = j;
        }
    }

    public sealed class DtwResult
    {
        public double Distance { get; }
        public double CumulativeCost { get; }
        public IReadOnlyList<DtwPathPoint> Path { get; }
        public int PathLength => Path.Count;

        public DtwResult(double distance, double cumulativeCost, IReadOnlyList<DtwPathPoint> path)
        {
            Distance = distance;
            CumulativeCost = cumulativeCost;
            Path = path ?? throw new ArgumentNullException(nameof(path));
        }
    }

    public sealed class TemplateDistance
    {
        public RecognitionSequence Prototype { get; }
        public double Distance { get; }
        public int PathLength { get; }

        public TemplateDistance(RecognitionSequence prototype, double distance, int pathLength)
        {
            Prototype = prototype ?? throw new ArgumentNullException(nameof(prototype));
            Distance = distance;
            PathLength = pathLength;
        }
    }

    public sealed class WordDistance
    {
        public string Word { get; }
        public double Distance { get; }
        public RecognitionSequence WinningPrototype { get; }

        public WordDistance(string word, double distance, RecognitionSequence winningPrototype)
        {
            Word = word;
            Distance = distance;
            WinningPrototype = winningPrototype;
        }
    }

    public sealed class RecognitionResult
    {
        public IReadOnlyList<TemplateDistance> TemplateDistances { get; }
        public IReadOnlyList<WordDistance> WordDistances { get; }
        public string PredictedWord { get; }
        public RecognitionSequence WinningPrototype { get; }

        public RecognitionResult(
            IReadOnlyList<TemplateDistance> templateDistances,
            IReadOnlyList<WordDistance> wordDistances,
            string predictedWord,
            RecognitionSequence winningPrototype)
        {
            TemplateDistances = templateDistances;
            WordDistances = wordDistances;
            PredictedWord = predictedWord;
            WinningPrototype = winningPrototype;
        }
    }
}
