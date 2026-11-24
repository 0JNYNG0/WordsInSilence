using System;

namespace WordsInSilence.Recognition.Runtime
{
    /// <summary>One already-normalized [16,3] feature frame and its exact validity mask.</summary>
    public sealed class NormalizedRecognitionFrame
    {
        public const int LandmarkCount = 16;
        public const int CoordinateCount = LandmarkCount * 3;

        readonly double[] coordinates;
        readonly bool[] validMask;

        public double Timestamp { get; }

        public NormalizedRecognitionFrame(double timestamp, double[] coordinates, bool[] validMask)
        {
            if (double.IsNaN(timestamp) || double.IsInfinity(timestamp))
                throw new ArgumentOutOfRangeException(nameof(timestamp), "timestamp must be finite");
            if (coordinates == null) throw new ArgumentNullException(nameof(coordinates));
            if (validMask == null) throw new ArgumentNullException(nameof(validMask));
            if (coordinates.Length != CoordinateCount)
                throw new ArgumentException($"coordinates must contain exactly {CoordinateCount} values", nameof(coordinates));
            if (validMask.Length != LandmarkCount)
                throw new ArgumentException($"valid mask must contain exactly {LandmarkCount} values", nameof(validMask));

            for (int landmark = 0; landmark < LandmarkCount; landmark++)
            {
                if (!validMask[landmark]) continue;
                for (int axis = 0; axis < 3; axis++)
                {
                    double value = coordinates[landmark * 3 + axis];
                    if (double.IsNaN(value) || double.IsInfinity(value))
                        throw new ArgumentException($"valid coordinate is non-finite at landmark={landmark}, axis={axis}", nameof(coordinates));
                }
            }

            Timestamp = timestamp;
            this.coordinates = (double[])coordinates.Clone();
            this.validMask = (bool[])validMask.Clone();
        }

        public double[] CopyCoordinates() => (double[])coordinates.Clone();
        public bool[] CopyValidMask() => (bool[])validMask.Clone();
    }
}
