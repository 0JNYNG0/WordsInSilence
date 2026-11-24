using System;
using WordsInSilence.Recognition.Core;

namespace WordsInSilence.Recognition.Evaluation
{
    /// <summary>Immutable decision returned by the provisional Step 8C distance gate.</summary>
    public sealed class DistanceGateDecision
    {
        public string PredictedWord { get; }
        public RecognitionSequence WinningPrototype { get; }
        public double BestWordDistance { get; }
        public double MaximumBestWordDistance { get; }
        public bool Accepted { get; }

        public DistanceGateDecision(
            string predictedWord,
            RecognitionSequence winningPrototype,
            double bestWordDistance,
            double maximumBestWordDistance,
            bool accepted)
        {
            PredictedWord = predictedWord;
            WinningPrototype = winningPrototype;
            BestWordDistance = bestWordDistance;
            MaximumBestWordDistance = maximumBestWordDistance;
            Accepted = accepted;
        }
    }

    /// <summary>Pure C# policy layered outside the externally approved RecognitionCore v1.0.</summary>
    public static class DistanceGatePolicy
    {
        public static DistanceGateDecision Evaluate(RecognitionResult result, double maximumBestWordDistance)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            if (double.IsNaN(maximumBestWordDistance) || double.IsInfinity(maximumBestWordDistance) || maximumBestWordDistance < 0.0)
                throw new ArgumentOutOfRangeException(nameof(maximumBestWordDistance));

            WordDistance winner = null;
            foreach (WordDistance candidate in result.WordDistances)
            {
                if (string.Equals(candidate.Word, result.PredictedWord, StringComparison.Ordinal))
                {
                    winner = candidate;
                    break;
                }
            }
            if (winner == null) throw new ArgumentException("recognition result does not contain its predicted word distance", nameof(result));
            if (winner.WinningPrototype == null || result.WinningPrototype == null ||
                !string.Equals(winner.WinningPrototype.SampleId, result.WinningPrototype.SampleId, StringComparison.Ordinal))
                throw new ArgumentException("recognition result winning prototype is inconsistent", nameof(result));

            return new DistanceGateDecision(
                result.PredictedWord,
                result.WinningPrototype,
                winner.Distance,
                maximumBestWordDistance,
                winner.Distance <= maximumBestWordDistance);
        }
    }
}
