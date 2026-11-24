using System;
using System.Collections.Generic;
using System.Linq;

namespace WordsInSilence.Recognition.Core
{
    /// <summary>Threshold-free RecognitionCore v1.0 using the approved nine train prototypes.</summary>
    public sealed class RecognitionCore
    {
        public const double RelativeBand = 0.10;
        static readonly string[] Words = { "GREETING", "LOVE", "THANK_YOU" };
        readonly RecognitionTemplateLibrary library;

        public RecognitionCore(RecognitionTemplateLibrary library)
        {
            this.library = library ?? throw new ArgumentNullException(nameof(library));
        }

        public RecognitionResult Recognize(RecognitionSequence query)
        {
            if (query == null) throw new ArgumentNullException(nameof(query));
            var templateDistances = new List<TemplateDistance>(library.Templates.Count);
            foreach (RecognitionSequence prototype in library.Templates)
            {
                DtwResult dtw = MaskAwareDtw.Compute(query, prototype, RelativeBand);
                templateDistances.Add(new TemplateDistance(prototype, dtw.Distance, dtw.PathLength));
            }

            var wordDistances = new List<WordDistance>(Words.Length);
            foreach (string word in Words)
            {
                TemplateDistance best = null;
                foreach (TemplateDistance candidate in templateDistances.Where(item => item.Prototype.Word == word))
                {
                    if (best == null || candidate.Distance < best.Distance) best = candidate;
                }
                if (best == null) throw new InvalidOperationException($"no prototype exists for word {word}");
                wordDistances.Add(new WordDistance(word, best.Distance, best.Prototype));
            }

            WordDistance winner = wordDistances[0];
            for (int i = 1; i < wordDistances.Count; i++)
            {
                WordDistance candidate = wordDistances[i];
                if (candidate.Distance < winner.Distance ||
                    (candidate.Distance == winner.Distance && string.CompareOrdinal(candidate.Word, winner.Word) < 0))
                    winner = candidate;
            }
            return new RecognitionResult(templateDistances, wordDistances, winner.Word, winner.WinningPrototype);
        }
    }
}
