using System.IO;
using UnityEngine;

namespace WordsInSilence.Recognition.Common
{
    /// <summary>
    /// Resolves the single shared frozen common-2D classifier for the whole app, so the
    /// Practice route and the World route provably use the same implementation and the
    /// same artifact bytes.
    ///
    /// This is a lazily created cache over one immutable, read-only model, not a mutable
    /// singleton: it exposes no setter for production code, performs no writes and does
    /// no dynamic object search or reflection discovery. Tests may inject their own
    /// <see cref="ICommonSignClassifier"/> through the adapter constructor instead.
    /// </summary>
    public static class CommonSignClassifierProvider
    {
        public const string ModelDirectoryName = "CommonModelV1_6";

        static readonly object Gate = new object();
        static FrozenCommonSignClassifier shared;

        public static string DefaultModelDirectory
        {
            get
            {
                return Path.Combine(
                    Application.streamingAssetsPath,
                    "RecognitionCoreV1",
                    ModelDirectoryName);
            }
        }

        /// <summary>
        /// The process-wide frozen classifier. A failed load still returns a classifier
        /// that reports the failure on every call, so callers remain fail-closed.
        /// </summary>
        public static FrozenCommonSignClassifier Shared
        {
            get
            {
                lock (Gate)
                {
                    if (shared == null)
                        shared = FrozenCommonSignClassifier.Load(DefaultModelDirectory);
                    return shared;
                }
            }
        }
    }
}
