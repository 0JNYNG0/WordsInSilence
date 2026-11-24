// F20 EditMode tests for the shared 46-joint recognition pipeline.
//
// Everything here is synthetic and in memory: no camera, no persistentDataPath, no private
// learner data, no model file and no fixture read. The bank, the references and the gates
// are all constructed by the test itself, so no production cutoff is asserted anywhere.
using System;
using NUnit.Framework;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandRecognitionPipelineTests
    {
        const int SourceFrames = 4;
        const string Participant = "participant_a";
        static readonly string[] Words = { "kiwi", "zebra" };   // arbitrary labels, no word rules

        // ---- synthetic material ------------------------------------------------

        static void Arrays(double seed, out double[] coordinates, out bool[] mask)
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            coordinates = new double[SourceFrames * points * dims];
            mask = new bool[SourceFrames * points];
            for (int frame = 0; frame < SourceFrames; frame++)
                for (int point = 0; point < points; point++)
                {
                    int slot = frame * points + point;
                    mask[slot] = true;
                    coordinates[slot * dims] = seed + point * 0.13 + frame * 0.05;
                    coordinates[slot * dims + 1] = seed * 0.5 + point * 0.07 - frame * 0.03;
                    coordinates[slot * dims + 2] = 0.0;
                }
        }

        static FullHandFeatureSequence Sequence(double seed)
        {
            double[] coordinates; bool[] mask;
            Arrays(seed, out coordinates, out mask);
            return FullHandPairGeometry.BuildFeatures(coordinates, mask, SourceFrames);
        }

        static FullHandCommonCapture Capture(double seed)
        {
            double[] coordinates; bool[] mask;
            Arrays(seed, out coordinates, out mask);
            return new FullHandCommonCapture(SourceFrames, coordinates, mask, mask, new bool[mask.Length],
                                             mask.Length, 0, 640, 480, "test_convention");
        }

        static FullHandFeatureSequence UnusableSequence()
        {
            int points = FullHandCommonSchema.PointCount;
            int dims = FullHandCommonSchema.CoordinateDimensions;
            var coordinates = new double[SourceFrames * points * dims];
            for (int index = 0; index < coordinates.Length; index++) coordinates[index] = double.NaN;
            return FullHandPairGeometry.BuildFeatures(coordinates,
                new bool[SourceFrames * points], SourceFrames);
        }

        static FullHandModel Model(int width)
        {
            var mean = new double[width];
            var scale = new double[width];
            var coefficients = new double[width];
            for (int index = 0; index < width; index++)
            {
                mean[index] = 0.1 * index;
                scale[index] = 1.0 + 0.01 * index;
                coefficients[index] = ((index % 3) - 1) * 0.25;
            }
            return new FullHandModel(mean, scale, coefficients, 0.2);
        }

        /// <summary>Two classes, two references each, all explicit and in memory.</summary>
        static FullHandInMemoryCommonBank Bank()
        {
            return new FullHandInMemoryCommonBank("model_test_v1", Words,
                new[] { "common_0", "common_1", "common_2", "common_3" },
                new[] { 0, 0, 1, 1 },
                new[] { Sequence(1.0), Sequence(1.4), Sequence(3.0), Sequence(3.4) },
                Model(FullHandPairConstants.FixedFeatureCount),
                Model(FullHandPairConstants.AlignedFeatureCount));
        }

        static string Hash(int value)
        {
            return value.ToString("x").PadLeft(64, '0');
        }

        static FullHandGateConfig RejectAllGates()
        {
            var gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
                gates[index] = FullHandGate.RejectEverything("gate_reject", FullHandMethods.All[index]);
            return new FullHandGateConfig("config_reject", gates);
        }

        static FullHandGateConfig GatesAt(string id, double[] qualities)
        {
            var gates = new FullHandGate[FullHandMethods.All.Length];
            for (int index = 0; index < gates.Length; index++)
                gates[index] = FullHandGate.AtQualityThreshold(id + "_" + index,
                    FullHandMethods.All[index], qualities[index]);
            return new FullHandGateConfig(id, gates);
        }

        static FullHandSample Query(double seed, string id, int hash)
        {
            return new FullHandSample(id, Participant, Hash(hash), null, Sequence(seed));
        }

        static FullHandSample Reference(double seed, string id, int hash, string label,
                                        string participant)
        {
            return new FullHandSample(id, participant, Hash(hash), label, Sequence(seed));
        }

        static FullHandParticipantReferenceSet Set(params FullHandSample[] samples)
        {
            return new FullHandParticipantReferenceSet(Participant, samples);
        }

        static FullHandRecognitionPipeline Pipeline(FullHandParticipantReferenceSet set)
        {
            return new FullHandRecognitionPipeline(Bank(), RejectAllGates(), Participant, set);
        }

        static double[] Qualities(FullHandBankOutcome outcome)
        {
            var values = new double[FullHandMethods.All.Length];
            for (int index = 0; index < values.Length; index++)
                values[index] = outcome.Decision(FullHandMethods.All[index]).Quality;
            return values;
        }

        // ---- tests -------------------------------------------------------------

        [Test]
        public void BaselineMatchesTheExistingScorerOnANonSelfQuery()
        {
            FullHandInMemoryCommonBank bank = Bank();
            FullHandSample query = Query(7.0, "query_1", 1);
            FullHandEvaluation direct = FullHandScorer.Evaluate(bank, query.Sequence);
            var pipeline = new FullHandRecognitionPipeline(bank, RejectAllGates(), Participant, null);

            FullHandBankOutcome before = pipeline.EvaluateSample(query, null).Before;
            Assert.AreEqual(bank.ReferenceCount, before.EffectiveReferenceCount);
            Assert.AreEqual(0, before.ExcludedCommonReferenceCount);
            foreach (string method in FullHandMethods.All)
            {
                FullHandMethodResult expected = direct.Method(method);
                FullHandMethodDecision actual = before.Decision(method);
                Assert.AreEqual(expected.Status, actual.Status, method);
                Assert.AreEqual(expected.Top1, actual.Top1, method);
                Assert.AreEqual(expected.Score, actual.Score, 0.0, method);
                Assert.AreEqual(expected.Margin, actual.Margin, 0.0, method);
                Assert.AreEqual(expected.RawScores, actual.CopyRawScores(), method);
            }
        }

        [Test]
        public void WithoutOwnReferencesBeforeAndAfterAreIdenticalAndScoredOnce()
        {
            FullHandRecognitionOutcome outcome =
                Pipeline(null).EvaluateSample(Query(7.0, "query_1", 1), null);

            Assert.IsTrue(outcome.ReusedCommonEvaluation);
            Assert.AreEqual(1, outcome.ScoringPasses, "identical input must not be scored twice");
            Assert.AreEqual(0, outcome.After.ParticipantReferenceCount);
            foreach (string method in FullHandMethods.All)
            {
                FullHandMethodDecision before = outcome.Before.Decision(method);
                FullHandMethodDecision after = outcome.After.Decision(method);
                Assert.AreEqual(before.Top1, after.Top1, method);
                Assert.AreEqual(before.Score, after.Score, 0.0, method);
                Assert.AreEqual(before.Decision, after.Decision, method);
            }
        }

        [Test]
        public void OwnReferencesChangeOnlyThePersonalBankAndAreScoredSeparately()
        {
            FullHandSample query = Query(7.0, "query_1", 1);
            FullHandRecognitionOutcome plain = Pipeline(null).EvaluateSample(query, null);
            FullHandRecognitionOutcome withOwn = Pipeline(
                Set(Reference(7.9, "own_0", 20, Words[0], Participant)))
                .EvaluateSample(query, null);

            Assert.IsFalse(withOwn.ReusedCommonEvaluation);
            Assert.AreEqual(2, withOwn.ScoringPasses);
            Assert.AreEqual(plain.QueryFeatureDigest, withOwn.QueryFeatureDigest,
                            "before and after must share the exact query digest");
            Assert.AreEqual(plain.Before.EffectiveReferenceCount,
                            withOwn.Before.EffectiveReferenceCount);
            Assert.AreEqual(withOwn.Before.EffectiveReferenceCount + 1,
                            withOwn.After.EffectiveReferenceCount);
            Assert.AreEqual(1, withOwn.After.ParticipantReferenceCount);
            foreach (string method in FullHandMethods.All)
                Assert.AreEqual(plain.Before.Decision(method).Score,
                                withOwn.Before.Decision(method).Score, 0.0, method);
        }

        [Test]
        public void ASelfAliasReferenceIsExcludedByIdSourceHashOrFeatureDigest()
        {
            FullHandSample query = Query(7.0, "query_1", 1);
            // a renamed, re-hashed but byte-identical capture
            FullHandSample alias = Reference(7.0, "own_renamed", 99, Words[0], Participant);
            Assert.AreEqual(query.FeatureDigest, alias.FeatureDigest);

            FullHandRecognitionOutcome byDigest =
                Pipeline(Set(alias)).EvaluateSample(query, null);
            Assert.AreEqual(1, byDigest.After.ExcludedParticipantReferenceCount);
            Assert.AreEqual(1, byDigest.ScoringPasses, "an excluded alias leaves nothing to add");

            FullHandRecognitionOutcome bySource = Pipeline(
                Set(Reference(8.5, "own_other", 1, Words[0], Participant)))
                .EvaluateSample(query, null);
            Assert.AreEqual(1, bySource.After.ExcludedParticipantReferenceCount);

            FullHandRecognitionOutcome byId = Pipeline(
                Set(Reference(8.5, "query_1", 77, Words[0], Participant)))
                .EvaluateSample(query, null);
            Assert.AreEqual(1, byId.After.ExcludedParticipantReferenceCount);
        }

        [Test]
        public void ACommonReferenceMatchingTheQueryIsExcludedAndCounted()
        {
            // the query is the same capture as common_0, under a different id
            FullHandRecognitionOutcome outcome =
                Pipeline(null).EvaluateSample(Query(1.0, "query_renamed", 5), null);
            Assert.AreEqual(1, outcome.Before.ExcludedCommonReferenceCount);
            Assert.AreEqual(3, outcome.Before.EffectiveReferenceCount);

            FullHandRecognitionOutcome byId =
                Pipeline(null).EvaluateSample(Query(7.0, "common_2", 6), null);
            Assert.AreEqual(1, byId.Before.ExcludedCommonReferenceCount);
        }

        [Test]
        public void AReferenceSetFromAnotherParticipantIsRefused()
        {
            Assert.Throws<FullHandFormatException>(delegate
            {
                Set(Reference(2.0, "own_0", 20, Words[0], "participant_b"));
            });
            var foreignSet = new FullHandParticipantReferenceSet("participant_b",
                new[] { Reference(2.0, "own_0", 20, Words[0], "participant_b") });
            Assert.Throws<FullHandFormatException>(delegate
            {
                new FullHandRecognitionPipeline(Bank(), RejectAllGates(), Participant, foreignSet);
            });
        }

        [Test]
        public void AQueryFromAnotherParticipantIsRefusedBeforeScoring()
        {
            var foreign = new FullHandSample("query_1", "participant_b", Hash(1), null,
                                             Sequence(7.0));
            // the refusal must not depend on holding any current-participant references
            Assert.Throws<FullHandFormatException>(delegate
            {
                Pipeline(null).EvaluateSample(foreign, null);
            });
            Assert.Throws<FullHandFormatException>(delegate
            {
                Pipeline(Set(Reference(7.9, "own_0", 20, Words[0], Participant)))
                    .EvaluateSample(foreign, null);
            });

            FullHandRecognitionOutcome own =
                Pipeline(null).EvaluateSample(Query(7.0, "query_1", 1), null);
            Assert.AreEqual(Participant, own.ParticipantId);
            Assert.IsNotNull(own.Before.Decision(FullHandMethods.Hybrid).Top1);
        }

        [Test]
        public void AReferenceForAnUnsupportedWordIsRefused()
        {
            FullHandParticipantReferenceSet set =
                Set(Reference(2.0, "own_0", 20, "not_in_this_bank", Participant));
            Assert.Throws<FullHandFormatException>(delegate
            {
                new FullHandRecognitionPipeline(Bank(), RejectAllGates(), Participant, set);
            });
        }

        [Test]
        public void DuplicateIdsSourceHashesAndDigestsAreRefused()
        {
            Assert.Throws<FullHandFormatException>(delegate
            {
                Set(Reference(2.0, "own_0", 20, Words[0], Participant),
                    Reference(2.5, "own_0", 21, Words[0], Participant));
            }, "duplicate id");
            Assert.Throws<FullHandFormatException>(delegate
            {
                Set(Reference(2.0, "own_0", 20, Words[0], Participant),
                    Reference(2.5, "own_1", 20, Words[0], Participant));
            }, "duplicate source hash");
            Assert.Throws<FullHandFormatException>(delegate
            {
                Set(Reference(2.0, "own_0", 20, Words[0], Participant),
                    Reference(2.0, "own_1", 21, Words[0], Participant));
            }, "duplicate feature digest");
        }

        [Test]
        public void MalformedIdentifiersAndHashesAreRefused()
        {
            Assert.Throws<FullHandFormatException>(delegate
            {
                new FullHandSample("", Participant, Hash(1), null, Sequence(2.0));
            });
            Assert.Throws<FullHandFormatException>(delegate
            {
                new FullHandSample("has space", Participant, Hash(1), null, Sequence(2.0));
            });
            Assert.Throws<FullHandFormatException>(delegate
            {
                new FullHandSample("ok", Participant, "not-a-sha", null, Sequence(2.0));
            });
            // an unknown source hash stays unknown; it is never invented
            Assert.IsNull(new FullHandSample("ok", Participant, null, null,
                                             Sequence(2.0)).SourceSha256);
        }

        [Test]
        public void ADeclaredTargetChangesMetadataOnly()
        {
            FullHandSample query = Query(7.0, "query_1", 1);
            FullHandRecognitionOutcome plain = Pipeline(null).EvaluateSample(query, null);
            string top = plain.Before.Decision(FullHandMethods.Hybrid).Top1;
            FullHandRecognitionOutcome targeted = Pipeline(null).EvaluateSample(query, top);

            Assert.IsTrue(targeted.TargetSupported);
            Assert.IsTrue(targeted.Before.Decision(FullHandMethods.Hybrid).TargetMatches);
            foreach (string method in FullHandMethods.All)
            {
                Assert.AreEqual(plain.Before.Decision(method).Top1,
                                targeted.Before.Decision(method).Top1, method);
                Assert.AreEqual(plain.Before.Decision(method).Score,
                                targeted.Before.Decision(method).Score, 0.0, method);
                Assert.AreEqual(plain.Before.Decision(method).Decision,
                                targeted.Before.Decision(method).Decision, method);
            }
        }

        [Test]
        public void AnUnsupportedTargetIsReportedAndNeverSucceeds()
        {
            FullHandSample query = Query(7.0, "query_1", 1);
            double[] qualities = Qualities(Pipeline(null).EvaluateSample(query, null).Before);
            var pipeline = new FullHandRecognitionPipeline(Bank(),
                GatesAt("config_permissive", qualities), Participant, null);

            FullHandRecognitionOutcome outcome = pipeline.EvaluateSample(query, "third_game_word");
            Assert.IsFalse(outcome.TargetSupported);
            foreach (string method in FullHandMethods.All)
            {
                FullHandMethodDecision decision = outcome.Before.Decision(method);
                Assert.AreEqual(FullHandDecisions.UnsupportedTarget, decision.Decision, method);
                Assert.IsFalse(decision.TargetSupported, method);
                Assert.IsFalse(decision.TargetMatches, method);
            }
        }

        [Test]
        public void ArbitraryVocabularyLabelsAreHandledWithoutWordRules()
        {
            var pipeline = Pipeline(null);
            string[] vocabulary = pipeline.CopyVocabulary();
            Assert.AreEqual(Words, vocabulary);
            Assert.IsTrue(pipeline.SupportsLabel(Words[1]));
            Assert.IsFalse(pipeline.SupportsLabel("third_game_word"));

            FullHandBankOutcome before =
                pipeline.EvaluateSample(Query(7.0, "query_1", 1), null).Before;
            Assert.Contains(before.Decision(FullHandMethods.MlOnly).Top1, vocabulary);
            Assert.AreEqual(vocabulary, before.Decision(FullHandMethods.MlOnly).CopyClasses());
        }

        [Test]
        public void EveryGateDirectionEqualityAndRejectAllHold()
        {
            FullHandSample query = Query(7.0, "query_1", 1);
            FullHandBankOutcome rejected = Pipeline(null).EvaluateSample(query, null).Before;
            double[] qualities = Qualities(rejected);

            foreach (string method in FullHandMethods.All)
                Assert.AreEqual(FullHandDecisions.Abstained,
                                rejected.Decision(method).Decision, method);

            FullHandBankOutcome equal = new FullHandRecognitionPipeline(Bank(),
                GatesAt("config_equal", qualities), Participant, null)
                .EvaluateSample(query, null).Before;
            var above = new double[qualities.Length];
            for (int index = 0; index < qualities.Length; index++)
                above[index] = qualities[index] + 1.0;
            FullHandBankOutcome tighter = new FullHandRecognitionPipeline(Bank(),
                GatesAt("config_tighter", above), Participant, null)
                .EvaluateSample(query, null).Before;

            foreach (string method in FullHandMethods.All)
            {
                Assert.AreEqual(FullHandDecisions.Accepted, equal.Decision(method).Decision,
                                method + " must accept at equality");
                Assert.AreEqual(FullHandDecisions.Abstained, tighter.Decision(method).Decision,
                                method + " must abstain below its threshold");
            }
            // direction: DTW is a cost, so its quality is the negated score
            FullHandMethodDecision dtw = equal.Decision(FullHandMethods.DtwOnly);
            Assert.AreEqual(-dtw.Score, dtw.Quality, 0.0);
            FullHandMethodDecision ml = equal.Decision(FullHandMethods.MlOnly);
            Assert.AreEqual(ml.Score, ml.Quality, 0.0);
            FullHandMethodDecision hybrid = equal.Decision(FullHandMethods.Hybrid);
            Assert.AreEqual(hybrid.Score, hybrid.Quality, 0.0);

            // a non-finite quality is evidence of nothing and is never accepted
            FullHandGate finite = FullHandGate.AtQualityThreshold("g", FullHandMethods.Hybrid, 0.0);
            Assert.IsTrue(finite.AcceptsQuality(0.0));
            Assert.IsFalse(finite.AcceptsQuality(double.NaN));
            Assert.IsFalse(finite.AcceptsQuality(double.PositiveInfinity));
            Assert.IsFalse(finite.AcceptsQuality(double.NegativeInfinity));
            FullHandGate none = FullHandGate.RejectEverything("g", FullHandMethods.DtwOnly);
            Assert.IsFalse(none.AcceptsQuality(0.0));
            Assert.IsFalse(none.AcceptsQuality(double.NaN));
            Assert.IsFalse(none.AcceptsQuality(double.PositiveInfinity));
            Assert.IsFalse(none.AcceptsQuality(double.NegativeInfinity));
        }

        [Test]
        public void AGateConfigMustCoverExactlyTheThreeMethods()
        {
            Assert.Throws<FullHandFormatException>(delegate
            {
                new FullHandGateConfig("config_short", new[]
                {
                    FullHandGate.RejectEverything("g", FullHandMethods.DtwOnly),
                    FullHandGate.RejectEverything("g", FullHandMethods.MlOnly)
                });
            });
            Assert.Throws<FullHandFormatException>(delegate
            {
                FullHandGate.AtQualityThreshold("g", FullHandMethods.Hybrid, double.NaN);
            });
            Assert.Throws<FullHandFormatException>(delegate
            {
                FullHandGate.RejectEverything("", FullHandMethods.Hybrid);
            });
            Assert.Throws<FullHandFormatException>(delegate
            {
                FullHandGate.RejectEverything("g", "some_other_method");
            });
        }

        [Test]
        public void TechnicalFailuresAreReportedAsFailuresAndNotAsAbstention()
        {
            var incomplete = new FullHandInMemoryCommonBank("model_gap_v1", Words,
                new[] { "common_0", "common_1" }, new[] { 0, 0 },
                new[] { Sequence(1.0), Sequence(1.4) },
                Model(FullHandPairConstants.FixedFeatureCount),
                Model(FullHandPairConstants.AlignedFeatureCount));
            FullHandSample query = Query(7.0, "query_1", 1);
            FullHandBankOutcome missing = new FullHandRecognitionPipeline(incomplete,
                GatesAt("config_permissive", new[] { -1e9, -1e9, -1e9 }), Participant, null)
                .EvaluateSample(query, null).Before;
            foreach (string method in FullHandMethods.All)
            {
                FullHandMethodDecision decision = missing.Decision(method);
                Assert.AreEqual(FullHandDecisions.TechnicalFailure, decision.Decision, method);
                Assert.AreEqual(1, decision.CopyClassesWithoutEvidence().Length, method);
                Assert.IsTrue(double.IsNaN(decision.Quality), method);
            }

            var unusable = new FullHandSample("query_broken", Participant, Hash(9), null,
                                              UnusableSequence());
            FullHandBankOutcome broken = new FullHandRecognitionPipeline(Bank(),
                GatesAt("config_permissive", new[] { -1e9, -1e9, -1e9 }), Participant, null)
                .EvaluateSample(unusable, null).Before;
            foreach (string method in FullHandMethods.All)
                Assert.AreEqual(FullHandDecisions.TechnicalFailure,
                                broken.Decision(method).Decision, method);
        }

        [Test]
        public void SelectionByMethodNameRejectsAnUnknownMode()
        {
            FullHandBankOutcome before =
                Pipeline(null).EvaluateSample(Query(7.0, "query_1", 1), null).Before;
            Assert.IsNotNull(before.Decision(FullHandMethods.Hybrid));
            Assert.AreEqual(FullHandMethods.Hybrid,
                            before.Decision(FullHandMethods.Hybrid).Method);
            Assert.Throws<FullHandFormatException>(delegate { before.Decision("best_of_three"); });
            Assert.Throws<FullHandFormatException>(delegate { before.Decision(null); });
        }

        [Test]
        public void TheCaptureEntryNormalizesOnceAndBothResultsShareThatDigest()
        {
            var pipeline = Pipeline(Set(Reference(7.9, "own_0", 20, Words[0], Participant)));
            FullHandSample normalized = pipeline.NormalizeCapture(Capture(7.0), "query_1", Hash(1));
            FullHandRecognitionOutcome outcome =
                pipeline.Evaluate(Capture(7.0), "query_1", Hash(1), Words[0]);

            Assert.AreEqual(normalized.FeatureDigest, outcome.QueryFeatureDigest);
            Assert.AreEqual("query_1", outcome.QuerySampleId);
            Assert.AreEqual(Participant, outcome.ParticipantId);
            Assert.AreEqual("model_test_v1", outcome.CommonModelId);
            Assert.AreEqual("config_reject", outcome.GateConfigId);
            Assert.AreEqual(FullHandPipelineBanks.CommonOnly, outcome.Before.BankId);
            Assert.AreEqual(FullHandPipelineBanks.CommonPlusParticipant, outcome.After.BankId);
            Assert.AreNotSame(outcome.Before, outcome.After);
        }

        [Test]
        public void InputsAndResultsAreImmutable()
        {
            FullHandInMemoryCommonBank bank = Bank();
            FullHandSample own = Reference(7.9, "own_0", 20, Words[0], Participant);
            FullHandParticipantReferenceSet set = Set(own);
            var pipeline = new FullHandRecognitionPipeline(bank, RejectAllGates(), Participant, set);
            FullHandRecognitionOutcome outcome =
                pipeline.EvaluateSample(Query(7.0, "query_1", 1), null);

            string[] vocabulary = pipeline.CopyVocabulary();
            vocabulary[0] = "MUTATED";
            Assert.AreNotEqual("MUTATED", pipeline.CopyVocabulary()[0]);
            Assert.AreNotEqual("MUTATED", bank.CopyVocabulary()[0]);

            FullHandSample[] samples = set.CopySamples();
            samples[0] = null;
            Assert.IsNotNull(set.At(0));

            FullHandMethodDecision decision = outcome.Before.Decision(FullHandMethods.Hybrid);
            double[] raw = decision.CopyRawScores();
            double original = raw[0];
            raw[0] = 12345.0;
            Assert.AreEqual(original, decision.CopyRawScores()[0], 0.0);
            string[] classes = decision.CopyClasses();
            classes[0] = "MUTATED";
            Assert.AreNotEqual("MUTATED", decision.CopyClasses()[0]);
            FullHandMethodDecision[] decisions = outcome.Before.CopyDecisions();
            decisions[0] = null;
            Assert.IsNotNull(outcome.Before.Decision(FullHandMethods.DtwOnly));

            double[] coefficients = bank.HybridModel.CopyCoefficients();
            coefficients[0] += 1.0;
            Assert.AreNotEqual(coefficients[0], bank.HybridModel.CopyCoefficients()[0]);
        }
    }
}
