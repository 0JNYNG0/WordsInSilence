using NUnit.Framework;
using WordsInSilence.Settings;

namespace WordsInSilence.Tests.EditMode
{
    public class RecognitionSettingsTests
    {
        [Test]
        public void Default_ReturnsNonNull()
        {
            var settings = RecognitionSettings.Default;
            Assert.IsNotNull(settings);
        }

        [Test]
        public void GetWeights_0Exemplars_FullGlobal()
        {
            var settings = RecognitionSettings.Default;
            settings.GetWeights(0, out float gw, out float pw);
            Assert.AreEqual(1.00f, gw, 0.001f);
            Assert.AreEqual(0.00f, pw, 0.001f);
        }

        [Test]
        public void GetWeights_2Exemplars_FullGlobal()
        {
            var settings = RecognitionSettings.Default;
            settings.GetWeights(2, out float gw, out float pw);
            Assert.AreEqual(1.00f, gw, 0.001f);
            Assert.AreEqual(0.00f, pw, 0.001f);
        }

        [Test]
        public void GetWeights_3Exemplars_MixedWeights()
        {
            var settings = RecognitionSettings.Default;
            settings.GetWeights(3, out float gw, out float pw);
            Assert.AreEqual(0.75f, gw, 0.001f);
            Assert.AreEqual(0.25f, pw, 0.001f);
        }

        [Test]
        public void GetWeights_7Exemplars_MorePersonal()
        {
            var settings = RecognitionSettings.Default;
            settings.GetWeights(7, out float gw, out float pw);
            Assert.AreEqual(0.65f, gw, 0.001f);
            Assert.AreEqual(0.35f, pw, 0.001f);
        }

        [Test]
        public void GetWeights_15Exemplars_MaxPersonal()
        {
            var settings = RecognitionSettings.Default;
            settings.GetWeights(15, out float gw, out float pw);
            Assert.AreEqual(0.60f, gw, 0.001f);
            Assert.AreEqual(0.40f, pw, 0.001f);
        }

        [Test]
        public void GetWeights_SumAlwaysOne()
        {
            var settings = RecognitionSettings.Default;
            for (int i = 0; i <= 20; i++)
            {
                settings.GetWeights(i, out float gw, out float pw);
                Assert.AreEqual(1.0f, gw + pw, 0.001f,
                    $"Weights don't sum to 1 for {i} exemplars");
            }
        }

        [Test]
        public void DefaultTimings_MatchGuideSpec()
        {
            var s = RecognitionSettings.Default;
            Assert.AreEqual(1.5f, s.NeutralHoldForArmed, 0.01f);
            Assert.AreEqual(2.0f, s.NeutralHoldForEnd, 0.01f);
            Assert.AreEqual(6.0f, s.ArmedTimeoutSeconds, 0.01f);
            Assert.AreEqual(8.0f, s.MaxCaptureDurationSeconds, 0.01f);
            Assert.AreEqual(0.35f, s.PreCaptureBufferSeconds, 0.01f);
        }
    }
}
