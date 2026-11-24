using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.PlayMode
{
    public sealed class LiveMediaPipeDiagnosticHarnessV1Tests
    {
        [UnityTest]
        public IEnumerator DiagnosticPrefab_LoadsInitializesRuntimeAndDoesNotClaimPhysicalProviderReadiness()
        {
            GameObject prefab = Resources.Load<GameObject>("Diagnostics/LiveMediaPipeInputDiagnostic");
            Assert.IsNotNull(prefab);
            GameObject instance = Object.Instantiate(prefab);
            try
            {
                yield return null;
                var bridge = instance.GetComponent<MediaPipeHandPoseObservationBridge>();
                var harness = instance.GetComponent<LiveMediaPipeInputDiagnosticHarness>();
                Assert.IsNotNull(bridge);
                Assert.IsNotNull(harness);
                Assert.IsTrue(harness.RuntimeInitialized, harness.Status);
                Assert.IsFalse(bridge.IsBound, "No HandPoseIntegrator exists in the isolated prefab test scene");
                Assert.IsFalse(bridge.HasReceivedObservation);
                StringAssert.Contains("explicit Start Capture", harness.Status);
            }
            finally
            {
                Object.Destroy(instance);
            }
            yield return null;
        }
    }
}
