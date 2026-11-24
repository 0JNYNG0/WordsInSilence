// M3_GROUND_VISUAL_A10 - PlayMode regression for the placeholder ground visual.
//
// PlayMode is required because the behaviour under test happens in Awake, which does not
// run for a non-ExecuteAlways MonoBehaviour in EditMode. The serialized choice is written
// through reflection while the host GameObject is still INACTIVE, so Awake observes it
// exactly as a scene or prefab override would, and then the object is activated.
//
// The point of these tests is the invariant the fix promises: hiding the placeholder
// changes the RENDERER and nothing else. The collider, its enabled state, the size and the
// position must be identical in both settings, so character collision cannot regress.
//
// Owned objects only. No webcam, no learner storage, no scene load, no recognition and no
// personal data.
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using WordsInSilence.Practice.PetRobotPortfolioQuest;

namespace WordsInSilence.Tests.PlayMode
{
    [TestFixture]
    public sealed class PetRobotPortfolioQuestGroundVisualTests
    {
        const string GroundName = "PortfolioQuestGround";
        static readonly Vector3 ExpectedLocalPosition = new Vector3(0f, -0.25f, 0f);
        static readonly Vector3 ExpectedLocalScale = new Vector3(24f, 0.5f, 18f);

        readonly List<Object> owned = new List<Object>();

        [UnityTearDown]
        public IEnumerator TearDown()
        {
            for (int index = 0; index < owned.Count; index++)
                if (owned[index] != null) Object.Destroy(owned[index]);
            owned.Clear();
            yield return null;
        }

        static void SetPrivateField(object target, string fieldName, object value)
        {
            FieldInfo field = target.GetType().GetField(
                fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field, "PetRobotPortfolioQuestVisuals." + fieldName
                + " was not found; the serialized field name changed and this test must be "
                + "updated rather than skipped");
            field.SetValue(target, value);
        }

        /// <summary>
        /// Builds the visuals component with a chosen serialized state, exactly as a scene
        /// or prefab override supplies it: the host stays inactive until the field is set,
        /// so Awake sees the value.
        /// </summary>
        GameObject CreateVisuals(bool hide, Material material)
        {
            var host = new GameObject("PortfolioQuestPresentationHost");
            owned.Add(host);
            host.SetActive(false);
            var visuals = host.AddComponent<PetRobotPortfolioQuestVisuals>();
            SetPrivateField(visuals, "hideTemporaryGroundVisual", hide);
            SetPrivateField(visuals, "groundVisualMaterial", material);
            host.SetActive(true);          // Awake runs here, with the choice already set
            return host;
        }

        static Transform Ground(GameObject host)
        {
            Transform ground = host.transform.Find(GroundName);
            Assert.IsNotNull(ground, "the placeholder ground child must still be created");
            return ground;
        }

        static void AssertColliderAndGeometryUnchanged(Transform ground)
        {
            var collider = ground.GetComponent<BoxCollider>();
            Assert.IsNotNull(collider, "the collider must never be deleted");
            Assert.IsTrue(collider.enabled, "the collider must stay enabled");
            Assert.IsFalse(collider.isTrigger, "the collider must stay solid");
            Assert.AreEqual(ExpectedLocalPosition, ground.localPosition);
            Assert.AreEqual(ExpectedLocalScale, ground.localScale);
            Assert.IsTrue(ground.gameObject.activeInHierarchy,
                "the GameObject itself is never disabled");
        }

        [UnityTest]
        public IEnumerator HiddenSetting_DisablesOnlyTheRenderer()
        {
            GameObject host = CreateVisuals(true, null);
            yield return null;

            Transform ground = Ground(host);
            AssertColliderAndGeometryUnchanged(ground);

            var renderer = ground.GetComponent<MeshRenderer>();
            Assert.IsNotNull(renderer, "the renderer component still exists, just disabled");
            Assert.IsFalse(renderer.enabled,
                "the hidden setting must disable the renderer and nothing else");
        }

        [UnityTest]
        public IEnumerator DefaultSetting_PreservesTheExistingBehaviourOfOtherScenes()
        {
            GameObject host = CreateVisuals(false, null);
            yield return null;

            Transform ground = Ground(host);
            AssertColliderAndGeometryUnchanged(ground);

            var renderer = ground.GetComponent<MeshRenderer>();
            Assert.IsTrue(renderer.enabled,
                "a scene that sets neither field keeps today's visible placeholder");
        }

        [UnityTest]
        public IEnumerator BothSettings_ProduceIdenticalColliderAndGeometry()
        {
            GameObject hidden = CreateVisuals(true, null);
            GameObject visible = CreateVisuals(false, null);
            yield return null;

            Transform hiddenGround = Ground(hidden);
            Transform visibleGround = Ground(visible);
            var hiddenCollider = hiddenGround.GetComponent<BoxCollider>();
            var visibleCollider = visibleGround.GetComponent<BoxCollider>();

            Assert.AreEqual(visibleGround.localPosition, hiddenGround.localPosition);
            Assert.AreEqual(visibleGround.localScale, hiddenGround.localScale);
            Assert.AreEqual(visibleCollider.enabled, hiddenCollider.enabled);
            Assert.AreEqual(visibleCollider.size, hiddenCollider.size);
            Assert.AreEqual(visibleCollider.center, hiddenCollider.center);
            Assert.AreNotEqual(visibleGround.GetComponent<MeshRenderer>().enabled,
                               hiddenGround.GetComponent<MeshRenderer>().enabled,
                               "only the renderer differs between the two settings");
        }

        [UnityTest]
        public IEnumerator ExplicitMaterial_IsAppliedOnlyWhenNotHidden()
        {
            // build the test material from the shader the primitive already uses, so this
            // never depends on a built-in shader name being resolvable
            GameObject probe = CreateVisuals(false, null);
            yield return null;
            Shader shader = Ground(probe).GetComponent<MeshRenderer>().sharedMaterial.shader;
            Assert.IsNotNull(shader);
            var material = new Material(shader);
            owned.Add(material);

            GameObject visible = CreateVisuals(false, material);
            yield return null;
            Assert.AreSame(material, Ground(visible).GetComponent<MeshRenderer>().sharedMaterial,
                "an explicitly named material replaces the built-in default");

            GameObject hidden = CreateVisuals(true, material);
            yield return null;
            Assert.IsFalse(Ground(hidden).GetComponent<MeshRenderer>().enabled,
                "hiding wins over a supplied material; nothing is drawn either way");
        }

#if UNITY_EDITOR
        /// <summary>
        /// Read-only guard that the exact route override still exists, so a later scene
        /// re-serialization cannot silently drop it. It reads only general scene text and
        /// asserts nothing about any other scene content.
        /// </summary>
        [Test]
        public void RouteScene_StillCarriesTheExactGroundOverride()
        {
            const string scenePath =
                "Assets/_Project/Scenes/VerticalSlice/PetRobotM3RouteVerticalSlice.unity";
            Assert.IsTrue(System.IO.File.Exists(scenePath), "the route scene must exist");
            string text = System.IO.File.ReadAllText(scenePath);

            StringAssert.Contains(
                "target: {fileID: 11400002, guid: 10c00000000000000000000000000010, type: 3}",
                text, "the override must still target the visuals component in the prefab");
            StringAssert.Contains("propertyPath: hideTemporaryGroundVisual", text);
        }
#endif
    }
}
