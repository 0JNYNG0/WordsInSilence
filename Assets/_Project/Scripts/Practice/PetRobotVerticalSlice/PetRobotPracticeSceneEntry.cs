using UnityEngine;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public sealed class PetRobotPracticeSceneEntry : MonoBehaviour
    {
        [SerializeField] string practicePrefabResource =
            "VerticalSlice/PetRobotSignPracticeVerticalSlice";
        [SerializeField] PetRobotCandidateCameraStartupRoute candidateCameraRoute;
        [SerializeField] Font presentationFont;

        void Awake()
        {
            GameObject prefab = Resources.Load<GameObject>(practicePrefabResource);
            if (prefab == null)
            {
                Debug.LogError("[Step10A] Pet-robot practice prefab is missing: " + practicePrefabResource);
                return;
            }
            GameObject instance = Instantiate(prefab);
            instance.name = "PetRobotSignPracticeVerticalSlice";
            // F30: presentation only. The instance's own Awake has already built the guided UI,
            // so the font is applied to the live instance before its first render. Nothing else
            // about the instantiation, the camera route or the bootstrap changes.
            if (presentationFont != null)
            {
                GuidedSignPracticePresentationController guided =
                    instance.GetComponentInChildren<
                        GuidedSignPracticePresentationController>(true);
                if (guided != null) guided.SetPresentationFont(presentationFont);
            }
            PetRobotPracticeVerticalSliceBootstrap bootstrap =
                instance.GetComponent<PetRobotPracticeVerticalSliceBootstrap>();
            if (bootstrap == null || candidateCameraRoute == null)
            {
                Debug.LogError("[Step10A v1.1] Explicit candidate-only camera route wiring is missing.");
                return;
            }
            bootstrap.ConfigureCandidateCameraRoute(candidateCameraRoute);
        }
    }
}
