using UnityEngine;

namespace WordsInSilence.Practice.PetRobotWorldIntegration
{
    /// <summary>Replaceable primitive art shell used only by the Step 10B world slice.</summary>
    public sealed class PetRobotWorldPlaceholderVisuals : MonoBehaviour
    {
        [SerializeField] Transform playerAnchor;
        [SerializeField] Transform petRobotAnchor;

        void Awake()
        {
            if (!enabled) return;
            if (playerAnchor != null && playerAnchor.childCount == 0)
                CreateVisual(PrimitiveType.Capsule, "PlayerPlaceholder", playerAnchor,
                    new Vector3(0f, 1f, 0f), new Vector3(0.7f, 1f, 0.7f));
            if (petRobotAnchor != null && petRobotAnchor.childCount == 0)
            {
                CreateVisual(PrimitiveType.Cube, "PetRobotBody", petRobotAnchor,
                    new Vector3(0f, 0.65f, 0f), new Vector3(1.1f, 0.8f, 0.8f));
                CreateVisual(PrimitiveType.Sphere, "PetRobotHead", petRobotAnchor,
                    new Vector3(0f, 1.45f, 0f), new Vector3(0.85f, 0.7f, 0.7f));
            }
            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ground.name = "PracticeWorldGround";
            ground.transform.SetParent(transform, false);
            ground.transform.localPosition = new Vector3(0f, -0.25f, 0f);
            ground.transform.localScale = new Vector3(24f, 0.5f, 18f);
        }

        static void CreateVisual(
            PrimitiveType type,
            string objectName,
            Transform parent,
            Vector3 localPosition,
            Vector3 localScale)
        {
            GameObject value = GameObject.CreatePrimitive(type);
            value.name = objectName;
            value.transform.SetParent(parent, false);
            value.transform.localPosition = localPosition;
            value.transform.localScale = localScale;
            Collider collider = value.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
        }
    }
}
