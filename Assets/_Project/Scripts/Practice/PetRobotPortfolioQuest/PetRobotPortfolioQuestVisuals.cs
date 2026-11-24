using UnityEngine;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    /// <summary>Step 10C-only visual children; gameplay collision remains on Step 10B anchors.</summary>
    public sealed class PetRobotPortfolioQuestVisuals : MonoBehaviour
    {
        [SerializeField] Transform playerAnchor;
        [SerializeField] Transform petRobotAnchor;
        [SerializeField] Transform educationNpcAnchor;
        [SerializeField] GameObject playerVisualPrefab;
        [SerializeField] GameObject petRobotVisualPrefab;
        [SerializeField] GameObject educationNpcVisualPrefab;

        [Header("Temporary ground placeholder")]
        [Tooltip("Hide the placeholder ground RENDERER only. The collider, its size and its " +
                 "position are unchanged, so character collision is identical. Use this in a " +
                 "scene that already has authored terrain.")]
        [SerializeField] bool hideTemporaryGroundVisual;

        [Tooltip("Optional explicit material for the placeholder ground. Leave empty unless a " +
                 "visible floor is actually required: CreatePrimitive attaches the built-in " +
                 "default material, which has no URP variant and draws magenta.")]
        [SerializeField] Material groundVisualMaterial;

        void Awake()
        {
            AddVisual(playerVisualPrefab, playerAnchor, "PortfolioPlayerVisual", Vector3.one);
            AddVisual(petRobotVisualPrefab, petRobotAnchor, "PortfolioPetRobotVisual", Vector3.one);
            AddVisual(educationNpcVisualPrefab, educationNpcAnchor,
                "PortfolioEducationNpcVisual", Vector3.one);
            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ground.name = "PortfolioQuestGround";
            ground.transform.SetParent(transform, false);
            ground.transform.localPosition = new Vector3(0f, -0.25f, 0f);
            ground.transform.localScale = new Vector3(24f, 0.5f, 18f);
            ApplyGroundVisual(ground);
        }

        /// <summary>
        /// Touches the RENDERER only. The collider CreatePrimitive attached, its size and its
        /// position are left exactly as they were, so character collision is unchanged in
        /// every case.
        ///
        /// CreatePrimitive attaches the built-in default material, which has no URP variant,
        /// so this project draws the placeholder with the magenta error shader. A scene that
        /// already has authored terrain can hide the placeholder; a scene that still needs a
        /// visible floor can name one valid material. Both fields default to off, so a scene
        /// that sets neither keeps the previous behaviour.
        /// </summary>
        void ApplyGroundVisual(GameObject ground)
        {
            MeshRenderer renderer = ground.GetComponent<MeshRenderer>();
            if (renderer == null) return;
            if (hideTemporaryGroundVisual)
            {
                renderer.enabled = false;
                return;
            }
            if (groundVisualMaterial != null)
                renderer.sharedMaterial = groundVisualMaterial;
        }

        static void AddVisual(GameObject source, Transform anchor, string objectName, Vector3 scale)
        {
            if (source == null || anchor == null || anchor.childCount != 0) return;
            GameObject value = Instantiate(source, anchor, false);
            value.name = objectName;
            value.transform.localPosition = Vector3.zero;
            value.transform.localRotation = Quaternion.identity;
            value.transform.localScale = scale;
            foreach (Collider collider in value.GetComponentsInChildren<Collider>(true))
                Destroy(collider);
        }
    }
}

