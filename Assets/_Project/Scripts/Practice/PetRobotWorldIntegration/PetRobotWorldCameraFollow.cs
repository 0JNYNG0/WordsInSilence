using UnityEngine;

namespace WordsInSilence.Practice.PetRobotWorldIntegration
{
    public sealed class PetRobotWorldCameraFollow : MonoBehaviour
    {
        [SerializeField] Transform target;
        [SerializeField] Vector3 offset = new Vector3(0f, 5.5f, -7.5f);
        [SerializeField] Vector3 lookOffset = new Vector3(0f, 1f, 0f);

        void LateUpdate()
        {
            if (target == null) return;
            transform.position = target.position + offset;
            transform.LookAt(target.position + lookOffset);
        }
    }
}
