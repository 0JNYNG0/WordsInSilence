using UnityEngine;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    /// <summary>
    /// Moves the existing pet-robot interaction anchor after player movement.
    /// Visual-local animation remains owned by the child robot Animator.
    /// </summary>
    public sealed class PetRobotFloatingCompanionController : MonoBehaviour
    {
        const float DefaultSideOffset = 1.15f;
        const float DefaultHeightOffset = 1.25f;
        const float DefaultForwardOffset = -0.15f;
        const float DefaultFollowSharpness = 8f;
        const float DefaultRotationSharpness = 8f;
        const float DefaultSnapDistance = 6f;

        [SerializeField] Transform playerTransform;
        [SerializeField] Transform companionAnchor;
        [SerializeField] float sideOffset = DefaultSideOffset;
        [SerializeField] float heightOffset = DefaultHeightOffset;
        [SerializeField] float forwardOffset = DefaultForwardOffset;
        [SerializeField] float followSharpness = DefaultFollowSharpness;
        [SerializeField] float rotationSharpness = DefaultRotationSharpness;
        [SerializeField] float snapDistance = DefaultSnapDistance;

        public Transform PlayerTransform => playerTransform;
        public Transform CompanionAnchor => companionAnchor != null ? companionAnchor : transform;
        public float SideOffset => sideOffset;
        public float HeightOffset => heightOffset;
        public float ForwardOffset => forwardOffset;
        public float FollowSharpness => followSharpness;
        public float RotationSharpness => rotationSharpness;
        public float SnapDistance => snapDistance;

        void LateUpdate()
        {
            Follow(Time.deltaTime);
        }

        bool Follow(float deltaTime)
        {
            if (playerTransform == null || !IsFinite(deltaTime) || deltaTime <= 0f)
                return false;

            Transform anchor = CompanionAnchor;
            Vector3 playerPosition = playerTransform.position;
            Quaternion playerRotation = playerTransform.rotation;
            if (!IsFinite(playerPosition) || !IsFinite(playerRotation)) return false;

            float side = FiniteOrZero(sideOffset);
            float height = FiniteOrZero(heightOffset);
            float forward = FiniteOrZero(forwardOffset);
            Vector3 targetPosition = playerPosition + playerTransform.right * side +
                playerTransform.up * height + playerTransform.forward * forward;
            if (!IsFinite(targetPosition)) return false;

            float safeSnapDistance = NonNegativeOrZero(snapDistance);
            Vector3 currentPosition = anchor.position;
            Vector3 difference = targetPosition - currentPosition;
            float distanceSquared = difference.sqrMagnitude;
            float snapSquared = safeSnapDistance * safeSnapDistance;
            bool snap = !IsFinite(currentPosition) || !IsFinite(distanceSquared) ||
                !IsFinite(snapSquared) || distanceSquared > snapSquared;

            if (snap)
            {
                anchor.position = targetPosition;
                anchor.rotation = playerRotation;
                return true;
            }

            float positionFactor = ExponentialFactor(followSharpness, deltaTime);
            float rotationFactor = ExponentialFactor(rotationSharpness, deltaTime);
            if (positionFactor > 0f)
                anchor.position = Vector3.Lerp(currentPosition, targetPosition, positionFactor);
            if (rotationFactor > 0f)
            {
                Quaternion currentRotation = anchor.rotation;
                anchor.rotation = IsFinite(currentRotation)
                    ? Quaternion.Slerp(currentRotation, playerRotation, rotationFactor)
                    : playerRotation;
            }
            return true;
        }

        static float ExponentialFactor(float sharpness, float deltaTime)
        {
            float safeSharpness = NonNegativeOrZero(sharpness);
            if (safeSharpness <= 0f) return 0f;
            float exponent = -safeSharpness * Mathf.Min(deltaTime, 1f);
            float value = 1f - Mathf.Exp(exponent);
            return IsFinite(value) ? Mathf.Clamp01(value) : 1f;
        }

        static float FiniteOrZero(float value) => IsFinite(value) ? value : 0f;

        static float NonNegativeOrZero(float value) =>
            IsFinite(value) && value >= 0f ? value : 0f;

        static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

        static bool IsFinite(Vector3 value) =>
            IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);

        static bool IsFinite(Quaternion value) =>
            IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z) && IsFinite(value.w);

        public bool StepForTests(float deltaTime) => Follow(deltaTime);

        public Vector3 TargetPositionForTests()
        {
            if (playerTransform == null) return CompanionAnchor.position;
            return playerTransform.position + playerTransform.right * FiniteOrZero(sideOffset) +
                playerTransform.up * FiniteOrZero(heightOffset) +
                playerTransform.forward * FiniteOrZero(forwardOffset);
        }

        public void ConfigureForTests(
            Transform player,
            Transform anchor,
            float side = DefaultSideOffset,
            float height = DefaultHeightOffset,
            float forward = DefaultForwardOffset,
            float positionSharpness = DefaultFollowSharpness,
            float turnSharpness = DefaultRotationSharpness,
            float snap = DefaultSnapDistance)
        {
            playerTransform = player;
            companionAnchor = anchor;
            sideOffset = side;
            heightOffset = height;
            forwardOffset = forward;
            followSharpness = positionSharpness;
            rotationSharpness = turnSharpness;
            snapDistance = snap;
        }
    }
}
