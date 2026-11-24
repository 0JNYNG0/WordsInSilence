using UnityEngine;

namespace WordsInSilence.Practice.PetRobotWorldIntegration
{
    [RequireComponent(typeof(CharacterController))]
    public sealed class PetRobotWorldPlayerController : MonoBehaviour
    {
        [SerializeField] float moveSpeed = 4f;
        [SerializeField] Animator playerAnimator;
        CharacterController characterController;

        void Awake()
        {
            characterController = GetComponent<CharacterController>();
            if (playerAnimator == null)
                playerAnimator = GetComponentInChildren<Animator>(true);
            SetRunning(false);
        }

        void Update()
        {
            float horizontal = 0f;
            float vertical = 0f;
            if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow)) horizontal -= 1f;
            if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) horizontal += 1f;
            if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow)) vertical -= 1f;
            if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow)) vertical += 1f;
            Move(new Vector2(horizontal, vertical), Time.deltaTime);
        }

        public void Move(Vector2 input, float deltaTime)
        {
            if (characterController == null) characterController = GetComponent<CharacterController>();
            bool isRunning = characterController != null && input.sqrMagnitude > 0f;
            SetRunning(isRunning);
            if (!isRunning) return;
            Vector3 direction = new Vector3(input.x, 0f, input.y).normalized;
            characterController.Move(direction * moveSpeed * Mathf.Max(0f, deltaTime));
            transform.forward = direction;
        }

        void OnDisable()
        {
            SetRunning(false);
        }

        void SetRunning(bool value)
        {
            if (playerAnimator != null) playerAnimator.SetBool("IsRun", value);
        }
    }
}
