using System.Collections.Generic;
using UnityEngine;

namespace WordsInSilence.Practice.PetRobotPortfolioQuest
{
    public sealed class GreetingNpcTriggerRelay : MonoBehaviour
    {
        [SerializeField] InWorldGreetingNpcDialogueController dialogueController;

        readonly HashSet<int> activePlayerColliders = new HashSet<int>();

        // one-time only, for the open question of whether this trigger fires at runtime
        static bool loggedFirstEnter;
        static bool loggedFirstExit;

        void OnTriggerEnter(Collider other)
        {
            if (!IsPlayer(other)) return;
            int id = other.GetInstanceID();
            if (activePlayerColliders.Add(id) && activePlayerColliders.Count == 1)
            {
                if (!loggedFirstEnter)
                {
                    loggedFirstEnter = true;
                    Debug.Log("[GreetingNpcTrigger] first player enter, controller="
                              + (dialogueController != null));
                }
                if (dialogueController != null)
                    dialogueController.SetPlayerInsideGreetingTrigger(true);
            }
        }

        void OnTriggerExit(Collider other)
        {
            if (!IsPlayer(other)) return;
            int id = other.GetInstanceID();
            if (activePlayerColliders.Remove(id) && activePlayerColliders.Count == 0)
            {
                if (!loggedFirstExit)
                {
                    loggedFirstExit = true;
                    Debug.Log("[GreetingNpcTrigger] first player exit");
                }
                if (dialogueController != null)
                    dialogueController.SetPlayerInsideGreetingTrigger(false);
            }
        }

        void OnDisable()
        {
            activePlayerColliders.Clear();
            if (dialogueController != null)
                dialogueController.SetPlayerInsideGreetingTrigger(false);
        }

        static bool IsPlayer(Collider other)
        {
            if (other == null) return false;
            Transform t = other.transform;
            while (t != null)
            {
                if (t.CompareTag("Player")) return true;
                t = t.parent;
            }
            return false;
        }
    }
}
