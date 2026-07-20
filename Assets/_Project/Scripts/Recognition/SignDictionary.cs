using System.Collections.Generic;
using UnityEngine;

namespace WordsInSilence.Recognition
{
    [CreateAssetMenu(menuName = "SilentSign/SignDictionary", fileName = "DefaultSignDictionary")]
    public class SignDictionary : ScriptableObject
    {
        public List<SignEntry> signs = new List<SignEntry>();
    }
}
