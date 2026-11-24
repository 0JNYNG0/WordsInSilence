using System;
using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace WordsInSilence.Practice
{
    // Isolated integration scene. The Python service is the sole camera owner.
    // Classifier rank is not an educational correctness verdict.
    public sealed class LocalFeedbackBridgeDemo : MonoBehaviour
    {
        [Serializable] public sealed class Word { public string id; public string label; }
        [Serializable] public sealed class Meta { public string token; public Word[] words; public string scope; }
        [Serializable] public sealed class ServiceState { public string state; public string error; }
        [Serializable] public sealed class Rank { public string word; public string label; public float probability; }
        [Serializable] public sealed class Card { public string message; public string display_status; public bool educational_wrong; }
        [Serializable] public sealed class Result { public string word; public string label; public Rank[] tcn_top3; public Card[] cards; public bool correction_validated; public bool overall_pass; }
        [Serializable] public sealed class SmokeReport { public bool passed; public bool camera_opened; public string source; public string word; public int candidates; public int cards; public string error; }

        string url = "http://127.0.0.1:8814";
        string word = "WORD1133";
        string message = "연결을 확인하고 있어요.";
        Meta meta; ServiceState status; Result result;
        bool busy; bool selfTest; bool quitting;
        string reportPath;
        Texture2D preview;
        GUIStyle title, body, rankStyle;

        IEnumerator Start()
        {
            Application.runInBackground = true;
            foreach (var arg in Environment.GetCommandLineArgs())
            {
                if (arg == "--feedback-self-test") selfTest = true;
                if (arg.StartsWith("--feedback-url=")) url = arg.Substring(15).TrimEnd('/');
                if (arg.StartsWith("--feedback-output=")) reportPath = arg.Substring(18);
            }
            Uri address;
            if (!Uri.TryCreate(url, UriKind.Absolute, out address) || address.Scheme != "http" || address.Host != "127.0.0.1")
            {
                message = "이 시험은 같은 PC의 로컬 분석기만 연결합니다.";
                if (selfTest) FinishTest(false, message);
                yield break;
            }
            yield return Get("/api/meta", text => meta = JsonUtility.FromJson<Meta>(text));
            if (meta == null || string.IsNullOrEmpty(meta.token))
            {
                if (selfTest) FinishTest(false, message);
                yield break;
            }
            if (selfTest)
            {
                // Read-only fixture test: never POST /open or access a camera.
                yield return Get("/api/status", text => status = JsonUtility.FromJson<ServiceState>(text));
                yield return Get("/api/result", text => result = JsonUtility.FromJson<Result>(text));
                bool ok = status != null && status.state == "done" && result != null
                    && result.word == word && result.tcn_top3 != null && result.tcn_top3.Length == 3
                    && result.cards != null && result.cards.Length > 0 && !result.correction_validated;
                FinishTest(ok, ok ? null : message);
                yield break;
            }
            message = "잘하다 · 카메라 열기를 누르면 촬영을 준비합니다.";
            while (!quitting)
            {
                yield return Get("/api/status", text => status = JsonUtility.FromJson<ServiceState>(text));
                if (status != null && status.state == "done" && result == null)
                    yield return Get("/api/result", text => result = JsonUtility.FromJson<Result>(text));
                if (status != null && (status.state == "preview" || status.state == "countdown" || status.state == "recording"))
                {
                    using (var request = UnityWebRequestTexture.GetTexture(url + "/preview.jpg"))
                    {
                        request.timeout = 3;
                        yield return request.SendWebRequest();
                        if (request.result == UnityWebRequest.Result.Success)
                        {
                            if (preview != null) Destroy(preview);
                            preview = DownloadHandlerTexture.GetContent(request);
                        }
                    }
                }
                yield return new WaitForSecondsRealtime(.25f);
            }
        }

        IEnumerator Get(string route, Action<string> accept)
        {
            using (var request = UnityWebRequest.Get(url + route))
            {
                request.timeout = 5;
                yield return request.SendWebRequest();
                if (request.result == UnityWebRequest.Result.Success) accept(request.downloadHandler.text);
                else message = "분석기 연결을 확인해 주세요. " + request.error;
            }
        }

        IEnumerator Command(string route)
        {
            if (busy || meta == null) yield break;
            busy = true;
            using (var request = new UnityWebRequest(url + route, "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes("{\"word\":\"" + word + "\"}"));
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                request.SetRequestHeader("X-Session-Token", meta.token);
                request.timeout = 8;
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success) message = request.downloadHandler.text;
                else
                {
                    if (route == "/api/start") result = null;
                    message = "요청을 전달했어요.";
                }
            }
            busy = false;
        }

        void FinishTest(bool passed, string error)
        {
            var report = new SmokeReport { passed = passed, camera_opened = false,
                source = "saved public result fixture; not new inference or webcam validation",
                word = result == null ? null : result.word,
                candidates = result == null || result.tcn_top3 == null ? 0 : result.tcn_top3.Length,
                cards = result == null || result.cards == null ? 0 : result.cards.Length, error = error };
            if (!string.IsNullOrEmpty(reportPath)) File.WriteAllText(reportPath, JsonUtility.ToJson(report, true), Encoding.UTF8);
            Debug.Log("LOCAL_FEEDBACK_BRIDGE_TEST " + JsonUtility.ToJson(report));
            Application.Quit(passed ? 0 : 2);
        }

        void OnGUI()
        {
            if (selfTest) return;
            if (title == null)
            {
                title = new GUIStyle(GUI.skin.label) { fontSize = 26, fontStyle = FontStyle.Bold };
                body = new GUIStyle(GUI.skin.label) { fontSize = 18, wordWrap = true };
                rankStyle = new GUIStyle(body) { fontStyle = FontStyle.Bold };
            }
            GUILayout.BeginArea(new Rect(24, 20, Screen.width - 48, Screen.height - 40));
            GUILayout.Label("잘하다 · Unity 연결 시험", title);
            GUILayout.Label("기존 AI 결과 연결 확인용 · 최종 게임 화면과 교정 정확도 검증은 별도", body);
            GUILayout.Label("분석기: " + (status == null ? "연결 대기" : status.state), body);
            GUILayout.Label(message, body);
            if (status != null && !string.IsNullOrEmpty(status.error)) GUILayout.Label(status.error, body);
            GUILayout.BeginHorizontal();
            GUI.enabled = !busy && meta != null;
            if (GUILayout.Button("카메라 열기", GUILayout.Height(40))) StartCoroutine(Command("/api/open"));
            if (GUILayout.Button("3초 뒤 촬영", GUILayout.Height(40))) StartCoroutine(Command("/api/start"));
            if (GUILayout.Button("수동 종료", GUILayout.Height(40))) StartCoroutine(Command("/api/finish"));
            if (GUILayout.Button("카메라 닫기", GUILayout.Height(40))) StartCoroutine(Command("/api/close"));
            GUI.enabled = true;
            if (GUILayout.Button("웹 비교 화면", GUILayout.Height(40))) Application.OpenURL(url);
            GUILayout.EndHorizontal();
            if (preview != null && result == null) GUILayout.Label(preview, GUILayout.Width(480), GUILayout.Height(270));
            if (result != null && result.tcn_top3 != null)
            {
                GUILayout.Space(15);GUILayout.BeginHorizontal();
                for (int i = 0; i < result.tcn_top3.Length; i++)
                {
                    var r = result.tcn_top3[i];GUI.contentColor = r.word == word ? new Color(.45f,.8f,1f) : Color.white;
                    GUILayout.Label((i+1) + "위 " + r.label, rankStyle);
                }
                GUI.contentColor = Color.white;GUILayout.EndHorizontal();
                GUILayout.Label("인식 후보와 동작 교정 판단은 서로 달라요.", body);
                if (result.cards != null) foreach (var c in result.cards) GUILayout.Label("• " + c.message, body);
                GUILayout.Label("확인하지 못한 부분은 틀렸다는 뜻이 아닙니다.", body);
            }
            GUILayout.EndArea();
        }
        void OnApplicationQuit() { quitting = true; /* Do not close another client's camera. */ }
        void OnDestroy() { if (preview != null) Destroy(preview); }
    }
}
