using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace WordsInSilence.Camera.WebCamSpike
{
    /// <summary>
    /// 01_WebCamSpike Scene의 UI 바인딩.
    /// Inspector에서 각 UI 요소를 연결한다.
    /// </summary>
    public sealed class WebCamSpikeUI : MonoBehaviour
    {
        [Header("Controls")]
        [SerializeField] TMP_Dropdown _dropdownDevice;
        [SerializeField] Toggle _toggleMirror;
        [SerializeField] Button _buttonStart;
        [SerializeField] Button _buttonStop;
        [SerializeField] Button _buttonRestart;

        [Header("Status")]
        [SerializeField] TMP_Text _textStatus;
        [SerializeField] TMP_Text _textDiagnostics;

        [Header("Controller Reference")]
        [SerializeField] WebCamSpikeController _controller;

        void Awake()
        {
            _buttonStart.onClick.AddListener(() => _controller.OnStartClicked());
            _buttonStop.onClick.AddListener(() => _controller.OnStopClicked());
            _buttonRestart.onClick.AddListener(() => _controller.OnRestartClicked());
            _toggleMirror.onValueChanged.AddListener(v => _controller.OnMirrorToggled(v));
        }

        public void SetDeviceOptions(List<string> deviceNames)
        {
            _dropdownDevice.ClearOptions();
            _dropdownDevice.AddOptions(deviceNames);
        }

        public string GetSelectedDeviceName()
        {
            if (_dropdownDevice.options.Count == 0) return null;
            string name = _dropdownDevice.options[_dropdownDevice.value].text;
            return name == "(카메라 없음)" ? null : name;
        }

        public void SetStatus(string message)
        {
            if (_textStatus != null)
                _textStatus.text = message;
        }

        public void SetDiagnostics(string message)
        {
            if (_textDiagnostics != null)
                _textDiagnostics.text = message;
        }

        public void SetButtons(bool startEnabled, bool stopEnabled, bool restartEnabled)
        {
            if (_buttonStart != null) _buttonStart.interactable = startEnabled;
            if (_buttonStop != null) _buttonStop.interactable = stopEnabled;
            if (_buttonRestart != null) _buttonRestart.interactable = restartEnabled;
        }
    }
}
