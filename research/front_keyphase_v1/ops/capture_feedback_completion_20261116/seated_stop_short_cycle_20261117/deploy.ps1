$ErrorActionPreference = 'Stop'
$wisRoot = 'C:\MyUnityProjects\WordsInSilence'
$wisDir = Join-Path $wisRoot 'research\front_keyphase_v1\ops\capture_feedback_completion_20261116\seated_stop_short_cycle_20261117'
$wisExpected = Join-Path $wisRoot 'research\front_keyphase_v1\ops\capture_feedback_completion_20261116\selective_rtmw_anchor_repair_20261117\serve.py'
$wisListener = Get-NetTCPConnection -LocalPort 8821 -State Listen -ErrorAction SilentlyContinue
if ($wisListener) {
    $wisProc = Get-CimInstance Win32_Process -Filter "ProcessId=$($wisListener[0].OwningProcess)"
    if (-not $wisProc.CommandLine.Contains($wisExpected)) { throw 'Unexpected service; leave it untouched.' }
    $wisMeta = Invoke-RestMethod 'http://127.0.0.1:8821/api/meta'
    $wisStatus = Invoke-RestMethod 'http://127.0.0.1:8821/api/status' -Headers @{'X-Session-Token'=$wisMeta.token}
    if ($wisStatus.camera_thread_active -or $wisStatus.state -notin @('ready','done','error')) { throw 'Capture or analysis active; leave service running.' }
    & taskkill.exe /PID $wisProc.ProcessId /T /F
    if ($LASTEXITCODE -ne 0) { throw 'Old service did not stop.' }
    Start-Sleep -Milliseconds 500
}
$wisPython = Join-Path $wisRoot 'research\front_keyphase_v1\ops\word100_signrep_feasibility_20261029\runtime\Scripts\python.exe'
$wisNew = Start-Process -FilePath $wisPython -ArgumentList @('-X','utf8','-B',(Join-Path $wisDir 'serve.py')) -WorkingDirectory $wisRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $wisDir 'server_stdout.log') -RedirectStandardError (Join-Path $wisDir 'server_stderr.log') -PassThru
Write-Output "Started verified local server launcher PID $($wisNew.Id). Camera remains off."
