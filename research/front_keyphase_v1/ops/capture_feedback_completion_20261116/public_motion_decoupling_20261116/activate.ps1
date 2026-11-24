$ErrorActionPreference = 'Stop'
$wisRoot = 'C:\MyUnityProjects\WordsInSilence'
$wisRun = Join-Path $wisRoot 'research\front_keyphase_v1\ops\capture_feedback_completion_20261116\public_motion_decoupling_20261116'
foreach ($wisCheck in @('CONSUMER_CHECK','CARD_CHECK','CAPTURE_CHECK')) {
    $wisEvidence = Get-Content -LiteralPath (Join-Path $wisRun "web\$wisCheck.json") -Raw | ConvertFrom-Json
    if (!$wisEvidence.passed) { throw "Runtime check failed: $wisCheck" }
}
$wisMeta = Invoke-RestMethod -Uri 'http://127.0.0.1:8821/api/meta'
$wisStatus = Invoke-RestMethod -Uri 'http://127.0.0.1:8821/api/status' -Headers @{'X-Session-Token'=$wisMeta.token}
if ($wisStatus.camera_thread_active -or $wisStatus.state -notin @('ready','done')) { throw 'Camera or analysis is active; keep existing server.' }
$wisOwner = (Get-NetTCPConnection -LocalPort 8821 -State Listen).OwningProcess
$wisProcesses = @(Get-CimInstance Win32_Process)
$wisOld = $wisProcesses | Where-Object ProcessId -eq $wisOwner
if (!$wisOld.CommandLine.Contains('WordsInSilence') -or !$wisOld.CommandLine.Contains('public_shape_endpoint_learning_20261116\serve.py')) { throw 'Listener is not the expected prior project server.' }
$wisStop = [System.Collections.Generic.List[int]]::new();$wisStop.Add([int]$wisOwner)
do {
    $wisAdded = $false
    foreach ($wisProcess in $wisProcesses) {
        if ($wisStop.Contains([int]$wisProcess.ParentProcessId) -and !$wisStop.Contains([int]$wisProcess.ProcessId)) {
            $wisProjectChild = $wisProcess.CommandLine -and $wisProcess.CommandLine.Contains('WordsInSilence')
            $wisOwnedConsole = $wisProcess.Name -eq 'conhost.exe' -and $wisProcess.CommandLine -like '*Windows\system32\conhost.exe*'
            if (!$wisProjectChild -and !$wisOwnedConsole) { throw 'Unexpected child process; retain server.' }
            $wisStop.Add([int]$wisProcess.ProcessId);$wisAdded=$true
        }
    }
} while ($wisAdded)
for ($wisIndex=$wisStop.Count-1; $wisIndex -ge 0; $wisIndex--) { Stop-Process -Id $wisStop[$wisIndex] -ErrorAction SilentlyContinue }
$wisPython = Join-Path $wisRoot 'research\front_keyphase_v1\ops\word100_signrep_feasibility_20261029\runtime\Scripts\python.exe'
$wisNew = Start-Process -FilePath $wisPython -ArgumentList @('-X','utf8','-B',(Join-Path $wisRun 'serve.py')) -WorkingDirectory $wisRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $wisRun 'server_stdout.log') -RedirectStandardError (Join-Path $wisRun 'server_stderr.log') -PassThru
@{launcher_pid=$wisNew.Id;old_server_pid=$wisOwner;stopped_project_pids=@($wisStop);entrypoint=(Join-Path $wisRun 'serve.py');camera_started=$false;at=(Get-Date -Format o)} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $wisRun 'web\ACTIVATION.json') -Encoding utf8
Write-Output "Started latest local server launcher $($wisNew.Id), camera remains off."
