$ErrorActionPreference = 'Stop'
$reportRoot = $PSScriptRoot
$reportHealth = $null
try { $reportHealth = Invoke-RestMethod -Uri 'http://127.0.0.1:8791/health' -TimeoutSec 2 } catch { }
if ($reportHealth) {
    if ($reportHealth.unit -ne 'word100_core_stage_evidence_20261031') { throw 'Port 8791 belongs to a different service. Do not replace it.' }
    Write-Output 'Already available: http://127.0.0.1:8791/'
    exit 0
}
$reportPython = Join-Path (Split-Path $reportRoot -Parent) 'word100_signrep_feasibility_20261029/runtime/Scripts/python.exe'
$reportScript = Join-Path $reportRoot 'report_server.py'
if (-not (Test-Path -LiteralPath $reportPython)) { throw 'Existing project Python is missing. No installation or download was attempted.' }
$reportStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$reportOut = Join-Path $reportRoot ('server_' + $reportStamp + '.out.log')
$reportErr = Join-Path $reportRoot ('server_' + $reportStamp + '.err.log')
Start-Process -FilePath $reportPython -ArgumentList @('"' + $reportScript + '"') -WorkingDirectory $reportRoot -WindowStyle Hidden -RedirectStandardOutput $reportOut -RedirectStandardError $reportErr
Write-Output 'Starting read-only report: http://127.0.0.1:8791/ . This does not start training or camera capture.'
