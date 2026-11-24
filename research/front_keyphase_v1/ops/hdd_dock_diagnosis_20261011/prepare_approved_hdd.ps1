param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$recordDir = $PSScriptRoot
$stage = 'preflight'
$mutationStarted = $false
$started = Get-Date

function Save-Record($name, $value) {
    $value | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $recordDir $name) -Encoding UTF8
}
function Assert-ApprovedTarget {
    $usbDisks = @(Get-Disk | Where-Object { $_.BusType -eq 'USB' })
    if ($usbDisks.Count -ne 1) { throw 'Expected exactly one USB disk.' }
    $disk = $usbDisks[0]
    if ($disk.Number -ne 1 -or $disk.Size -ne 3000592982016 -or
        $disk.FriendlyName -ne 'ASMT ASM1156-PM' -or
        $disk.UniqueId -ne '5000000000000001' -or
        $disk.IsBoot -or $disk.IsSystem -or $disk.IsOffline -or $disk.IsReadOnly -or
        $disk.PartitionStyle -ne 'GPT') { throw 'Approved disk identity or state changed.' }
    $parts = @(Get-Partition -DiskNumber $disk.Number | Sort-Object PartitionNumber)
    $sizes = @(2550005760L, 2147483648L, 2995656278016L)
    if ($parts.Count -ne 3) { throw 'Expected the three original Linux RAID partitions.' }
    for ($index = 0; $index -lt 3; $index++) {
        if ($parts[$index].Size -ne $sizes[$index] -or
            ([string]$parts[$index].GptType).Trim('{}') -ne 'a19d880f-05fc-4d3b-a006-743f0f84911e' -or
            $parts[$index].IsBoot -or $parts[$index].IsSystem -or
            [int][char]$parts[$index].DriveLetter -ne 0) { throw 'Original partition fingerprint changed.' }
    }
    if ((Get-Disk -Number 0).BusType -ne 'NVMe') { throw 'Internal disk layout changed.' }
    if (@(Get-Volume | Where-Object DriveLetter -eq 'W').Count -ne 0 -or
        (Test-Path -LiteralPath 'W:\')) { throw 'Drive letter W is already in use.' }
    return $disk
}
function Internal-Layout {
    @(Get-Partition -DiskNumber 0 | Sort-Object PartitionNumber |
        Select-Object PartitionNumber,Offset,Size,GptType,DriveLetter) | ConvertTo-Json -Compress
}
function Save-Stage($name) {
    Save-Record 'FORMAT_PROGRESS.json' @{stage=$name; at=(Get-Date -Format o); pid=$PID; mutation_started=$mutationStarted}
}

try {
    $target = Assert-ApprovedTarget
    $beforeInternal = Internal-Layout
    if ($ValidateOnly) {
        Save-Record 'FORMAT_PREFLIGHT.json' @{passed=$true; at=(Get-Date -Format o); disk_number=$target.Number; size_bytes=$target.Size; scope='Remove original three partitions, GPT, NTFS WIS_DATA (W:), 16MiB generated-file verification'; mutation_started=$false}
        Write-Output 'Approved-target preflight passed; validation only, no disk writes.'
        exit 0
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator token required.' }
    $oncePath = Join-Path $recordDir 'FORMAT_STARTED.json'
    $onceHandle = [IO.File]::Open($oncePath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $onceHandle.Dispose()
    Save-Record 'FORMAT_STARTED.json' @{at=($started.ToString('o')); pid=$PID; user_authorization='2026-10-11 user: proceed, in response to explicit all-data deletion/format proposal'; disk_number=1; size_bytes=3000592982016; before_internal_layout=$beforeInternal}
    Save-Stage 'preflight_passed'
    $stage = 'administrator_health_query'
    $healthRows = @()
    try {
        $physicalTarget = @(Get-PhysicalDisk | Where-Object { $_.BusType -eq 'USB' -and $_.Size -eq 3000592982016 })
        if ($physicalTarget.Count -ne 1) { throw 'Physical target cannot be identified uniquely.' }
        $healthRows = @($physicalTarget[0] | Get-StorageReliabilityCounter |
            Select-Object DeviceId,Temperature,PowerOnHours,ReadErrorsTotal,ReadErrorsUncorrected,WriteErrorsTotal,WriteErrorsUncorrected)
        Save-Record 'ADMIN_HEALTH_QUERY.json' @{query_succeeded=$true; rows=$healthRows; at=(Get-Date -Format o)}
    } catch {
        Save-Record 'ADMIN_HEALTH_QUERY.json' @{query_succeeded=$false; error=$_.Exception.Message; at=(Get-Date -Format o)}
    }
    foreach ($healthRow in $healthRows) {
        if ($healthRow.ReadErrorsUncorrected -gt 0 -or $healthRow.WriteErrorsUncorrected -gt 0) {
            throw 'Uncorrected errors reported; stopping before deletion for review.'
        }
    }
    $target = Assert-ApprovedTarget
    $mutationStarted = $true
    $stage = 'clear_original_partitions'
    Save-Stage $stage
    Clear-Disk -InputObject $target -RemoveData -RemoveOEM -Confirm:$false
    $stage = 'initialize_gpt'
    Save-Stage $stage
    $cleared = Get-Disk -Number 1
    if ($cleared.Size -ne 3000592982016 -or $cleared.BusType -ne 'USB' -or $cleared.IsBoot -or $cleared.IsSystem) { throw 'Target changed after clear.' }
    if ($cleared.PartitionStyle -eq 'RAW') { Initialize-Disk -Number 1 -PartitionStyle GPT | Out-Null }
    elseif ($cleared.PartitionStyle -ne 'GPT') { throw 'Unexpected partition style after clear.' }
    $stage = 'create_ntfs_volume'
    Save-Stage $stage
    $partition = New-Partition -DiskNumber 1 -UseMaximumSize -DriveLetter W
    Format-Volume -Partition $partition -FileSystem NTFS -NewFileSystemLabel 'WIS_DATA' -AllocationUnitSize 4096 -Force -Confirm:$false | Out-Null
    $volume = Get-Volume -DriveLetter W
    $mounted = Get-Partition -DriveLetter W
    if ($mounted.DiskNumber -ne 1 -or $volume.FileSystemType -ne 'NTFS' -or
        $volume.FileSystemLabel -ne 'WIS_DATA' -or $volume.Size -lt 2990000000000) { throw 'Formatted volume verification failed.' }
    $stage = 'generated_file_write_read'
    Save-Stage $stage
    $probeRoot = [IO.Path]::GetFullPath('W:\')
    $probeDir = [IO.Path]::GetFullPath((Join-Path $probeRoot ('.wis_storage_check_' + [Guid]::NewGuid().ToString('N'))))
    if (-not $probeDir.StartsWith($probeRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Probe directory outside approved volume.' }
    New-Item -ItemType Directory -Path $probeDir | Out-Null
    $probeFile = [IO.Path]::GetFullPath((Join-Path $probeDir 'generated_16MiB.bin'))
    if ([IO.Path]::GetDirectoryName($probeFile) -ne $probeDir) { throw 'Probe file path mismatch.' }
    $buffer = New-Object byte[] (16 * 1024 * 1024)
    $rng = New-Object System.Random -ArgumentList 20261011
    $rng.NextBytes($buffer)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $expected = [BitConverter]::ToString($hasher.ComputeHash($buffer)).Replace('-','') }
    finally { $hasher.Dispose() }
    $stream = [IO.File]::Open($probeFile,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try { $stream.Write($buffer,0,$buffer.Length); $stream.Flush($true) }
    finally { $stream.Dispose() }
    $actual = (Get-FileHash -LiteralPath $probeFile -Algorithm SHA256).Hash
    if ($actual -ne $expected) { throw 'Generated-file SHA256 mismatch.' }
    Remove-Item -LiteralPath $probeFile -Force
    Remove-Item -LiteralPath $probeDir -Force
    if ((Internal-Layout) -ne $beforeInternal) { throw 'Internal disk partition metadata unexpectedly changed.' }
    $stage = 'complete'
    $volume = Get-Volume -DriveLetter W
    Save-Record 'FORMAT_RESULT.json' @{success=$true; at=(Get-Date -Format o); started=($started.ToString('o')); disk_number=1; size_bytes=3000592982016; partition_style='GPT'; drive='W:\'; label=$volume.FileSystemLabel; file_system=$volume.FileSystemType.ToString(); volume_size_bytes=$volume.Size; remaining_bytes=$volume.SizeRemaining; generated_file_bytes=$buffer.Length; expected_sha256=$expected; read_sha256=$actual; probe_removed=$true; internal_partition_metadata_unchanged=$true; health_limit='Quick format and small file check are not a SMART or surface health test.'}
    Save-Stage 'complete'
    exit 0
} catch {
    Save-Record 'FORMAT_FAILURE.json' @{at=(Get-Date -Format o); stage=$stage; mutation_started=$mutationStarted; error=$_.Exception.Message; detail=($_ | Out-String)}
    exit 1
}
