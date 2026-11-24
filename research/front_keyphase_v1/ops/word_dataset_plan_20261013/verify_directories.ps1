$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$unit = $PSScriptRoot
$reportPath = Join-Path $unit 'INDEPENDENT_DIRECTORY_CHECK.json'
if (Test-Path -LiteralPath $reportPath) { throw 'Reuse completed independent check' }
$audit = Get-Content -LiteralPath (Join-Path $unit 'ARCHIVE_AUDIT.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$root = 'W:\WordsInSilence\data\AIHub_103\videos'
$archives = @(Get-ChildItem -LiteralPath $root -File -Recurse -Filter '*.zip')
$counts = @{D=0; F=0; L=0; R=0; U=0}
$bytes = @{D=[long]0; F=[long]0; L=[long]0; R=[long]0; U=[long]0}
$front = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
$groups = @{}
$people = [System.Collections.Generic.HashSet[int]]::new()
$codes = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
$nonVideo = 0
$duplicateFront = 0
foreach ($file in $archives) {
    $split = if ($file.FullName -match '[\\/]1\.Training[\\/]') {'1.Training'} elseif ($file.FullName -match '[\\/]2\.Validation[\\/]') {'2.Validation'} else {throw 'Unknown split'}
    $z = [System.IO.Compression.ZipFile]::OpenRead($file.FullName)
    try {
        foreach ($entry in $z.Entries) {
            if (-not $entry.Name) { continue }
            if ($entry.Name -notmatch '^NIA_SL_(WORD\d+)_REAL(\d+)_([DFLRU])\.mp4$') { $nonVideo++; continue }
            $code = $Matches[1]; $person = [int]$Matches[2]; $view = $Matches[3]
            $counts[$view]++; $bytes[$view] += $entry.Length
            [void]$codes.Add($code); [void]$people.Add($person)
            $key = "$split|$code|$person"
            if (-not $groups.ContainsKey($key)) { $groups[$key] = [System.Collections.Generic.HashSet[string]]::new() }
            [void]$groups[$key].Add($view)
            if ($view -eq 'F' -and -not $front.Add("$split|$($entry.Name.Substring(0,$entry.Name.Length-4))")) { $duplicateFront++ }
        }
    } finally { $z.Dispose() }
}
$matched = 0; $cleanMatched = 0; $missing = 0; $roles = @{}
$manifest = Join-Path (Split-Path $unit -Parent) 'word_acquisition_inventory_20261011\FRONT_MANIFEST.jsonl'
foreach ($line in [System.IO.File]::ReadLines($manifest,[System.Text.Encoding]::UTF8)) {
    $row = $line | ConvertFrom-Json
    if ($front.Contains("$($row.split)|$($row.clip_id)")) {
        $matched++
        if ($row.metadata_clean) {
            $cleanMatched++
            if (-not $roles.ContainsKey($row.role)) { $roles[$row.role] = 0 }
            $roles[$row.role]++
        }
    } else { $missing++ }
}
$checks = @{
    archive_count = ($archives.Count -eq $audit.archive_count)
    archive_bytes = (($archives | Measure-Object Length -Sum).Sum -eq $audit.archive_bytes)
    front_count = ($front.Count -eq $audit.front_video_count)
    matched = ($matched -eq $audit.matched_annotation_count)
    clean_join = ($cleanMatched -eq $audit.metadata_video_join_eligible)
    missing = ($missing -eq $audit.missing_front_videos.Count)
    groups = ($groups.Count -eq $audit.multiview_group_count)
    all_groups_have_five = (@($groups.Values | Where-Object {$_.Count -ne 5}).Count -eq 0)
    no_extra_payload_types = ($nonVideo -eq 0)
    no_duplicate_front_keys = ($duplicateFront -eq 0)
}
foreach ($v in @('D','F','L','R','U')) {
    $checks["count_$v"] = ($counts[$v] -eq $audit.by_view.$v)
    $checks["bytes_$v"] = ($bytes[$v] -eq $audit.uncompressed_bytes_by_view.$v)
}
foreach ($role in $roles.Keys) { $checks["role_$role"] = ($roles[$role] -eq $audit.eligible_role_counts.$role) }
$result = [ordered]@{status='PASS'; verifier='Codex separate .NET ZIP API and PowerShell aggregation, not separate agent'; checks=$checks; by_view=$counts; bytes_by_view=$bytes; groups=$groups.Count; word_codes=$codes.Count; signer_ids=@($people | Sort-Object); matched=$matched; metadata_clean_join=$cleanMatched; missing=$missing; eligible_roles=$roles; full_crc_or_media_decode=$false}
if (@($checks.Values | Where-Object {-not $_}).Count -gt 0) { $result.status = 'FAIL' }
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding UTF8
$result | ConvertTo-Json -Depth 6
if ($result.status -ne 'PASS') { throw 'Independent verification failure' }
