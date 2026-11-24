$ErrorActionPreference = 'Stop'
$unityDataRoot = 'C:/Program Files/Unity/Hub/Editor/6000.3.21f1/Editor/Data'
$demoUnitRoot = 'C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1/ops/demo_handshape_learning_20261113'
$compilerArguments = [System.Collections.Generic.List[string]]::new()
$compilerArguments.Add('/nostdlib+')
$compilerArguments.Add('/target:library')
$compilerArguments.Add('/define:UNITY_EDITOR')
$compilerArguments.Add('/out:"' + $demoUnitRoot + '/LocalFeedbackBridgeCompileCheck.dll"')
foreach ($assembly in (Get-ChildItem -LiteralPath ($unityDataRoot + '/NetStandard/ref/2.1.0') -Filter '*.dll')) {
    $compilerArguments.Add('/reference:"' + $assembly.FullName + '"')
}
foreach ($assembly in (Get-ChildItem -LiteralPath ($unityDataRoot + '/Managed/UnityEngine') -Filter '*.dll')) {
    $compilerArguments.Add('/reference:"' + $assembly.FullName + '"')
}
$compilerArguments.Add('"C:/MyUnityProjects/WordsInSilence/Assets/_Project/Scripts/Practice/LocalFeedbackBridgeDemo.cs"')
$compilerArguments.Add('"C:/MyUnityProjects/WordsInSilence/Assets/WordsInSilenceValidation/Editor/LocalFeedbackBridgeBuild.cs"')
$compilerArguments | Set-Content -LiteralPath ($demoUnitRoot + '/bridge_compile.rsp') -Encoding utf8
& ($unityDataRoot + '/NetCoreRuntime/dotnet.exe') ($unityDataRoot + '/DotNetSdkRoslyn/csc.dll') ('@' + $demoUnitRoot + '/bridge_compile.rsp') *> ($demoUnitRoot + '/bridge_compile.log')
$compileExit = $LASTEXITCODE
Get-Content -LiteralPath ($demoUnitRoot + '/bridge_compile.log')
@{exit_code=$compileExit;scope='Standalone C# compile with installed Unity references; NOT Unity player build/runtime';editor_license_blocked=$true} | ConvertTo-Json | Set-Content -LiteralPath ($demoUnitRoot + '/BRIDGE_COMPILE.json') -Encoding utf8
exit $compileExit
