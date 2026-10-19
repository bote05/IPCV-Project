param([string]$Editor)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Editor) -or -not (Test-Path -LiteralPath $Editor)) { throw 'Pass -Editor with your Unity 6000.4.8f1 executable path' }
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$testRoot = Join-Path $env:TEMP ('ipcv-bridge-tests-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Join-Path $testRoot 'Assets'), (Join-Path $testRoot 'Packages'), (Join-Path $testRoot 'ProjectSettings') | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'IPCV Climbing Game/Assets/Scripts/Bridge') -Destination (Join-Path $testRoot 'Assets/Bridge') -Recurse
Copy-Item -LiteralPath (Join-Path $root 'IPCV Climbing Game/ProjectSettings/ProjectVersion.txt') -Destination (Join-Path $testRoot 'ProjectSettings/ProjectVersion.txt')
[IO.File]::WriteAllText((Join-Path $testRoot 'Packages/manifest.json'), '{"dependencies":{"com.unity.test-framework":"1.6.0","com.unity.modules.imageconversion":"1.0.0","com.unity.modules.imgui":"1.0.0","com.unity.modules.jsonserialize":"1.0.0"}}')
$env:IPCV_BRIDGE_ROOT = $root
$env:IPCV_BRIDGE_PYTHON = (& python -c 'import sys; print(sys.executable)').Trim()
$log = Join-Path $testRoot 'unity-test.log'
$results = Join-Path $testRoot 'test-results.xml'
Write-Output ('Isolated test project: ' + $testRoot)
$testProcess = Start-Process -FilePath $Editor -ArgumentList '-batchmode','-nographics','-projectPath',('"' + $testRoot + '"'),'-runTests','-testPlatform','PlayMode','-testResults',('"' + $results + '"'),'-logFile',('"' + $log + '"') -WindowStyle Hidden -PassThru
$testProcess.WaitForExit()
if (-not (Test-Path -LiteralPath $results)) {
    Get-Content -LiteralPath $log -Tail 35
    throw 'Unity did not produce test results. See the test log above.'
}
[xml]$xml = Get-Content -LiteralPath $results
Write-Output ('Unity tests: ' + $xml.'test-run'.result + ' | total ' + $xml.'test-run'.total + ' | passed ' + $xml.'test-run'.passed + ' | failed ' + $xml.'test-run'.failed)
Write-Output ('Results: ' + $results)
Write-Output ('Log: ' + $log)
Get-Content -LiteralPath $log | Select-String -Pattern '^IPCV_EVALUATION_CSV='
if ($xml.'test-run'.failed -ne '0' -or $xml.'test-run'.passed -ne $xml.'test-run'.total) { throw 'Unity tests did not all pass' }
