# Loads the latest Visual Studio's x64 developer environment (vcvars64.bat) into the
# environment of later steps, as running CMake from an x64 developer shell does.
#
# It also exports RACHIS_WINDOWS_SDK_DXC, the Windows SDK's dxc. The Vulkan SDK's dxc comes
# first on PATH in CI, so the jobs pass this one as RACHIS_DXC_EXECUTABLE to cook DXIL
# with the same compiler as a developer shell that has the Vulkan SDK installed.
$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
    throw 'No Visual Studio installation has the x64 C++ tools.'
}

$before = @{}
Get-ChildItem env: | ForEach-Object { $before[$_.Name] = $_.Value }
$lines = cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set"
if ($LASTEXITCODE -ne 0) {
    throw 'vcvars64.bat failed.'
}

$after = @{}
foreach ($line in $lines) {
    if ($line -match '^([^=]+)=(.*)$') {
        $after[$Matches[1]] = $Matches[2]
        if ($before[$Matches[1]] -ne $Matches[2]) {
            Add-Content -LiteralPath $env:GITHUB_ENV -Value "$($Matches[1])=$($Matches[2])"
        }
    }
}

$dxc = (Join-Path $after['WindowsSdkVerBinPath'] 'x64\dxc.exe') -replace '\\', '/'
if (-not (Test-Path -LiteralPath $dxc)) {
    throw "The Windows SDK has no dxc at $dxc."
}
Add-Content -LiteralPath $env:GITHUB_ENV -Value "RACHIS_WINDOWS_SDK_DXC=$dxc"
"Visual Studio: $vs"
"Windows SDK dxc: $dxc"
