# Installs the build tools that the Windows CI jobs pin: CMake and Ninja from their GitHub
# releases (the versions in flake.lock's nixpkgs), and the LunarG Vulkan SDK, which
# supplies the SPIR-V dxc, spirv-val, glslangValidator, and slangc with the dxcompiler
# that Slang's DXIL target loads. Every download is checked against a pinned SHA-256.
# Later steps find the tools through GITHUB_PATH and VULKAN_SDK.
param(
    [string] $ToolsDir = (Join-Path $env:RUNNER_TEMP 'rachis-tools')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$cmakeVersion = '4.4.4'
$cmakeSha256 = 'bace36e94b31c68ab6fa295f26dfa11219e0701cf7c94b0284a7d1cb13dac536'
$ninjaVersion = '1.13.2'
$ninjaSha256 = '07fc8261b42b20e71d1720b39068c2e14ffcee6396b76fb7a795fb460b78dc65'
$vulkanVersion = '1.4.341.1'
$vulkanSha256 = 'bcf2d75aa9556889ab974858666e20b3655b6055a0db704ccb47279ff33b5bfe'

function Get-VerifiedFile([string] $Uri, [string] $Path, [string] $Sha256) {
    Invoke-WebRequest -Uri $Uri -OutFile $Path -MaximumRetryCount 5 -RetryIntervalSec 10
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
    if ($actual -ne $Sha256) {
        throw "$Uri has SHA-256 $actual, expected $Sha256."
    }
}

if ($ToolsDir -match '\s') {
    throw "The tools directory must not contain spaces: $ToolsDir"
}
$downloads = Join-Path $ToolsDir 'downloads'
New-Item -ItemType Directory -Force -Path $downloads | Out-Null

$cmakeName = "cmake-$cmakeVersion-windows-x86_64"
$cmakeZip = Join-Path $downloads "$cmakeName.zip"
Get-VerifiedFile "https://github.com/Kitware/CMake/releases/download/v$cmakeVersion/$cmakeName.zip" $cmakeZip $cmakeSha256
Expand-Archive -LiteralPath $cmakeZip -DestinationPath $ToolsDir -Force
$cmakeBin = Join-Path $ToolsDir "$cmakeName\bin"

$ninjaZip = Join-Path $downloads "ninja-$ninjaVersion-win.zip"
Get-VerifiedFile "https://github.com/ninja-build/ninja/releases/download/v$ninjaVersion/ninja-win.zip" $ninjaZip $ninjaSha256
$ninjaDir = Join-Path $ToolsDir "ninja-$ninjaVersion"
Expand-Archive -LiteralPath $ninjaZip -DestinationPath $ninjaDir -Force

$vulkanInstaller = Join-Path $downloads "vulkansdk-windows-X64-$vulkanVersion.exe"
Get-VerifiedFile "https://sdk.lunarg.com/sdk/download/$vulkanVersion/windows/vulkansdk-windows-X64-$vulkanVersion.exe" $vulkanInstaller $vulkanSha256
$vulkanRoot = Join-Path $ToolsDir "VulkanSDK\$vulkanVersion"
$installer = Start-Process -FilePath $vulkanInstaller -Verb RunAs -Wait -PassThru -ArgumentList @(
    '--root', $vulkanRoot, '--accept-licenses', '--default-answer', '--confirm-command', 'install')
if ($installer.ExitCode -ne 0) {
    throw "The Vulkan SDK installer exited with code $($installer.ExitCode)."
}
Remove-Item -LiteralPath $vulkanInstaller

$vulkanBin = Join-Path $vulkanRoot 'Bin'
$required = @(
    (Join-Path $cmakeBin 'cmake.exe'),
    (Join-Path $ninjaDir 'ninja.exe'),
    (Join-Path $vulkanBin 'dxc.exe'),
    (Join-Path $vulkanBin 'spirv-val.exe'),
    (Join-Path $vulkanBin 'glslangValidator.exe'),
    (Join-Path $vulkanBin 'slangc.exe'),
    (Join-Path $vulkanBin 'dxcompiler.dll'))
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing after installation: $path"
    }
}

Add-Content -LiteralPath $env:GITHUB_PATH -Value $vulkanBin, $ninjaDir, $cmakeBin
Add-Content -LiteralPath $env:GITHUB_ENV -Value "VULKAN_SDK=$vulkanRoot"
