# scripts/build.ps1 -- configure and build Stage 0 against the pinned toolchain.
#
# Two Visual Studio toolsets are installed on the measurement machine:
#   VS 2022 Build Tools   MSVC 14.44 / cl 19.44   <- the one CUDA 13.1 accepts
#   VS 2026               MSVC 14.50 / 14.51      <- rejected by nvcc
#
# Auto-detection picks the newest, which is the wrong one, so the toolset is
# pinned two ways: this script enters the VS 2022 environment before invoking
# CMake, and CMakeLists.txt fails the configure step outright if it ends up
# with MSVC_VERSION >= 1950. -allow-unsupported-compiler is never used.

[CmdletBinding()]
param(
    [string]$BuildDir  = "build",
    [string]$Config    = "Release",
    [switch]$Clean,
    [switch]$Test
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $root = & $vswhere -version "[17.0,18.0)" -products * -latest -property installationPath
        if ($root) { $vcvars = Join-Path $root "VC\Auxiliary\Build\vcvars64.bat" }
    }
}
if (-not (Test-Path $vcvars)) {
    throw "VS 2022 vcvars64.bat not found. CUDA 13.1 requires the 2019-2022 toolset."
}

$cl = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe"
if (-not (Test-Path $cl)) {
    $toolsRoot = Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $vcvars))) "Tools\MSVC"
    $newest = Get-ChildItem $toolsRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
    $cl = Join-Path $newest.FullName "bin\HostX64\x64\cl.exe"
}
$clFwd = $cl -replace '\\', '/'

$build = Join-Path $repo $BuildDir
if ($Clean -and (Test-Path $build)) { Remove-Item $build -Recurse -Force }

$cfg = @(
    "-S `"$repo`"", "-B `"$build`"", "-G Ninja",
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCMAKE_C_COMPILER=`"$clFwd`"",
    "-DCMAKE_CXX_COMPILER=`"$clFwd`"",
    "-DCMAKE_CUDA_HOST_COMPILER=`"$clFwd`""
) -join " "

$cmd = "call `"$vcvars`" >nul 2>&1 && cl 2>&1 | findstr /C:`"Version`" && cmake $cfg && cmake --build `"$build`" --config $Config"
if ($Test) { $cmd += " && ctest --test-dir `"$build`" --output-on-failure" }

Write-Host "pinned host compiler: $cl"
cmd /c $cmd
exit $LASTEXITCODE
