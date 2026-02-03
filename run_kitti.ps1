param(
  [string]$Sequence = '00',
  [string]$Configs = '',
  [string]$MaxFrames = 'null',
  [switch]$NoGui
)

$ErrorActionPreference = 'Stop'

function Resolve-ConfigList {
  param([string]$ConfigsValue)
  if ([string]::IsNullOrWhiteSpace($ConfigsValue)) {
    return @(
      # 'config\kitti_orb_brutematch.yaml'
      # 'config\kitti_sift_flannmatch.yaml',
      # 'config\kitti_superpoint_flannmatch.yaml',
      # 'config\kitti_superpoint_supergluematch.yaml',
      # 'config\kitti_xfeat_xfeatmatch.yaml',
      'config\kitti_swiftfeat_swiftfeatmatch.yaml'
      # 'config\kitti_xfeat_lightgluematch.yaml'
    )
  }

  return $ConfigsValue.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
}

function New-TempYamlPath {
  param([string]$BaseName)
  $tmpDir = [System.IO.Path]::GetTempPath()
  return (Join-Path $tmpDir $BaseName)
}

$configList = Resolve-ConfigList -ConfigsValue $Configs

foreach ($cfg in $configList) {
  if (-not (Test-Path -LiteralPath $cfg)) {
    Write-Error "Config not found: $cfg"
  }

  $tmpName = [System.IO.Path]::GetFileName($cfg)
  $tmpPath = New-TempYamlPath -BaseName $tmpName
  $content = Get-Content -LiteralPath $cfg -Raw
  $updated = $content -replace '(?m)^(\s*sequence:\s*).+$', "`${1}'$Sequence'"
  if (-not [string]::IsNullOrWhiteSpace($MaxFrames)) {
    $updated = $updated -replace '(?m)^(\s*max_frames:\s*).+$', "`${1}$MaxFrames"
  }
  Set-Content -LiteralPath $tmpPath -Value $updated -NoNewline

  Write-Host "Running $cfg (sequence $Sequence)"
  $tmpPathForPython = $tmpPath -replace '\\', '/'
  if ($NoGui.IsPresent) {
    python main.py --config $tmpPathForPython --no-gui
  } else {
    python main.py --config $tmpPathForPython
  }

  $baseName = [System.IO.Path]::GetFileNameWithoutExtension($cfg)
  $outDir = Join-Path 'output' ("kitti_sequence_{0}" -f $Sequence)
  $txtPath = Join-Path $outDir ($baseName + '.txt')
  $pngPath = Join-Path $outDir ($baseName + '.png')

  if (-not (Test-Path -LiteralPath $txtPath)) {
    Write-Error "Missing output log: $txtPath"
  }
  if (-not (Test-Path -LiteralPath $pngPath)) {
    Write-Error "Missing output image: $pngPath"
  }
}
