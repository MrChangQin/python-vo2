param(
  [string]$Configs = '',
  [string]$MaxFrames = 'null',
  [switch]$NoGui
)

$ErrorActionPreference = 'Stop'

$sequences = @('00','01','02','03','04','05','06','07','08','09','10')

foreach ($seq in $sequences) {
  $splat = @{
    Sequence = $seq
  }
  if (-not [string]::IsNullOrWhiteSpace($Configs)) {
    $splat.Configs = $Configs
  }
  if (-not [string]::IsNullOrWhiteSpace($MaxFrames)) {
    $splat.MaxFrames = $MaxFrames
  }
  if ($NoGui.IsPresent) {
    $splat.NoGui = $true
  }
  .\run_kitti.ps1 @splat
}