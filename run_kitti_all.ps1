param(
  [string]$Configs = '',
  [string]$MaxFrames = 'null',
  [switch]$NoGui
)

$ErrorActionPreference = 'Stop'

$sequences = @('00','01','02','03','04','05','06','07','08','09','10')

foreach ($seq in $sequences) {
  $args = @('--sequence', $seq)
  if (-not [string]::IsNullOrWhiteSpace($Configs)) {
    $args += @('--configs', $Configs)
  }
  if (-not [string]::IsNullOrWhiteSpace($MaxFrames)) {
    $args += @('--max-frames', $MaxFrames)
  }
  if ($NoGui.IsPresent) {
    $args += '--no-gui'
  }
  .\run_kitti.ps1 @args
}
