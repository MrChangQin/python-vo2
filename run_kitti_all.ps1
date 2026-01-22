param(
  [string]$Configs = '',
  [string]$MaxFrames = 'null'
)

$ErrorActionPreference = 'Stop'

$sequences = @('00','01','02','03','04','05','06','07','08','09','10')

foreach ($seq in $sequences) {
  if ([string]::IsNullOrWhiteSpace($Configs) -and [string]::IsNullOrWhiteSpace($MaxFrames)) {
    .\run_kitti.ps1 --sequence $seq
  } elseif ([string]::IsNullOrWhiteSpace($Configs)) {
    .\run_kitti.ps1 --sequence $seq --max-frames $MaxFrames
  } elseif ([string]::IsNullOrWhiteSpace($MaxFrames)) {
    .\run_kitti.ps1 --sequence $seq --configs $Configs
  } else {
    .\run_kitti.ps1 --sequence $seq --configs $Configs --max-frames $MaxFrames
  }
}
