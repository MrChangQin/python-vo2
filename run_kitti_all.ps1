param(
  [string]$Configs = ''
)

$ErrorActionPreference = 'Stop'

$sequences = @('00','01','02','03','04','05','06','07','08','09','10')

foreach ($seq in $sequences) {
  if ([string]::IsNullOrWhiteSpace($Configs)) {
    .\run_kitti.ps1 --sequence $seq
  } else {
    .\run_kitti.ps1 --sequence $seq --configs $Configs
  }
}
