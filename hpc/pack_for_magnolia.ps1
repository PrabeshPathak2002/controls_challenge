# Pack code+model for Magnolia upload (excludes data/ — copy separately).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$out = Join-Path $root "controls_challenge_magnolia_code.tar.gz"
if (Test-Path $out) { Remove-Item $out }

# Prefer tar (Windows 10+)
tar -czf $out `
  build_steer_lookup.py `
  tinyphysics.py `
  requirements.txt `
  controllers `
  steer_opt `
  hpc `
  models/tinyphysics.onnx

Write-Host "Wrote $out"
Write-Host "Upload with FileZilla/scp to magnolia.mcsr.olemiss.edu:"
Write-Host "  scp $out prabeshpathak@magnolia.mcsr.olemiss.edu:~/"
Write-Host "Then on Magnolia:"
Write-Host "  mkdir -p controls_challenge && tar -xzf ~/controls_challenge_magnolia_code.tar.gz -C controls_challenge"
Write-Host "Also upload/sync the data/ folder (large)."
Write-Host "SSH: ssh prabeshpathak@magnolia.mcsr.olemiss.edu"
