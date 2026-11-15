# One-off helper: wait until the CPU GPR run has saved its current case, then stop it and train the remaining GPR
# cases on the GPU (frees the CPU for the network jobs that share the GPU). Log: logs\final_gpr_gpu.log
param([int]$CpuGprPid, [string]$DoneCase = "nmos_on", [string[]]$Remaining = @("nmos_off", "pmos_on", "pmos_off"))
Set-Location $PSScriptRoot
$done = "runs\final\gpr\saved_models\$DoneCase.joblib"
while (-not (Test-Path $done)) { Start-Sleep -Seconds 30 }
Start-Sleep -Seconds 20                                   # let it finish writing the metrics of that case
Stop-Process -Id $CpuGprPid -Force -ErrorAction SilentlyContinue
Write-Output "$(Get-Date -Format 'HH:mm') $DoneCase saved; CPU GPR stopped; GPR on the GPU for $($Remaining -join ', ')"
& python -u train.py --models gpr --cases @Remaining --tag final *> logs\final_gpr_gpu.log
Write-Output "$(Get-Date -Format 'HH:mm') GPR on GPU finished (exit $LASTEXITCODE)"
