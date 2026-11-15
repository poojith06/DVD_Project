# Full ML_Models_v2 pipeline on Windows (same steps as run_all.sh). From inside ML_Models_v2:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
# Logs: logs\<step>.log
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null

function Step($log, [string[]]$cmd) {
    Write-Output "$(Get-Date -Format 'HH:mm') start $log"
    & python -u @cmd *> "logs\$log.log"
    if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $log (see logs\$log.log)"; exit 1 }
}

# 1. near-rail training weight: tuning runs (resnet_mlp, OFF cases), chosen on validation only
foreach ($A in 0, 10, 30) {
    Step "tune_A$A" @("train.py", "--models", "resnet_mlp", "--cases", "nmos_off", "pmos_off",
                      "--rail-weight", "$A", "--tag", "tune_A$A")
}
Step "choose_weight" @("choose_weight.py")
$A = (Get-Content runs\chosen_rail_weight.txt).Trim()

# 2. all four model types on all four cases (80:10:10 split) with the chosen weight
foreach ($M in "resnet_mlp", "hybrid_analytic_mlp", "pinn", "gpr") {
    Step "final_$M" @("train.py", "--models", $M, "--rail-weight", $A, "--tag", "final")
}

# 3. best model per case on validation -> results\ ; 4. Part B ; 5. Part C
Step "select_best" @("select_best.py", "--tag", "final")
Step "evaluate_gates" @("evaluate_gates.py")
Step "evaluate_cla" @("evaluate_cla.py")
Write-Output "$(Get-Date -Format 'HH:mm') done"
