# Training run that produced the first final model set (step 2): rail-weight tuning, the four model types on the four
# cases, selection, and a Part B / Part C check. Several network trainings share the GPU (one small MLP uses only
# ~35 % of it) and GPR runs on the CPU at the same time. From inside step2_training\code:
#   powershell -ExecutionPolicy Bypass -File run_parallel.ps1
# Outputs: runs\, results\ (selection, gate_results, cla_results) - all git-ignored scratch; the submitted files are
# never overwritten.
# Logs: logs\<step>.log (stdout) and logs\<step>.err (warnings / errors)
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null

function Launch($log, [string[]]$arguments) {
    Write-Host "$(Get-Date -Format 'HH:mm') start $log"
    $p = Start-Process -FilePath python -ArgumentList (@("-u") + $arguments) -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput "logs\$log.log" -RedirectStandardError "logs\$log.err" -NoNewWindow -PassThru
    $null = $p.Handle          # keeps the exit code readable after the process ends
    return $p
}
function WaitAll($procs) {
    foreach ($p in $procs) {
        $p.WaitForExit()
        if ($p.ExitCode -ne 0) { Write-Output "FAILED (exit $($p.ExitCode)): see logs"; exit 1 }
    }
}

# GPR on the CPU (it does not use the near-rail weight, so it can start right away)
$env:GPR_DEVICE = "cpu"; $env:GPR_THREADS = "8"
$gpr = Launch "final_gpr" @("train.py", "--models", "gpr", "--tag", "final")
Remove-Item Env:GPR_DEVICE, Env:GPR_THREADS

# 1. tuning runs for the near-rail weight, all three at once on the GPU
$tune = foreach ($A in 0, 10, 30) {
    Launch "tune_A$A" @("train.py", "--models", "resnet_mlp", "--cases", "nmos_off", "pmos_off",
                        "--rail-weight", "$A", "--tag", "tune_A$A")
}
WaitAll $tune
& python -u choose_weight.py *> logs\choose_weight.log
$A = (Get-Content runs\chosen_rail_weight.txt).Trim()
Write-Output "$(Get-Date -Format 'HH:mm') chosen rail weight $A"

# 2. the three network models on all four cases, at once on the GPU
$final = foreach ($M in "resnet_mlp", "hybrid_analytic_mlp", "pinn") {
    Launch "final_$M" @("train.py", "--models", $M, "--rail-weight", $A, "--tag", "final")
}
WaitAll ($final + @($gpr))

# 3. selection, Part B, Part C (evaluated on the selected models in results\, outputs in results\)
$gates = "..\..\..\partB_gates\step4_ml_prediction\evaluate_gates.py"
$cla = "..\..\..\partC_cla\step7_ml_prediction\evaluate_cla.py"
foreach ($s in @(@("select_best.py", "--tag", "final"),
                 @($gates, "--models-dir", "results", "--out", "results\gate_results"),
                 @($cla, "--predictions", "results\gate_results\gate_predictions.csv", "--out", "results\cla_results"))) {
    $name = [IO.Path]::GetFileNameWithoutExtension($s[0])
    & python -u @s *> "logs\$name.log"
    if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $name"; exit 1 }
}
Write-Output "$(Get-Date -Format 'HH:mm') done"
