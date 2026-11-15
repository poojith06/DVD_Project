# Same steps as run_all.ps1, but in parallel: several network trainings share the GPU (one small MLP uses only
# ~35 % of it) and GPR runs on the CPU at the same time. From inside ml:
#   powershell -ExecutionPolicy Bypass -File run_parallel.ps1
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

# 3. selection, Part B, Part C
foreach ($s in @(@("select_best.py", "--tag", "final"), @("evaluate_gates.py"), @("evaluate_cla.py"))) {
    & python -u @s *> "logs\$($s[0].Replace('.py','')).log"
    if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $($s[0])"; exit 1 }
}
Write-Output "$(Get-Date -Format 'HH:mm') done"
