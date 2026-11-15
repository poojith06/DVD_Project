# Deliverable 9: run every step that regenerates the deliverables, in order (Windows / PowerShell).
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Fast        # saved models and saved SPICE data (minutes)
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -SkipSpice   # re-train, keep saved SPICE data
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -SkipTrain   # re-run SPICE, keep saved models
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1              # everything (SPICE + training, ~10+ hours)
# -Fast = -SkipSpice -SkipTrain. All outputs go to $Out (default pipeline_out\); the submitted files in
# results and the data folders are never overwritten. See REPRODUCE.md for the step list and seeds.
param([switch]$Fast, [switch]$SkipSpice, [switch]$SkipTrain, [switch]$Yes,
      [string]$Out = "pipeline_out", [string]$Ngspice = "", [string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($Fast) { $SkipSpice = $true; $SkipTrain = $true }
if (-not ($SkipSpice -and $SkipTrain) -and -not $Yes) {
    $a = Read-Host "Slow steps selected (SPICE regeneration and/or training, many hours). Type YES to continue"
    if ($a -ne "YES") { Write-Output "aborted"; exit 1 }
}
$Out = (New-Item -ItemType Directory -Force $Out).FullName
$FinalModels = Join-Path $PSScriptRoot "results\models"

function Step($n, $title) { Write-Output ""; Write-Output "=== STEP $n : $title ($(Get-Date -Format 'HH:mm'))" }
function Run($script, [string[]]$arguments) {
    & $Python -u $script @arguments
    if ($LASTEXITCODE -ne 0) { throw "failed: $script" }
}
function FindNgspice {
    if ($Ngspice) { return $Ngspice }
    foreach ($c in "C:\Program Files\Spice64\bin\ngspice_con.exe", "C:\Program Files (x86)\Spice64\bin\ngspice_con.exe") { if (Test-Path $c) { return $c } }
    foreach ($n in "ngspice_con", "ngspice") { if (Get-Command $n -ErrorAction SilentlyContinue) { return $n } }
    throw "ngspice not found: pass -Ngspice <path to ngspice_con.exe>"
}

# ---------------------------------------------------------------- 1. Part A datasets (SPICE)
Step 1 "Part A datasets: 4 single-transistor netlists, VDD = 1 V, rndseed 13 / 12 / 14 / 15"
if ($SkipSpice) { Write-Output "skipped: using the saved partA_transistor\data\*.csv (10,000 rows each)" }
else {
    $ng = FindNgspice; New-Item -ItemType Directory -Force (Join-Path $Out "spice\model") | Out-Null; Copy-Item model\22nm_HP.pm (Join-Path $Out "spice\model") -Force
    $d = Join-Path $Out "spice\partA_transistor"; New-Item -ItemType Directory -Force "$d\netlists", "$d\data" | Out-Null
    Copy-Item partA_transistor\netlists\*.cir "$d\netlists" -Force
    Push-Location "$d\netlists"
    foreach ($c in "nmos_on", "nmos_off", "pmos_on", "pmos_off") { & $ng -b "${c}_ngspice.cir" | Out-Null }
    Pop-Location
    Write-Output "regenerated CSVs in $d\data (compare with partA_transistor\data; the random stream depends on the ngspice build)"
}

# ---------------------------------------------------------------- 2. Part A data check (deliverable 3)
Step 2 "Part A data check: Kirchhoff residuals and trend plots"
Run validation_checks\data_check.py @("--out", (Join-Path $Out "partA_data_check"))

# ---------------------------------------------------------------- 3 + 4. training, comparison, resnet OFF tuning
$Models = $FinalModels
if ($SkipTrain) {
    Step 3 "Training and model comparison (80:10:10, split seed 42, 4 model types x 4 cases)"
    Write-Output "skipped: using the saved models in $FinalModels and the saved tables in results\partA"
    Step 4 "Resnet OFF tuning and selection on validation"
    Write-Output "skipped (saved result: results\partA\resnet_tuning.csv, chosen setting blocks=5)"
} else {
    Step 3 "Training and model comparison (80:10:10, split seed 42, model seed 0, 4 model types x 4 cases)"
    Push-Location ml
    foreach ($A in 0, 10, 30) { Run train.py @("--models", "resnet_mlp", "--cases", "nmos_off", "pmos_off", "--rail-weight", "$A", "--tag", "tune_A$A", "--force") }
    Run choose_weight.py @()
    $W = (Get-Content runs\chosen_rail_weight.txt).Trim()
    foreach ($M in "resnet_mlp", "hybrid_analytic_mlp", "pinn", "gpr") { Run train.py @("--models", $M, "--rail-weight", $W, "--tag", "final", "--force") }
    Step 4 "Resnet OFF tuning (7 single-setting variants) and selection on validation, then the final model set"
    $variants = [ordered]@{ "ep1000" = "epochs=1000"; "lr5e-4" = "lr=5e-4"; "wd0" = "weight_decay=0"; "bs64" = "batch_size=64";
                            "hub002" = "huber_beta=0.02"; "w512" = "width=512"; "b5" = "blocks=5" }
    foreach ($k in $variants.Keys) {
        Run train.py @("--models", "resnet_mlp", "--cases", "nmos_off", "pmos_off", "--model-params", $variants[$k], "--tag", "rn_$k", "--force")
    }
    Run choose_resnet.py @()
    Run assemble_tuned.py @()
    $env:ML_V2_RESULTS = (Join-Path $Out "results_tuned")
    Run select_best.py @("--tag", "final_tuned")
    Remove-Item Env:ML_V2_RESULTS
    Pop-Location
    # select_best copies the chosen model of each case to <results>\<case>\model.joblib; evaluate from a flat folder
    $Models = Join-Path $Out "models"; New-Item -ItemType Directory -Force $Models | Out-Null
    foreach ($c in "nmos_on", "nmos_off", "pmos_on", "pmos_off") { Copy-Item (Join-Path $Out "results_tuned\$c\model.joblib") (Join-Path $Models "$c.joblib") -Force }
}

# ---------------------------------------------------------------- 5. Part B
Step 5 "Part B: gate SPICE (5 netlists, rndseed = 16) and ML gate evaluation"
if ($SkipSpice) { Write-Output "skipped: using the saved partB_gates\data\*.csv (3,600 rows)" }
else {
    $ng = FindNgspice; New-Item -ItemType Directory -Force (Join-Path $Out "spice\model") | Out-Null; Copy-Item model\22nm_HP.pm (Join-Path $Out "spice\model") -Force; $d = Join-Path $Out "spice\partB_gates"; New-Item -ItemType Directory -Force "$d\netlists", "$d\data" | Out-Null
    Copy-Item partB_gates\netlists\* "$d\netlists" -Force
    Push-Location "$d\netlists"
    foreach ($g in "inv", "nand2", "nor2", "and2", "or2") { & $ng -b "${g}_ngspice.cir" | Out-Null }
    Pop-Location
    $env:GATE_DATA_DIR = "$d\data"
    Write-Output "evaluating the regenerated gate data in $d\data"
}
Run ml\evaluate_gates.py @("--models-dir", $Models, "--out", (Join-Path $Out "gate_results"))
Remove-Item Env:GATE_DATA_DIR -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- 6. Part C
Step 6 "Part C: CLA SPICE (rndseed = 16, 64 vectors x 200 corners) and ML evaluation (plain gate sum)"
if ($SkipSpice) { Write-Output "skipped: using the saved partC_cla\data\cla3.csv (12,800 rows)" }
else {
    $ng = FindNgspice; New-Item -ItemType Directory -Force (Join-Path $Out "spice\model") | Out-Null; Copy-Item model\22nm_HP.pm (Join-Path $Out "spice\model") -Force; $d = Join-Path $Out "spice\partC_cla"; New-Item -ItemType Directory -Force "$d\netlists", "$d\data" | Out-Null
    Copy-Item "partC_cla\netlists\*" "$d\netlists" -Force
    Push-Location "$d\netlists"; & $ng -b cla3_ngspice.cir | Out-Null; Pop-Location
    $env:CLA_DIR = $d
    Write-Output "evaluating the regenerated CLA data in $d\data"
}
Run ml\evaluate_cla.py @("--predictions", (Join-Path $Out "gate_results\gate_predictions.csv"), "--out", (Join-Path $Out "cla_results"))
Remove-Item Env:CLA_DIR -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- 7. check against the submitted results
Step 7 "Compare with the submitted results"
if ($SkipSpice -and $SkipTrain) { Run ml\reproduce_check.py @("--out", $Out) }
else { Write-Output "regenerated data or models: numbers can differ from the submitted ones (ngspice build, hardware); see $Out" }
Write-Output "`ndone ($(Get-Date -Format 'HH:mm')), outputs in $Out"
