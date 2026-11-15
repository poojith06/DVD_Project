# Wait for every resnet tuning run to finish, choose the setting on validation, build runs/final_tuned and write
# selection + Part B + Part C to results_tuned\ (git-ignored scratch; results\ and the submitted files stay as they are).
# Run from step2_training\code:  powershell -ExecutionPolicy Bypass -File finish_tuned.ps1
Set-Location $PSScriptRoot
$gates = "..\..\..\partB_gates\step4_ml_prediction\evaluate_gates.py"
$cla = "..\..\..\partC_cla\step7_ml_prediction\evaluate_cla.py"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -match "train.py") {
    Start-Sleep -Seconds 30
}
Write-Output "$(Get-Date -Format 'HH:mm') all training finished"
& python -u choose_resnet.py *> logs\choose_resnet.log; Get-Content logs\choose_resnet.log
& python -u assemble_tuned.py
$env:ML_V2_RESULTS = "results_tuned"
& python -u select_best.py --tag final_tuned *> logs\select_best_tuned.log
& python -u $gates --models-dir results_tuned --out results_tuned\gate_results *> logs\evaluate_gates_tuned.log
& python -u $cla --predictions results_tuned\gate_results\gate_predictions.csv --out results_tuned\cla_results *> logs\evaluate_cla_tuned.log
Remove-Item Env:ML_V2_RESULTS
Get-Content logs\select_best_tuned.log | Select-Object -First 5
Get-Content logs\evaluate_gates_tuned.log | Select-Object -Last 2
Get-Content logs\evaluate_cla_tuned.log | Select-Object -Last 3
Write-Output "$(Get-Date -Format 'HH:mm') done"
