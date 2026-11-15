# Wait for every resnet tuning run to finish, choose the setting on validation, build runs/final_tuned and write
# selection + Part B + Part C to results_tuned/ (results/ stays as it is).
Set-Location $PSScriptRoot
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -match "train.py") {
    Start-Sleep -Seconds 30
}
Write-Output "$(Get-Date -Format 'HH:mm') all training finished"
& python -u choose_resnet.py *> logs\choose_resnet.log; Get-Content logs\choose_resnet.log
& python -u assemble_tuned.py
$env:ML_V2_RESULTS = "results_tuned"
& python -u select_best.py --tag final_tuned *> logs\select_best_tuned.log
& python -u evaluate_gates.py *> logs\evaluate_gates_tuned.log
& python -u evaluate_cla.py *> logs\evaluate_cla_tuned.log
Remove-Item Env:ML_V2_RESULTS
Get-Content logs\select_best_tuned.log | Select-Object -First 5
Get-Content logs\evaluate_gates_tuned.log | Select-Object -Last 2
Get-Content logs\evaluate_cla_tuned.log | Select-Object -Last 3
Write-Output "$(Get-Date -Format 'HH:mm') done"
