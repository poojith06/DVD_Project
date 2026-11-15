# Tune resnet_mlp for the two OFF cases: 7 single-setting changes from the defaults (the defaults are runs/tune_A0).
# Three lanes run at once on the GPU. Then choose_resnet.py picks the setting on validation.
#   powershell -ExecutionPolicy Bypass -File tune_resnet.ps1        (from step2_training\code; outputs in runs\, logs\)
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$lanes = @(
    @(@("ep1000", "epochs=1000"), @("lr5e-4", "lr=5e-4"), @("wd0", "weight_decay=0")),
    @(@("bs64", "batch_size=64"), @("hub002", "huber_beta=0.02")),
    @(@("w512", "width=512"), @("b5", "blocks=5"))
)
$jobs = foreach ($lane in $lanes) {
    $cmds = ($lane | ForEach-Object {
        "python -u train.py --models resnet_mlp --cases nmos_off pmos_off --model-params `"$($_[1])`" --tag rn_$($_[0]) *> logs\rn_$($_[0]).log"
    }) -join "; "
    Start-Job -ScriptBlock { param($dir, $c) Set-Location $dir; Invoke-Expression $c } -ArgumentList $PSScriptRoot, $cmds
}
Write-Output "$(Get-Date -Format 'HH:mm') started $($jobs.Count) lanes"
$jobs | Wait-Job | Out-Null
Write-Output "$(Get-Date -Format 'HH:mm') tuning runs finished"
& python -u choose_resnet.py *> logs\choose_resnet.log
Get-Content logs\choose_resnet.log
