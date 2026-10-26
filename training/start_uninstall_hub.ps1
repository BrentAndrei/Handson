$env:TF_CPP_MIN_LOG_LEVEL=3
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "C:\Users\BRENT\Downloads\handson_fixed\.venv-convert\Scripts\python.exe"
$psi.Arguments = "-m pip uninstall -y tensorflow_hub"
$psi.UseShellExecute = $true
$p = [System.Diagnostics.Process]::Start($psi)
"Started uninstall PID: $($p.Id)"