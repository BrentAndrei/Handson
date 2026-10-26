$env:TF_CPP_MIN_LOG_LEVEL=3
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "C:\Users\BRENT\Downloads\handson_fixed\.venv-convert\Scripts\python.exe"
$psi.Arguments = "-m pip install tensorflowjs"
$psi.UseShellExecute = $true
$p = [System.Diagnostics.Process]::Start($psi)
"Started pip install PID: $($p.Id)"