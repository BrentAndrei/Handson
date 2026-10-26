$env:TF_CPP_MIN_LOG_LEVEL=3
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "C:\Users\BRENT\Downloads\handson_fixed\.venv-convert\Scripts\python.exe"
$psi.Arguments = "C:\Users\BRENT\Downloads\handson_fixed\handson\training\regen_labels.py"
$psi.UseShellExecute = $true
$p = [System.Diagnostics.Process]::Start($psi)
"Started regen_labels PID: $($p.Id)"