$env:TF_CPP_MIN_LOG_LEVEL=3
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "C:\Users\BRENT\Downloads\handson_fixed\.venv-convert\Scripts\python.exe"
$psi.Arguments = "C:\Users\BRENT\Downloads\handson_fixed\handson\training\train.py --dataset C:\Users\BRENT\Downloads\handson_fixed\combined_dataset.json --out C:\Users\BRENT\Downloads\handson_fixed\handson\training --epochs 300"
$psi.UseShellExecute = $true
$p = [System.Diagnostics.Process]::Start($psi)
"Started detached PID: $($p.Id)"