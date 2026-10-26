$env:TF_CPP_MIN_LOG_LEVEL=3
$env:TF_NUM_INTRAOP_THREADS=1
$env:TF_NUM_INTEROP_THREADS=1
$env:OMP_NUM_THREADS=1
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "C:\Users\BRENT\Downloads\handson_fixed\.venv-convert\Scripts\python.exe"
$psi.Arguments = "C:\Users\BRENT\Downloads\handson_fixed\handson\training\run_train_detached.py C:\Users\BRENT\Downloads\handson_fixed\handson\training\combined_dataset_merged.json C:\Users\BRENT\Downloads\handson_fixed\handson\training\train_out_merged 300"
$psi.UseShellExecute = $true
$p = [System.Diagnostics.Process]::Start($psi)
"Started detached launcher PID: $($p.Id)"