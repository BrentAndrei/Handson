$env:TF_CPP_MIN_LOG_LEVEL=3
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "C:\Users\BRENT\Downloads\handson_fixed\.venv-convert\Scripts\python.exe"
$psi.Arguments = "C:\Users\BRENT\Downloads\handson_fixed\handson\training\run_convert_detached.py C:\Users\BRENT\Downloads\handson_fixed\handson\training\train_out_merged\fsl_model.h5 C:\Users\BRENT\Downloads\handson_fixed\handson\training\train_out_merged\tfjs_model_merged"
$psi.UseShellExecute = $true
$p = [System.Diagnostics.Process]::Start($psi)
"Started detached converter PID: $($p.Id)"