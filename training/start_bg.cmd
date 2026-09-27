@echo off
cd /d "C:\Users\BRENT\Downloads\handson_fixed\handson\training"
start /b "" "C:\Users\BRENT\Downloads\handson_fixed\handson\training\.venv-train\Scripts\python.exe" train.py --dataset fsl_dataset_guiron_merged.json --out model_guiron1 --epochs 30 > training_log.txt 2>&1
