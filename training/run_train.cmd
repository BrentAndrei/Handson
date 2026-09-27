@echo off
cd /d "C:\Users\BRENT\Downloads\handson_fixed\handson\training"
"C:\Users\BRENT\Downloads\handson_fixed\handson\training\.venv-train\Scripts\python.exe" train.py --dataset fsl_dataset_guiron_merged.json --out model_guiron1 --random-split > training_log.txt 2>&1
