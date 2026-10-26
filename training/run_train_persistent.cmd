@echo off
set PYTHON=C:\Users\BRENT\Downloads\handson_fixed\.venv-convert\Scripts\python.exe
set TRAIN=C:\Users\BRENT\Downloads\handson_fixed\handson\training\train.py
set DS=C:\Users\BRENT\Downloads\handson_fixed\combined_dataset.json
set OUT=C:\Users\BRENT\Downloads\handson_fixed\handson\training
%PYTHON% %TRAIN% --dataset %DS% --out %OUT% --epochs 300